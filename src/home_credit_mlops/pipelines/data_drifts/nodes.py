"""Nodes for the 'data_drifts' pipeline.

Three public nodes wired in order by ``pipeline.py``:

  1. ``compute_psi_metrics`` — per-feature PSI (W6 01_PSI). Returns a DataFrame
     with one row per feature: feature, psi, drift_level. Cheap and is the
     primary signal in credit-scoring monitoring (Siddiqi).
  2. ``compute_evidently_report`` — Evidently 0.7 DataDriftPreset (W6 04). The
     statistical tests complement PSI (KS for continuous, chi² for categorical)
     and the rendered HTML is the artefact analysts open. Returns
     (html_string, summary_dict).
  3. ``raise_drift_alerts`` — combines PSI + Evidently summary, emits warnings
     to the logger, and returns an alerts JSON (status: ok / warning / critical,
     list of drifted features, timestamps) — the artefact a downstream alerting
     job (Slack/email/PagerDuty) would consume in production.

Reference vs current convention:
  reference = data the model was trained on (X_train_data)
  current   = data the model is being asked to score (X_val_data here; in
              production it would be a recent serving batch)
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. PSI                                                                       #
# --------------------------------------------------------------------------- #
def _psi_single(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int,
    bucket_type: str,
) -> float:
    """Population Stability Index for a single 1-D numeric array.

    Adapted from W6/01_datadrift_PSI (mwburke implementation), with two
    corrections vs the original:
      - the ``a_perc - e_perc`` term is computed AFTER zero-replacement
        (the original used a buggy generator with ``np.sum``);
      - quantile-based bucketing uses the REFERENCE distribution so bins
        stay stable across batches.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if reference.size == 0 or current.size == 0:
        return float("nan")

    if bucket_type == "bins":
        # Even-width bins anchored on the reference min/max
        breakpoints = np.linspace(reference.min(), reference.max(), n_bins + 1)
    elif bucket_type == "quantiles":
        # Quantile bins on the reference (robust to outliers)
        quantiles = np.linspace(0, 1, n_bins + 1)
        breakpoints = np.unique(np.quantile(reference, quantiles))
        if breakpoints.size < 2:
            # Constant feature in reference → no PSI defined
            return 0.0
    else:
        raise ValueError(f"Unknown bucket_type: {bucket_type!r}")

    # Make the outer edges open so points outside reference range still bin
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_counts / max(reference.size, 1)
    cur_pct = cur_counts / max(current.size, 1)

    # Avoid log(0) — standard PSI handling
    eps = 1e-4
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _classify_psi(psi: float, small: float, major: float) -> str:
    if np.isnan(psi):
        return "undefined"
    if psi < small:
        return "none"
    if psi < major:
        return "small"
    return "major"


def compute_psi_metrics(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """Compute PSI per feature and classify drift level.

    Drops the SK_ID_CURR identifier if present (not a model feature).
    Numeric columns only; the production pipeline already encodes categoricals
    upstream, so ``X_train_data`` is fully numeric.
    """
    psi_cfg = params["psi"]
    n_bins = psi_cfg["n_bins"]
    bucket_type = psi_cfg["bucket_type"]
    small = psi_cfg["small_drift_threshold"]
    major = psi_cfg["major_drift_threshold"]

    feature_cols = [c for c in reference.columns if c != "SK_ID_CURR"]
    shared_cols = [c for c in feature_cols if c in current.columns]
    if len(shared_cols) < len(feature_cols):
        missing = set(feature_cols) - set(shared_cols)
        logger.warning("PSI: %d features missing in current; skipped: %s", len(missing), sorted(missing)[:5])

    rows = []
    for feature in shared_cols:
        psi = _psi_single(
            reference[feature].to_numpy(),
            current[feature].to_numpy(),
            n_bins=n_bins,
            bucket_type=bucket_type,
        )
        rows.append({"feature": feature, "psi": psi, "drift_level": _classify_psi(psi, small, major)})

    psi_df = pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)
    logger.info(
        "PSI computed for %d features | none=%d small=%d major=%d undefined=%d",
        len(psi_df),
        (psi_df["drift_level"] == "none").sum(),
        (psi_df["drift_level"] == "small").sum(),
        (psi_df["drift_level"] == "major").sum(),
        (psi_df["drift_level"] == "undefined").sum(),
    )
    return psi_df


# --------------------------------------------------------------------------- #
# 2. Evidently                                                                 #
# --------------------------------------------------------------------------- #
def compute_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    params: Dict[str, Any],  # noqa: ARG001 — kept for future statistical-test config
) -> Tuple[str, Dict[str, Any]]:
    """Run Evidently's DataDriftPreset and return (HTML string, summary dict).

    The HTML is the analyst-facing artefact; the summary dict feeds the alert
    node so it does not have to re-parse HTML.
    """
    from evidently import Dataset, DataDefinition, Report
    from evidently.presets import DataDriftPreset

    feature_cols = [c for c in reference.columns if c != "SK_ID_CURR" and c in current.columns]

    # Cast to the same numeric dtypes on both sides so Evidently picks the same statistical test
    ref = reference[feature_cols].copy()
    cur = current[feature_cols].copy()

    data_definition = DataDefinition(numerical_columns=feature_cols)
    ref_dataset = Dataset.from_pandas(ref, data_definition=data_definition)
    cur_dataset = Dataset.from_pandas(cur, data_definition=data_definition)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    # Render — Evidently exposes either an HTML string or saves to disk; we use the string
    # so the caller (Kedro catalog) decides where it lands.
    html = snapshot.get_html() if hasattr(snapshot, "get_html") else snapshot._repr_html_()

    summary = snapshot.dict() if hasattr(snapshot, "dict") else {}
    logger.info("Evidently report generated (%d features compared)", len(feature_cols))
    return html, summary


# --------------------------------------------------------------------------- #
# 3. Alerts                                                                    #
# --------------------------------------------------------------------------- #
def raise_drift_alerts(
    psi_metrics: pd.DataFrame,
    evidently_summary: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide overall drift status and emit alerts.

    Status policy:
      - "critical" if ANY feature has PSI ≥ major_threshold OR if the share of
        features with PSI ≥ small_threshold exceeds max_drifted_share_critical.
      - "warning" if at least one feature drifted at small level but no critical.
      - "ok" otherwise.

    Returns a JSON-serialisable dict consumed downstream by an alerting job.
    """
    alert_cfg = params["alerts"]
    psi_cfg = params["psi"]
    critical_share = alert_cfg["max_drifted_share_critical"]
    major_threshold = psi_cfg["major_drift_threshold"]

    drifted_small = psi_metrics[psi_metrics["drift_level"].isin(["small", "major"])]
    drifted_major = psi_metrics[psi_metrics["drift_level"] == "major"]
    drifted_share = len(drifted_small) / max(len(psi_metrics), 1)

    if not drifted_major.empty:
        status = "critical"
        reason = f"{len(drifted_major)} feature(s) with PSI ≥ {major_threshold}"
    elif drifted_share >= critical_share:
        status = "critical"
        reason = f"{drifted_share:.0%} of features drifted (≥ {critical_share:.0%} threshold)"
    elif not drifted_small.empty:
        status = "warning"
        reason = f"{len(drifted_small)} feature(s) with small drift"
    else:
        status = "ok"
        reason = "no drift detected"

    # Emit at the right log level so existing alerting hooks (e.g. an MLflow alert
    # hook or a sidecar tailing the log) pick it up.
    log = {"ok": logger.info, "warning": logger.warning, "critical": logger.error}[status]
    log("DRIFT ALERT [%s]: %s", status.upper(), reason)
    for row in drifted_small.itertuples():
        log(
            "  drifted: %-35s PSI=%.4f level=%s",
            row.feature,
            row.psi,
            row.drift_level,
        )

    drifted_features = [
        {"feature": r.feature, "psi": float(r.psi), "drift_level": r.drift_level}
        for r in drifted_small.itertuples()
    ]

    # Evidently dataset-level drift share, when available
    dataset_drift_share = None
    try:
        metrics = evidently_summary.get("metrics", []) if evidently_summary else []
        if metrics:
            # Best-effort: 0.7 surfaces share_of_drifted_columns or similar in the first metric
            first = metrics[0]
            result = first.get("result", first.get("value", {})) if isinstance(first, dict) else {}
            for k in ("share_of_drifted_columns", "drift_share", "share"):
                if k in result:
                    dataset_drift_share = float(result[k])
                    break
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not parse Evidently summary: %s", exc)

    return {
        "status": status,
        "reason": reason,
        "drifted_features_count": len(drifted_small),
        "major_drift_features_count": len(drifted_major),
        "drifted_share": drifted_share,
        "evidently_dataset_drift_share": dataset_drift_share,
        "drifted_features": drifted_features,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
