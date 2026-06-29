"""Smoke tests for the Kedro project registration."""

from pathlib import Path

import yaml
from kedro.framework.startup import bootstrap_project

from home_credit_mlops.pipeline_registry import register_pipelines


class TestKedroProject:
    def test_all_pipelines_are_registered(self):
        """Every named pipeline must be registered and non-empty."""
        bootstrap_project(Path.cwd())
        pipelines = register_pipelines()

        required = [
            "__default__",
            "data_quality",
            "data_split",
            "data_cleaning",
            "data_feat_engineering",
            "model_selection",
            "model_train",
            "model_predict",
            "data_drifts",
        ]
        for name in required:
            assert name in pipelines, f"Pipeline '{name}' not registered"
            assert len(pipelines[name].nodes) > 0, f"Pipeline '{name}' has no nodes"

    def test_default_pipeline_chains_all_stages(self):
        """Default pipeline must include nodes from every sub-pipeline."""
        bootstrap_project(Path.cwd())
        pipelines = register_pipelines()
        default_node_names = {n.name for n in pipelines["__default__"].nodes}

        # Each sub-pipeline must contribute at least one node to the default run
        for name in ["data_quality", "data_split", "data_cleaning",
                     "data_feat_engineering", "model_selection", "model_train"]:
            sub_node_names = {n.name for n in pipelines[name].nodes}
            overlap = default_node_names & sub_node_names
            assert overlap, (
                f"Pipeline '{name}' contributes no nodes to __default__"
            )

    def test_all_pipeline_inputs_are_catalogued(self):
        """Every non-parameter input must be declared in catalog.yml.

        Undeclared inputs become MemoryDatasets that vanish between runs —
        this test catches wiring mistakes before a full pipeline run.
        """
        bootstrap_project(Path.cwd())
        pipelines = register_pipelines()
        catalog_raw = yaml.safe_load(
            Path("conf/base/catalog.yml").read_text(encoding="utf-8")
        )
        catalog_names = set(catalog_raw or {})

        missing_by_pipeline: dict[str, list[str]] = {}
        for name, pipe in pipelines.items():
            if name == "__default__":
                continue
            missing = sorted(
                inp
                for inp in pipe.inputs()
                if not inp.startswith("params:") and inp not in catalog_names
            )
            if missing:
                missing_by_pipeline[name] = missing

        assert missing_by_pipeline == {}, (
            f"Uncatalogued inputs found: {missing_by_pipeline}"
        )

    def test_model_train_outputs_are_catalogued(self):
        """Outputs produced by model_train must be in the catalog (not MemoryDataset)."""
        bootstrap_project(Path.cwd())
        pipelines = register_pipelines()
        catalog_raw = yaml.safe_load(
            Path("conf/base/catalog.yml").read_text(encoding="utf-8")
        )
        catalog_names = set(catalog_raw or {})

        must_be_persisted = [
            "production_model",
            "decision_threshold",
            "model_metrics",
            "shap_importance",
        ]
        for name in must_be_persisted:
            assert name in catalog_names, (
                f"'{name}' must be in catalog.yml so it survives between runs"
            )
