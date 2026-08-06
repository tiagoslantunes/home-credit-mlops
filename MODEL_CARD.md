# Model Card: Home Credit Default Risk

## Model overview

This proof of concept estimates the probability that a Home Credit applicant
will default. The end-to-end system uses a calibrated LightGBM classifier,
Kedro orchestration, MLflow experiment tracking and model registration, SHAP
explanations, batch scoring, and a FastAPI serving layer.

The project demonstrates MLOps practices. It is not a production lending
system and must not be used to approve, reject, price, or otherwise make real
credit decisions.

## Ownership and provenance

The system was created as a 2026 NOVA IMS group project by Alexandra Varela,
Francisca Fernandes, Mariana Melo, Rui Ferreira, and Tiago Antunes. The
[upstream repository](https://github.com/marianamelo0/home-credit-mlops) is the
source of record. Tiago's original contribution covers data splitting, model
selection, model training, MLflow, Optuna, and SHAP.

## Data

- **Source:** Kaggle's Home Credit Default Risk competition.
- **Target:** `TARGET`, where `1` denotes a default and `0` repayment.
- **Class balance:** approximately 8% positive examples in the project data.
- **Access:** users must accept Kaggle's competition rules and download the
  eight source CSV files themselves. Raw data is not redistributed here.
- **Split policy:** stratified train/validation splitting occurs before
  cleaning and feature engineering to reduce leakage risk.

The dataset represents historical lending data. It may encode past
institutional practices, missingness patterns, geographic effects, and other
proxies that should not be assumed fair or stable in a new population.

## Model and decision policy

- **Estimator:** LightGBM with isotonic probability calibration.
- **Feature contract:** 51 engineered features for the production endpoint.
- **Decision threshold:** 0.093 in the reported experiment.
- **Threshold objective:** cost-sensitive selection that places greater weight
  on missed defaults than false positives.
- **Explainability:** global SHAP importance, with permutation fallback for the
  calibrated estimator.

## Reported validation results

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7600 |
| PR-AUC | 0.2447 |
| F1 at the selected threshold | 0.2758 |
| Recall at the selected threshold | 0.6491 |
| Precision at the selected threshold | 0.1751 |

These values describe one project validation split and are not evidence of
performance on another country, lender, time period, or applicant population.
Reproduce the pipeline before relying on the numbers.

## Intended uses

- Demonstrating modular MLOps architecture and reproducible pipelines.
- Exploring data-quality gates, experiment tracking and model monitoring.
- Testing batch/online scoring consistency in a controlled environment.
- Teaching or reviewing credit-risk modelling patterns.

## Out-of-scope uses

- Real lending, eligibility, pricing or adverse-action decisions.
- Fully automated decisions about individuals.
- Deployment without independent legal, security, privacy and fairness review.
- Interpreting SHAP values as causal explanations.

## Limitations and risks

- No fairness audit or subgroup performance report is included.
- The validation split is not a temporal or external validation set.
- Drift checks detect distribution change but do not prove model degradation.
- The raw API endpoint uses fallbacks for target-encoded features when encoder
  maps are unavailable; those fallbacks can reduce fidelity.
- The example API does not provide authentication, authorization, rate
  limiting, TLS termination or audit logging.
- Model and cleaning artefacts are intentionally absent from version control.
- Hopsworks is optional and requires separate credentials and a dedicated
  dependency environment; the base pipeline remains functional when the API
  key is not configured.

## Monitoring recommendations

Track input schema, missingness, PSI/Evidently drift, score distribution,
calibration, recall and precision when labels arrive. Define alert ownership,
rollback criteria and a retraining approval process before deployment. Monitor
subgroups agreed with legal and risk stakeholders, not only aggregate metrics.

## Reproducibility

Follow the README to obtain the data, create the environment and run
`kedro run`. MLflow records search and training runs locally. The public CI
validates syntax, notebooks and data-independent tests; API contract tests run
after the pipeline has produced the required local artefacts.
