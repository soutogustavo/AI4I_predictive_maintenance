# Models

This folder contains the serialized model artifact produced by the training pipeline (`predictive_maintenance.train`). Nothing here is created manually.

## `pipeline.joblib`

A single serialized object (via `joblib`) containing everything needed for inference, without retraining:

- **`pipeline`**: the fitted scikit-learn `Pipeline` (preprocessing + LightGBM model)
- **`threshold`**: the decision threshold selected to meet the target recall (see Part 1 of the main README for the reasoning behind this choice)
- **`target_recall`**: the recall target used to derive that threshold, kept for traceability
- **`feature_names`**: post-preprocessing feature names, used to align SHAP explanations at serving time

## Regenerating this file

This file is not guaranteed to exist on a fresh clone (it is a generated artifact, not source code). Run the training pipeline to (re)create it:

```bash
python -m predictive_maintenance.train
```

The web app expects `pipeline.joblib` to exist here before it can serve predictions.