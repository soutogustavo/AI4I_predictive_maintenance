# AI4I Predictive Maintenance

Predictive Maintenance: End-to-End ML System Design and Deployment


## Setup Instructions

### Prerequisites
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) for dependency management

### Clone and install

```bash
git clone git@github.com:soutogustavo/AI4I_predictive_maintenance.git
cd ai4i_predictive_maintenance
uv sync
```

This installs all dependencies listed in `pyproject.toml`, including the project's
own package (`predictive_maintenance`, under `src/`).

### Run the training pipeline

The trained model artifact (`models/pipeline.joblib`) is committed to the repository
for convenience (see Part 3), but can be regenerated from scratch:

```bash
uv run python -m predictive_maintenance.train
```

This will:
- Load and preprocess the data (`data/ai4i2020.csv`)
- Train the LightGBM model inside a scikit-learn `Pipeline`
- Evaluate it on a held-out test set (PR-AUC, precision, recall)
- Generate evaluation figures in `reports/figures/`
- Serialize the fitted pipeline, threshold, and metadata to `models/pipeline.joblib`

### Run the web app locally

```bash
uv run streamlit run app/streamlit_app.py
```

This opens the app at `http://localhost:8501`. Requires `models/pipeline.joblib`
to exist (either committed, or regenerated via the command above).

### Live deployment

The app is deployed on Streamlit Community Cloud: **https://ips-predictive-maintenance.streamlit.app/**

## Problem Framing and System Design

### 1. Problem Statement, Delivery, Success, and Cost Asymmetry

**Problem statement:**
Maintenance engineers don't have tools to prioritize which machines should be
inspect first. Assuming they have a limited time and staff, the priorization
tends to be reactive, that is, when they visualize obvious symptoms or
fixed schedules rather than by an actual estimate of failure risk.
The proposed tool addresses that gap by coming up a ranked, explainable
risk score per machine. Thus, the engineers can focus attention where it matters most.

**How predictions are made available:**
The predictions are made available in a dashboard for the maintenance engineers.
It should list machines ranked by predicted failure risk as well as a
visual risk indicator and a short explanation of the main contribution factors
for each prediction (e.g. abnormal torque, elevated tool wear), so engineers
get a reason to act on, not just a score.

**What success looks like:**
Success is not maximizing model accuracy. It is whether the tool measurably
improves how engineers prioritize their limited inspection time compared
to the current reactive or fixed-schedule approach. In practice, that means
high-risk machines are surfaced early enough to act on, explanations are
trusted and actionable, and the false-positive rate stays low enough that
engineers don't start ignoring alerts, a well-calibrated system used
consistently is more valuable than a highly accurate one that gets tuned out.

**Cost of false negative vs. false positive:**
A false negative, predicting a machine is safe when it is about to fail,
carries a high cost: unplanned downtime, potential equipment damage, and
possible safety risk. A false positive carries a lower but non-trivial cost:
wasted inspection time and, if frequent, lack of trust in the system, which
can lead engineers to ignore alerts entirely.

---

### 2. System Design Diagram

See the system design diagram can be found in [miro board](https://miro.com/app/board/uXjVH5YN8Cs=/?share_link_id=640983449595).

**Components absent here, but required in production:**
- Continuous data ingestion pipeline from real sensors (vs. static CSV)
- **Input schema validation** at serving time (this challenge's data is clean and synthetic; real sensor data arrives with missing values, out-of-range readings, and malformed records that must be caught before they reach the model)
- Scheduled or triggered retraining pipeline
- Data and model drift monitoring
- Model versioning and experiment tracking (e.g. MLflow)
- Prediction logging for auditability
- A feedback loop where confirmed outcomes (machine actually failed or not) feed back into future training data
- Authentication/access control on the dashboard
- Alerting/notification system beyond the dashboard itself (e.g. email/Slack for high-risk machines)

---

### 3. Model Selection Strategy and Trade-offs

**Model selection strategy:**
Given the tabular, structured nature of the sensor data, gradient
boosted trees (LightGBM) were selected: they handle nonlinear interactions
between features like torque and rotational speed without manual feature engineering,
perform well on relatively small tabular datasets, and train fast enough to
iterate quickly. Thus, following the principle of keeping things simple
(i.e. specially for the first versions of a project).

**Interpretability vs. performance trade-off:**
The task prioritizes system design and explainability instead of maximum accuracy,
which directly informs this trade-off. The users of this tool are maintenance engineers,
not data scientists. Therefore, they need actionable reason attached to each risk score,
not just a probability. LightGBM paired with SHAP can deliver human-readable explanations
for each prediction (e.g. "elevated torque and abnormal power draw" for a specific flagged machine).


## Model Training and Evaluation

### Metric Justification

The predictive mainenance data is heavly imbalanced. Only 3.39% represents machine failure (i.e. 339 of 10k observations). Therefore, accuracy is not meaningful metric for this case. If we predict "no failure" for all observations, we would get 96.61% accuracy.

**PR-AUC (average precision)** was chosen as the main evaluation metric. Unlike ROC-AUC, PR-AUC focuses on how well the model trades off precision and recall across all thresholds for the minority class, which is the class that actually matters operationally. The model we trained for this problem achieved a **PR-AUC of 0.90** on the held-out test set.

### Threshold Selection and Operating Point

The selection of the classification threshold is critical for ensuring the model delivers business value. Since the cost of a false negative (a missed failure) is much higher than the cost of a false positive (inspecting a healthy machine), we cannot rely on the default 0.5 threshold. Instead, we evaluated three operating points on the precision-recall curve to find the best trade-off between
detecting failures and minimizing false alarms.

Three candidate operating points were evaluated:

| Target Recall | Threshold | Actual Recall | Precision | False Positives (FP) | False Negatives (FN) |
|---|---|---|---|---|---|
| 85% | 0.414 | 85.3% | 78.4% | 16 | 10 |
| 90% | 0.086 | 91.2% | 54.4% | 52 | 6 |
| 95% | 0.002 | 95.6% | 21.7% | 234 | 3 |

After evaluating the three operating points, the 85% recall operating point was selected as the default.

It's important to mention that pushing from 90% to 95% recall captures only 3 additional true failures, at the cost of 182 additional false positives (from 52 to 234 FP). A false
positive rate this high risks alert fatigue, engineers may start avoiding the tool if too many alerts turn out to be false alarms. This would neutralize the tool's value regardless of its recall.

See `reports/figures/precision_recall_curve.png`
and `reports/figures/precision_recall_vs_threshold.png` for the full curves.

Regarding the threshold, this should be a decision made together with the operations team, informed by the real cost of an inspection versus the real cost of a missed failure.

The role of this analysis is to make that trade-off explicit and visible, not to impose a single "correct" answer.

### Preventing Preprocessing Leakage

Preprocessing (one-hot encoding of `Type`) and the model are combined into a single
`sklearn.pipeline.Pipeline`, wrapped around a `ColumnTransformer`. This is what
guarantees, by construction rather than by manual discipline, that:

- `.fit()` learns the encoding vocabulary and model parameters **only from the
  training set**
- `.transform()` / `.predict_proba()` on the test set **reuses** that
  already-learned encoding, without re-fitting anything

The use of stratified sampling is crucial given the imbalanced nature of the dataset. A non-stratified split risks producing a test set with a meaningfully different failure rate than the training set, which would distort every metric above.

Therefore, the pipeline approach ensures that preprocessing steps are learned only from the training data and applied consistently to the test data, preventing data leakage and ensuring reliable model evaluation.


## Web App Deployment

**Live app:** https://ips-predictive-maintenance.streamlit.app/

### Output Design

The app accepts a single machine reading (Type, air/process temperature, rotational
speed, torque, tool wear) entered manually by the user, and returns:

1. **Failure risk score**: the model's predicted probability of failure
2. **Binary flag**: whether the reading crosses the operating threshold selected
   in Part 2 (~85% recall on held-out test data)
3. **Top contributing factors**: the three features with the largest SHAP impact
   on that specific prediction, with direction (increases/decreases risk)

This combination was chosen over a bare probability or flag alone because the
end user, a maintenance engineer, needs a reason to act on, not just a score
(see Part 1, "*how predictions are made available?*").

### Input Design Decision

The app takes a given machine readings as input and runs the failure risk
as described above. This decision was made because the dataset does not
contain persistent machine identity across rows, each row is an independent
reading, not a machine tracked over time. In a real deployment scenario,
the input would come from a live sensor feed tied to a tracked machine.

### Deployment

Deployed on **Streamlit Community Cloud**, connected directly to the project's
GitHub repository (`app/streamlit_app.py` as the entry point).

Both the raw dataset (`data/ai4i2020.csv`) and the trained model
artifact (`models/pipeline.joblib`) are committed to the repository,
in support of full reproducibility from a clean install and zero-setup
deployment on Streamlit Community Cloud, which only has access to the
repository's contents.

### Manual Validation

Before considering the app ready, 10 real readings from the dataset (5 confirmed
`Machine failure = 0`, 5 confirmed `Machine failure = 1`) were run directly against
`predict_risk()` and cross-checked against the app's UI.


## Production Readiness Critique

### 1. Monitoring

There are two metrics that should be tracked:

- **Prediction distribution shift**: track the distribution of predicted risk
  scores over time (e.g. daily histogram or rolling mean/percentiles). A sudden
  shift, more readings scoring near the threshold, or a spike in flagged
  machines, signals either a real change in fleet behavior or a data/model
  problem, and should trigger investigation either way.
- **Input data drift**: track the distribution of each input feature (especially
  torque, tool wear, and the derived `power` feature, the strongest predictors)
  against the training distribution. Unfortunately, a model trained on this data will silently
  degrade if incoming readings start falling outside the ranges it learned from.

A third metric that should be tracked once real outcomes become available is **realized precision/recall
against confirmed outcomes** (did a flagged machine actually fail, did an unflagged
one). This is the only metric that validates the model is still doing its job, not
just that its inputs/outputs look statistically similar to training time.

### 2. Retraining Strategy

I would retrain on a **fixed schedule** (e.g. monthly) rather than purely
trigger-based, since trigger-based retraining depends on drift detection being
reliable, which itself needs to mature over time. Scheduled retraining gives a
predictable baseline; drift alerts would additionally trigger an out-of-cycle
retrain if degradation is severe.

Before replacing the live model, the candidate model would be validated by:
- Evaluating on a held-out test set using the same metric (PR-AUC) and threshold
  selection process (target-recall-driven) used originally, not just comparing
  raw accuracy.
- A **Shadow deployment** could also work. This means running the new model
  alongside the current one on live traffic, logging both predictions without acting on the new model's output.
  Thus, comparing real-world behavior before promote it to production.

### 3. Risks and Mitigations

- **Synthetic training data**: this model was trained on synthetic data (AI4I 2020),
  not real sensor readings. Real deployment would very likely show a performance
  gap.<br>
  *Mitigation*: treat the current model as a starting point, not a
  production-ready model, and require validation on real historical data before
  any real rollout.

- **No persistent machine identity in the data**: the model predicts risk from a
  point-in-time reading, not from a tracked machine's history, so it cannot use
  trend signals (e.g. gradually increasing tool wear) that a fleet-based
  production system should have.<br>
  *Mitigation*: as real data becomes available,
  extend the schema to track machine identity over time and add trend-based
  features.

- **Alert fatigue**: the model is tuned for recall over precision (approx. 78%
  precision at the chosen threshold), so roughly 1 in 5 alerts will be a false
  alarm. If unmanaged, engineers may start ignoring alerts.<br>
  *Mitigation*: monitor
  realized precision over time, and treat the threshold as adjustable in
  collaboration with the operations team, not fixed permanently.

- **No input validation on the live app**: the current app trusts whatever the
  user enters; it doesn't check for physically implausible values (e.g. negative
  torque).<br>
  *Mitigation*: add basic input range validation before this is used
  beyond a demo/prototype context.
