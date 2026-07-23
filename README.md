# AI4I Predictive Maintenance

Predictive Maintenance: End-to-End ML System Design and Deployment

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
