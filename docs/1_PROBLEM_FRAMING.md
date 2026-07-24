
PART 1 — PROBLEM FRAMING AND SYSTEM DESIGN

1. Write a brief problem statement (1 paragraph). How will the predictions be made
available to users? What does success look like? How would you define the cost of a
false negative versus a false positive in this operational context?

2. Make a system design diagram showing the full pipeline: data ingestion, training,
serving, and monitoring. Note the components you would add in a production setting that
are absent from this challenge.

3. Describe your model selection strategy. What trade-offs did you consider between
interpretability and performance?

-------------------------------------------------------------------------------

ANSWERS:

1. Maintenance engineers currently have no systematic way to prioritize which
machines to inspect first, given limited time and staff. Without a risk-based
view, prioritization tends to be reactive, driven by obvious symptoms or
fixed schedules, rather than by an actual estimate of failure risk. This tool
addresses that gap by surfacing a ranked, explainable risk score per machine, so
engineers can focus attention where it matters most.

Predictions are exposed through a web dashboard that lists machines ranked by
predicted failure risk, along with a visual risk indicator (e.g. low/medium/high)
and a short explanation of the main contributing factor for each prediction
(e.g. abnormal torque, elevated tool wear). This gives engineers not just a score,
but a reason to act on it.

Success is not maximizing model accuracy. It is whether the tool measurably
improves how engineers prioritize their limited inspection time compared to the
current reactive or fixed-schedule approach. In practice, that means: high-risk
machines are surfaced early enough to act on, the explanation attached to each
score is trusted and actionable by the maintenance team, and the false-positive
rate is low enough that engineers don't start ignoring the alerts, a
well-calibrated system used consistently is more valuable than a highly
accurate one that gets tuned out.

A false negative, predicting a machine is safe when it is about to fail, carries
a high cost: unplanned downtime, potential equipment damage, and possible
safety risk. A false positive, flagging a machine that would not have failed,
carries a lower but non-trivial cost: wasted inspection time and, if it happens
often enough, erosion of trust in the system, which can lead engineers to
disregard alerts altogether, effectively neutralizing the tool. Given this
asymmetry, the system should be tuned to favor recall over precision, but
within limits that are actively monitored, since precision that's too low is
its own failure mode.

2.

Diagram was created on Miro board.

3.

Given this is tabular, structured sensor data, gradient boosted trees (LightGBM
or XGBoost) are the natural choice: they handle nonlinear interactions between
features like torque and rotational speed without manual feature engineering,
they perform well on relatively small tabular datasets, and they train fast
enough to iterate quickly, which matters given the challenge's time constraint.
This also aligns with prior production experience applying the same class of model
to industrial time-series and tabular problems at scale.

The task description explicitly states the emphasis is not on maximizing accuracy,
which directly informs this trade-off: a more complex model (e.g. a neural
network) might offer marginal performance gains but would cost more in tuning
time, require more data to justify its complexity, and reduce interpretability,
which matters here because the tool's end users are maintenance engineers, not
data scientists, who need to trust and act on a reason, not just a score.
Gradient boosted trees paired with SHAP values offer a strong balance:
competitive performance on tabular data, plus per-prediction explanations that
map to something an engineer can act on directly (e.g. "this machine's torque is
unusually high for its operating type"). This was a deliberate choice to prioritize
explainability given the stated goals of the tool, not a limitation of familiarity
with more complex approaches.
