"""Streamlit app for predictive maintenance"""

import pandas as pd
import streamlit as st

from predictive_maintenance.inference import load_artifact, predict_risk

st.set_page_config(
    page_title="Machine Failure Risk Assessment",
    layout="centered"
)


@st.cache_resource
def get_artifact():
    """Load the trained pipeline once. It keeps the model loaded in memory."""
    return load_artifact("models/pipeline.joblib")


def main():
    st.title(":wrench: Machine Failure Risk Assessment")
    st.caption(
        "Enter a machine's current operating readings to assess its failure risk. "
        "This tool prioritizes recall (catching real failures) over precision, "
        "given the higher cost of a missed failure versus a false alarm."
    )

    artifact = get_artifact()

    with st.form("reading_form"):
        col1, col2 = st.columns(2)
        with col1:
            machine_type = st.selectbox("Machine Type", options=["L", "M", "H"])
            air_temp = st.number_input("Air temperature [K]", value=298.0, step=0.1)
            process_temp = st.number_input("Process temperature [K]", value=308.0, step=0.1)
        with col2:
            rot_speed = st.number_input("Rotational speed [rpm]", value=1500, step=10)
            torque = st.number_input("Torque [Nm]", value=40.0, step=0.5)
            tool_wear = st.number_input("Tool wear [min]", value=100, step=1)

        submitted = st.form_submit_button("Assess Risk")

    if submitted:
        raw_input = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }])

        result = predict_risk(artifact, raw_input)
        risk_score = result["risk_score"][0]
        flagged = result["flagged"][0]

        st.divider()

        col1, col2 = st.columns(2)
        col1.metric("Failure Risk Score", f"{risk_score:.2%}")
        if flagged:
            col2.error(":warning: FLAGGED: Recommend inspection")
        else:
            col2.success(":white_check_mark: Normal: No action needed")

        st.subheader("Why this prediction?")
        shap_values_row = result["shap_values"][0]
        feature_names = result["feature_names"]

        clean_names = [n.replace("num__", "").replace("cat__", "") for n in feature_names]
        top_idx = sorted(
            range(len(shap_values_row)),
            key=lambda i: abs(shap_values_row[i]),
            reverse=True,
        )[:3]

        for i in top_idx:
            direction = "increases risk :arrow_up:" if shap_values_row[i] > 0 else "decreases risk :arrow_down:"
            st.write(f"- **{clean_names[i]}**: {direction} (SHAP = {shap_values_row[i]:+.3f})")

        st.caption(
            f"Operating threshold: {artifact['threshold']:.3f} "
            f"(selected to achieve ~{artifact['target_recall']:.0%} recall on held-out test data)."
        )


if __name__ == "__main__":
    main()
