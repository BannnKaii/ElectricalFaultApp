import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Electrical Fault Detection",
    page_icon="⚡",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "fault_models.joblib"

FEATURES = [
    "Ia", "Ib", "Ic",
    "Va", "Vb", "Vc",
    "Im", "Vm", "Pai", "Pav", "Fi", "Fv"
]

FEATURE_HELP = {
    "Ia": "Phase A current",
    "Ib": "Phase B current",
    "Ic": "Phase C current",
    "Va": "Phase A voltage",
    "Vb": "Phase B voltage",
    "Vc": "Phase C voltage",
    "Im": "Current magnitude",
    "Vm": "Voltage magnitude",
    "Pai": "Active power/current-related feature as defined in your dataset",
    "Pav": "Active power/voltage-related feature as defined in your dataset",
    "Fi": "Input/current frequency feature",
    "Fv": "Voltage frequency feature",
}

FAULT_MEANINGS = {
    "NF": "No fault / normal condition",
    "LG": "Line-to-ground fault",
    "LL": "Line-to-line fault",
    "LLG": "Double-line-to-ground fault",
    "LLL": "Three-phase fault",
    "LLLG": "Three-phase-to-ground fault",
}

# These are convenient starting values for testing the interface only.
# Replace them with actual measurements before interpreting a prediction.
DEFAULTS = {
    "Ia": -9.622, "Ib": 2.300, "Ic": 6.981,
    "Va": 0.0, "Vb": 0.0, "Vc": 0.0,
    "Im": 17913.348, "Vm": 147.466,
    "Pai": -33.716, "Pav": 0.648, "Fi": 50.0, "Fv": 50.0,
}

@st.cache_resource
def load_bundle(path_string):
    return joblib.load(path_string)

st.title("⚡ Electrical Fault Detection and Classification")
st.write(
    "Enter the 12 features used by the trained model, select an algorithm, "
    "and request a prediction."
)
st.info(
    "Use measurements in the same units and format as the training dataset. "
    "This is an educational ML demo, not a certified protection system for live electrical equipment."
)

if not MODEL_PATH.exists():
    st.error(
        "The file 'fault_models.joblib' was not found. "
        "Export the trained models from Colab and place that file beside app.py. "
        "See README.txt in the starter folder."
    )
    st.stop()

try:
    bundle = load_bundle(str(MODEL_PATH))
    models = bundle["models"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]
    saved_features = bundle["features"]
except Exception as exc:
    st.error(f"Could not load the saved model bundle: {exc}")
    st.stop()

if list(saved_features) != FEATURES:
    st.error(
        "The feature list stored in the model file does not match the feature order in app.py. "
        "Please check the export cell and notebook."
    )
    st.stop()

model_names = {
    "Support Vector Machine (SVM)": "SVM",
    "K-Nearest Neighbors (KNN)": "KNN",
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
}

with st.sidebar:
    st.header("Prediction settings")
    selected_label = st.selectbox("Choose model", list(model_names.keys()))
    selected_model_name = model_names[selected_label]
    st.caption("SVM and KNN inputs are scaled before prediction. Decision Tree and Random Forest use the original feature values.")

st.subheader("Enter electrical measurements")
st.caption("The initial values are only for testing the page layout. Replace them with actual PMU readings.")

with st.form("prediction_form"):
    cols = st.columns(3)
    values = {}

    for index, feature in enumerate(FEATURES):
        with cols[index % 3]:
            values[feature] = st.number_input(
                label=f"{feature} — {FEATURE_HELP[feature]}",
                value=float(DEFAULTS[feature]),
                format="%.6f",
                help=f"Enter {feature} using the same units and meaning as in the training dataset.",
                key=f"input_{feature}",
            )

    submitted = st.form_submit_button("Predict fault", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)

    try:
        model = models[selected_model_name]

        if selected_model_name in ("SVM", "KNN"):
            model_input = scaler.transform(input_df)
        else:
            model_input = input_df

        encoded_prediction = model.predict(model_input)
        predicted_label = str(label_encoder.inverse_transform(encoded_prediction.astype(int))[0])

        st.divider()
        st.subheader("Prediction result")

        if predicted_label == "NF":
            st.success(f"Predicted class: {predicted_label} — {FAULT_MEANINGS.get(predicted_label, predicted_label)}")
        else:
            st.warning(f"Predicted class: {predicted_label} — {FAULT_MEANINGS.get(predicted_label, predicted_label)}")

        left, right = st.columns(2)
        left.metric("Selected model", selected_model_name)
        right.metric("Predicted class", predicted_label)

        st.write("**Values submitted to the model**")
        st.dataframe(input_df, use_container_width=True, hide_index=True)

        st.caption(
            "The prediction is based on patterns learned from the training dataset. "
            "It may be unreliable for values outside the training distribution."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
