import streamlit as st
import numpy as np
import joblib
import pandas as pd

from descriptor_utils import read_xyz, extract_descriptors

# Page setup
st.set_page_config(
    page_title="Co Magnetic Predictor",
    layout="centered"
)

st.title("Co(II) Magnetic Property Predictor")

# Upload file
uploaded_file = st.file_uploader("Upload XYZ file", type=["xyz"])


if uploaded_file is not None:

    try:

        # Read XYZ
        atoms, coords = read_xyz(uploaded_file)

        # Extract descriptors
        desc = extract_descriptors(atoms, coords)

        # Prepare ML input
        X = np.array([[
            desc["BL1"],
            desc["BL2"],
            desc["BL3"],
            desc["BA1"],
            desc["BA2"],
            desc["BA3"]
        ]])

        # Load models
        model_D = joblib.load("models/GB_model_D.joblib")
        model_ED = joblib.load("models/GB_model_E_D.joblib")
        model_gx = joblib.load("models/GB_model_gx.joblib")
        model_gy = joblib.load("models/GB_model_gy.joblib")
        model_gz = joblib.load("models/GB_model_gz.joblib")

        # Predictions
        D = model_D.predict(X)[0]
        ED = model_ED.predict(X)[0]
        gx = model_gx.predict(X)[0]
        gy = model_gy.predict(X)[0]
        gz = model_gz.predict(X)[0]

        # Model errors (replace with your MAE)
        err_D = 3.20
        err_ED = 0.015
        err_gx = 0.03
        err_gy = 0.04
        err_gz = 0.05

        # Results
        st.subheader("Predicted Magnetic Parameters")

        results = pd.DataFrame({
            "Parameter": ["D", "E/D", "gx", "gy", "gz"],
            "Predicted Value": [
                round(D,3),
                round(ED,4),
                round(gx,3),
                round(gy,3),
                round(gz,3)
            ],
            "± Error": [
                err_D,
                err_ED,
                err_gx,
                err_gy,
                err_gz
            ]
        })

        st.table(results)

    except Exception as e:

        st.error(f"Error processing structure: {e}")
