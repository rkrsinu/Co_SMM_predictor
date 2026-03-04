import streamlit as st
import numpy as np
import joblib

from descriptor_utils import read_xyz, extract_descriptors

st.title("Co(II) Magnetic Property Predictor")

st.write(
"""
Upload a **.xyz structure of a three-coordinate Co complex**.
The app extracts structural descriptors and predicts:

• D  
• E/D  
• gx  
• gy  
• gz
"""
)

uploaded_file = st.file_uploader("Upload XYZ file", type=["xyz"])

if uploaded_file is not None:

    try:

        atoms,coords = read_xyz(uploaded_file)

        desc = extract_descriptors(atoms,coords)

        st.subheader("Extracted Descriptors")

        st.write(desc)

        X = np.array([[

            desc["BL1"],
            desc["BL2"],
            desc["BL3"],
            desc["BA1"],
            desc["BA2"],
            desc["BA3"]

        ]])

        # load models
        model_D = joblib.load("models/GB_model_D.joblib")
        model_ED = joblib.load("models/GB_model_E_D.joblib")
        model_gx = joblib.load("models/GB_model_gx.joblib")
        model_gy = joblib.load("models/GB_model_gy.joblib")
        model_gz = joblib.load("models/GB_model_gz.joblib")

        D = model_D.predict(X)[0]
        ED = model_ED.predict(X)[0]
        gx = model_gx.predict(X)[0]
        gy = model_gy.predict(X)[0]
        gz = model_gz.predict(X)[0]

        st.subheader("Predicted Magnetic Parameters")

        st.write(f"D = {D:.3f}")
        st.write(f"E/D = {ED:.4f}")
        st.write(f"gx = {gx:.3f}")
        st.write(f"gy = {gy:.3f}")
        st.write(f"gz = {gz:.3f}")

    except Exception as e:

        st.error(str(e))
