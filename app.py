import streamlit as st
import numpy as np
import joblib
import pandas as pd

from descriptor_utils import read_xyz, find_donors, compute_descriptors

# ==========================================================
# Page setup
# ==========================================================

st.set_page_config(
    page_title="Co Magnetic Predictor",
    layout="centered"
)

st.title("Three Coordinate Co(II) Magnetic Anisotropy Predictor")

# ==========================================================
# RF uncertainty function
# ==========================================================

def predict_with_uncertainty(model, X):

    tree_predictions = np.array([
        tree.predict(X)[0]
        for tree in model.estimators_
    ])

    prediction = np.mean(tree_predictions)

    uncertainty = np.std(tree_predictions)

    return prediction, uncertainty


# ==========================================================
# Upload XYZ
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload XYZ file",
    type=["xyz"]
)

if uploaded_file is not None:

    try:

        atoms, coords = read_xyz(uploaded_file)

        co_index, donors, message = find_donors(
            atoms,
            coords
        )

        if message:
            st.error(message)
            st.stop()

        donor_indices = [d[0] for d in donors]

        BL, BA = compute_descriptors(
            coords,
            co_index,
            donor_indices
        )

        X = np.array([[
            BL[0],
            BL[1],
            BL[2],
            BA[0],
            BA[1],
            BA[2]
        ]])

        # ==================================================
        # Load models
        # ==================================================

        model_D = joblib.load(
            "models/RF_model_D.joblib"
        )

        model_ED = joblib.load(
            "models/RF_model_E_D.joblib"
        )

        model_gx = joblib.load(
            "models/RF_model_gx.joblib"
        )

        model_gy = joblib.load(
            "models/RF_model_gy.joblib"
        )

        model_gz = joblib.load(
            "models/RF_model_gz.joblib"
        )

        # ==================================================
        # Predictions with uncertainty
        # ==================================================

        D, ERR_D = predict_with_uncertainty(
            model_D,
            X
        )

        ED, ERR_ED = predict_with_uncertainty(
            model_ED,
            X
        )

        gx, ERR_gx = predict_with_uncertainty(
            model_gx,
            X
        )

        gy, ERR_gy = predict_with_uncertainty(
            model_gy,
            X
        )

        gz, ERR_gz = predict_with_uncertainty(
            model_gz,
            X
        )

        # ==================================================
        # Results
        # ==================================================

        st.subheader(
            "Predicted Magnetic Parameters"
        )

        results = pd.DataFrame({

            "Parameter": [
                "D (cm⁻¹)",
                "E/D",
                "gx",
                "gy",
                "gz"
            ],

            "Prediction": [

                f"{D:.2f} ± {ERR_D:.2f}",

                f"{ED:.4f} ± {ERR_ED:.4f}",

                f"{gx:.3f} ± {ERR_gx:.3f}",

                f"{gy:.3f} ± {ERR_gy:.3f}",

                f"{gz:.3f} ± {ERR_gz:.3f}"
            ]
        })

        st.table(results)

        st.markdown(
            "For more details visit: "
            "[https://pubs.acs.org/doi/full/10.1021/acs.inorgchem.6c01031](https://pubs.acs.org/doi/full/10.1021/acs.inorgchem.6c01031)"
        )

    except Exception as e:

        st.error(
            f"Error during prediction: {e}"
        )
