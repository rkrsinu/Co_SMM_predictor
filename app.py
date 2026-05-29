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
# Helper function for RF uncertainty
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

    atoms, coords = read_xyz(uploaded_file)

    co_index, donors, message = find_donors(atoms, coords)

    if message:
        st.warning(message)
        st.stop()

    donor_indices = [d[0] for d in donors]

    BL, BA = compute_descriptors(
        coords,
        co_index,
        donor_indices
    )

    # ======================================================
    # Show detected donors
    # ======================================================

    st.subheader("Detected donor atoms")

    donor_table = []

    for i, d in enumerate(donor_indices):

        donor_table.append({
            "Donor atom index": d + 1,
            "Atom": atoms[d],
            "Co–L bond length (Å)": round(BL[i], 3)
        })

    st.table(pd.DataFrame(donor_table))

    confirm = st.radio(
        "Are these donor atoms correct?",
        ["Yes", "No"],
        index=None
    )

    run_prediction = False

    # ======================================================
    # Automatic donor selection accepted
    # ======================================================

    if confirm == "Yes":
        run_prediction = True

    # ======================================================
    # Manual donor selection
    # ======================================================

    elif confirm == "No":

        manual = st.text_input(
            "Enter donor atom indices separated by comma (example: 12,34,56)"
        )

        if manual:

            try:

                donor_indices = [
                    int(x.strip()) - 1
                    for x in manual.split(",")
                ]

                if len(donor_indices) != 3:
                    st.error(
                        "Please provide exactly three donor atom indices."
                    )
                    st.stop()

                BL, BA = compute_descriptors(
                    coords,
                    co_index,
                    donor_indices
                )

                st.subheader("Updated donor atoms")

                donor_table = []

                for i, d in enumerate(donor_indices):

                    donor_table.append({
                        "Donor atom index": d + 1,
                        "Atom": atoms[d],
                        "Co–L bond length (Å)": round(BL[i], 3)
                    })

                st.table(pd.DataFrame(donor_table))

                confirm2 = st.radio(
                    "Proceed with prediction?",
                    ["Yes", "No"],
                    index=None
                )

                if confirm2 == "Yes":
                    run_prediction = True

            except Exception as e:

                st.error(
                    f"Invalid atom indices. Error: {e}"
                )

    # ======================================================
    # Prediction
    # ======================================================

    if run_prediction:

        X = np.array([[
            BL[0],
            BL[1],
            BL[2],
            BA[0],
            BA[1],
            BA[2]
        ]])

        # ==================================================
        # Load RF models
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
        # Predict with uncertainty
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
        # Display results
        # ==================================================

        st.subheader("Predicted Magnetic Parameters")

        results = pd.DataFrame({

            "Parameter": [
                "D",
                "E/D",
                "gx",
                "gy",
                "gz"
            ],

            "Prediction": [

                f"{D:.3f} ± {ERR_D:.3f}",

                f"{ED:.4f} ± {ERR_ED:.4f}",

                f"{gx:.3f} ± {ERR_gx:.3f}",

                f"{gy:.3f} ± {ERR_gy:.3f}",

                f"{gz:.3f} ± {ERR_gz:.3f}"
            ]
        })

        st.table(results)

        st.caption(
            "Uncertainty is estimated from the spread of predictions across all trees in the Random Forest ensemble."
        )

        st.markdown(
            "For more details visit: "
            "[https://doi.org/10.26434/chemrxiv-2024-97555](https://doi.org/10.26434/chemrxiv-2024-97555)"
        )
