import streamlit as st
import numpy as np
import joblib
import pandas as pd

from descriptor_utils import read_xyz, find_donors, compute_descriptors

st.set_page_config(page_title="Co Magnetic Predictor", layout="centered")
st.title("Co(II) Magnetic Property Predictor")

ERR_D = 12.5
ERR_ED = 0.05
ERR_gx = 0.08
ERR_gy = 0.09
ERR_gz = 0.1

uploaded_file = st.file_uploader("Upload XYZ file", type=["xyz"])

if uploaded_file is not None:

    atoms, coords = read_xyz(uploaded_file)

    co_index, donors, message = find_donors(atoms, coords)

    if message:
        st.warning(message)
        st.stop()

    donor_indices = [d[0] for d in donors]

    BL, BA = compute_descriptors(coords, co_index, donor_indices)

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

    if confirm == "Yes":
        run_prediction = True

    elif confirm == "No":

        manual = st.text_input(
            "Enter donor atom indices separated by comma (example: 12,34,56)"
        )

        if manual:

            try:

                donor_indices = [int(x.strip()) - 1 for x in manual.split(",")]

                BL, BA = compute_descriptors(coords, co_index, donor_indices)

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

            except:
                st.error("Invalid atom indices. Please enter valid numbers.")

    if run_prediction:

        X = np.array([[BL[0], BL[1], BL[2], BA[0], BA[1], BA[2]]])

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

        results = pd.DataFrame({
            "Parameter": ["D", "E/D", "gx", "gy", "gz"],
            "Prediction": [
                f"{round(D,3)} ± {ERR_D}",
                f"{round(ED,4)} ± {ERR_ED}",
                f"{round(gx,3)} ± {ERR_gx}",
                f"{round(gy,3)} ± {ERR_gy}",
                f"{round(gz,3)} ± {ERR_gz}"
            ]
        })

        st.table(results)

        st.caption("Prediction uncertainty corresponds to model MAE on the test dataset.")

        st.markdown(
            "For more details visit: "
            "[https://doi.org/10.26434/chemrxiv-2024-97555](https://doi.org/10.26434/chemrxiv-2024-97555)"
        )
