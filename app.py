import streamlit as st
import numpy as np
import joblib
import pandas as pd

from descriptor_utils import read_xyz, find_donors, compute_descriptors


# ------------------------------------------------------------
# PAGE SETTINGS
# ------------------------------------------------------------
st.set_page_config(
    page_title="Co Magnetic Predictor",
    page_icon="🧲",
    layout="centered"
)

# ------------------------------------------------------------
# CUSTOM COLOR STYLE
# ------------------------------------------------------------
st.markdown("""
<style>

.main-title{
    font-size:40px;
    font-weight:700;
    color:#0e76a8;
}

.section{
    font-size:24px;
    font-weight:600;
    color:#ff4b4b;
    margin-top:20px;
}

.result-card{
    background-color:#f0f7ff;
    padding:20px;
    border-radius:10px;
    border:1px solid #c9e2ff;
    margin-bottom:10px;
}

.stButton>button {
    background-color:#0e76a8;
    color:white;
    border-radius:8px;
    height:3em;
    width:100%;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------
st.markdown(
    '<p class="main-title">🧲 Three-Coordinate Co(II) Magnetic Anisotropy Predictor</p>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# MODEL ERRORS
# ------------------------------------------------------------
ERR_D = 12.5
ERR_ED = 0.05
ERR_gx = 0.08
ERR_gy = 0.09
ERR_gz = 0.1


# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
st.markdown('<p class="section">Upload Structure</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload XYZ file of the complex",
    type=["xyz"]
)


# ------------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------------
if uploaded_file is not None:

    atoms, coords = read_xyz(uploaded_file)

    co_index, donors, message = find_donors(atoms, coords)

    if message:
        st.warning(message)
        st.stop()

    donor_indices = [d[0] for d in donors]

    BL, BA = compute_descriptors(coords, co_index, donor_indices)


    # --------------------------------------------------------
    # DONOR TABLE
    # --------------------------------------------------------
    st.markdown('<p class="section">Detected Donor Atoms</p>', unsafe_allow_html=True)

    donor_table = []

    for i, d in enumerate(donor_indices):
        donor_table.append({
            "Donor Atom Index": d + 1,
            "Atom": atoms[d],
            "Co–L Bond Length (Å)": round(BL[i], 3)
        })

    st.dataframe(pd.DataFrame(donor_table), use_container_width=True)


    # --------------------------------------------------------
    # CONFIRMATION
    # --------------------------------------------------------
    confirm = st.radio(
        "Are these donor atoms correct?",
        ["Yes", "No"],
        index=None
    )

    run_prediction = False


    # --------------------------------------------------------
    # AUTO MODE
    # --------------------------------------------------------
    if confirm == "Yes":
        run_prediction = True


    # --------------------------------------------------------
    # MANUAL MODE
    # --------------------------------------------------------
    elif confirm == "No":

        manual = st.text_input(
            "Enter donor atom indices separated by comma (example: 12,34,56)"
        )

        if manual:

            try:

                donor_indices = [int(x.strip()) - 1 for x in manual.split(",")]

                BL, BA = compute_descriptors(coords, co_index, donor_indices)

                st.markdown('<p class="section">Updated Donor Atoms</p>', unsafe_allow_html=True)

                donor_table = []

                for i, d in enumerate(donor_indices):
                    donor_table.append({
                        "Donor Atom Index": d + 1,
                        "Atom": atoms[d],
                        "Co–L Bond Length (Å)": round(BL[i], 3)
                    })

                st.dataframe(pd.DataFrame(donor_table), use_container_width=True)

                confirm2 = st.radio(
                    "Proceed with prediction?",
                    ["Yes", "No"],
                    index=None
                )

                if confirm2 == "Yes":
                    run_prediction = True

            except:
                st.error("Invalid atom indices. Please enter valid numbers.")


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------
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


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------
        st.markdown('<p class="section">Predicted Magnetic Parameters</p>', unsafe_allow_html=True)


        col1, col2, col3 = st.columns(3)

        col1.metric("D", f"{round(D,3)}", f"± {ERR_D}")
        col2.metric("E/D", f"{round(ED,4)}", f"± {ERR_ED}")
        col3.metric("gx", f"{round(gx,3)}", f"± {ERR_gx}")

        col4, col5 = st.columns(2)

        col4.metric("gy", f"{round(gy,3)}", f"± {ERR_gy}")
        col5.metric("gz", f"{round(gz,3)}", f"± {ERR_gz}")


        st.caption(
            "Prediction uncertainty corresponds to model MAE on the test dataset."
        )

        st.markdown(
            "🔗 **Reference:** "
            "[ChemRxiv DOI](https://doi.org/10.26434/chemrxiv-2024-97555)"
        )
