import streamlit as st
import numpy as np
import joblib
import pandas as pd

from descriptor_utils import read_xyz, find_donors, compute_descriptors

# --------------------------------------------------

# Page configuration

# --------------------------------------------------

st.set_page_config(
page_title="Co Magnetic Predictor",
layout="centered",
page_icon="🧲"
)

# --------------------------------------------------

# Custom CSS styling

# --------------------------------------------------

st.markdown("""

<style>

.title-box{
    background: linear-gradient(90deg,#3a7bd5,#00d2ff);
    padding:18px;
    border-radius:10px;
    text-align:center;
    color:white;
    font-size:32px;
    font-weight:700;
}

.section{
    background-color:#f7f9ff;
    padding:15px;
    border-radius:10px;
    border:1px solid #e2e6ff;
    margin-bottom:10px;
}

</style>

""", unsafe_allow_html=True)

# --------------------------------------------------

# Title

# --------------------------------------------------

st.markdown(
'<div class="title-box">Three Coordinate Co(II) Magnetic Anisotropy Predictor</div>',
unsafe_allow_html=True
)

st.write("")

# --------------------------------------------------

# Model uncertainties

# --------------------------------------------------

ERR_D = 12.5
ERR_ED = 0.05
ERR_gx = 0.08
ERR_gy = 0.09
ERR_gz = 0.10

# --------------------------------------------------

# Upload structure

# --------------------------------------------------

st.markdown("### 📂 Upload XYZ Structure")

uploaded_file = st.file_uploader("Upload XYZ file", type=["xyz"])

if uploaded_file is not None:

```
atoms, coords = read_xyz(uploaded_file)

co_index, donors, message = find_donors(atoms, coords)

if message:
    st.warning(message)
    st.stop()

donor_indices = [d[0] for d in donors]

BL, BA = compute_descriptors(coords, co_index, donor_indices)


# --------------------------------------------------
# Detected donors
# --------------------------------------------------
st.markdown("### 🧪 Detected Donor Atoms")

donor_table = []

for i, d in enumerate(donor_indices):
    donor_table.append({
        "Donor atom index": d + 1,
        "Atom": atoms[d],
        "Co–L bond length (Å)": round(BL[i], 3)
    })

df = pd.DataFrame(donor_table)

st.dataframe(df, use_container_width=True)


# --------------------------------------------------
# Confirmation
# --------------------------------------------------
st.markdown("### ✔ Confirm Donor Atoms")

confirm = st.radio(
    "Are these donor atoms correct?",
    ["Yes", "No"],
    index=None
)

run_prediction = False


# --------------------------------------------------
# If donors are correct
# --------------------------------------------------
if confirm == "Yes":
    run_prediction = True


# --------------------------------------------------
# Manual donor selection
# --------------------------------------------------
elif confirm == "No":

    manual = st.text_input(
        "Enter donor atom indices separated by comma (example: 12,34,56)"
    )

    if manual:

        try:

            donor_indices = [int(x.strip()) - 1 for x in manual.split(",")]

            BL, BA = compute_descriptors(coords, co_index, donor_indices)

            st.markdown("### 🔄 Updated Donor Atoms")

            donor_table = []

            for i, d in enumerate(donor_indices):

                donor_table.append({
                    "Donor atom index": d + 1,
                    "Atom": atoms[d],
                    "Co–L bond length (Å)": round(BL[i], 3)
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


# --------------------------------------------------
# Run prediction
# --------------------------------------------------
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


    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    st.markdown("### 🧲 Predicted Magnetic Parameters")

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

    st.success("Prediction completed successfully!")

    st.dataframe(results, use_container_width=True)

    st.caption("Prediction uncertainty corresponds to model MAE on the test dataset.")

    st.markdown(
        "📄 For more details visit: https://doi.org/10.26434/chemrxiv-2024-97555"
    )
```
