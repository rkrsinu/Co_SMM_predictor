import streamlit as st
import numpy as np
import joblib
import pandas as pd

from descriptor_utils import read_xyz, find_donors, compute_descriptors

st.set_page_config(page_title="Co Magnetic Predictor",layout="centered")

st.title("Co(II) Magnetic Property Predictor")

uploaded_file = st.file_uploader("Upload XYZ file", type=["xyz"])

if uploaded_file:

    atoms,coords = read_xyz(uploaded_file)

    co_index, donors = find_donors(atoms,coords)

    donor_indices=[d[0] for d in donors]

    BL,BA = compute_descriptors(coords,co_index,donor_indices)

    st.subheader("Detected donor atoms")

    donor_display=[]

    for idx,d in enumerate(donor_indices):

        donor_display.append({
            "Donor atom index":d+1,
            "Atom":atoms[d],
            "Bond length (Å)":round(BL[idx],3)
        })

    st.table(pd.DataFrame(donor_display))

    confirm = st.radio("Are these donor atoms correct?",["Yes","No"])

    if confirm=="No":

        manual = st.text_input(
        "Enter donor atom indices separated by comma (example: 10,25,41)")

        if manual:

            donor_indices=[int(x.strip())-1 for x in manual.split(",")]

            BL,BA = compute_descriptors(coords,co_index,donor_indices)

            st.write("Updated bond lengths:")

            st.write(BL)

            confirm2 = st.radio("Proceed with prediction?",["Yes","No"])

            if confirm2=="Yes":
                proceed=True
            else:
                proceed=False
        else:
            proceed=False

    else:
        proceed=True


    if proceed:

        X = np.array([[BL[0],BL[1],BL[2],BA[0],BA[1],BA[2]]])

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
            "Parameter":["D","E/D","gx","gy","gz"],
            "Value":[round(D,3),round(ED,4),round(gx,3),round(gy,3),round(gz,3)]
        })

        st.table(results)

        st.markdown(
        "For more details visit: "
        "[https://doi.org/10.26434/chemrxiv-2024-97555](https://doi.org/10.26434/chemrxiv-2024-97555)"
        )
