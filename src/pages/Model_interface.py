import streamlit as st
from src.pages.Interface_functions import set_uploaded_data

st.title("Analyse your MRI")

set_uploaded_data(st)

st.divider()

# TODO : 
# SHAP
# in expender : fix metrics to help interpr : base on val_ds ?