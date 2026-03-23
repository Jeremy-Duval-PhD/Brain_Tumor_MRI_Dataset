import streamlit as st
from src.pages.Interface_functions import get_uploaded_data

st.title("Analyse your MRI")

get_uploaded_data(st)

st.divider()

# TODO : 
# SHAP
# in expender : fix metrics to help interpr : base on val_ds ?