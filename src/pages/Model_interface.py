import streamlit as st
import pandas as pd
from src.pages.Interface_functions import set_uploaded_data, set_model_visualisation
import src.pages.Licence_and_disclaimer as legal

if 'diclaimer_read' not in st.session_state or not st.session_state['diclaimer_read']:
    legal.dialog_important_info()

st.title("Analyse your MRI")

uploader_section = st.empty()

st.divider()

process_section = st.empty()

progress_bar = process_section.progress(0.0, text="")
st.session_state["progress_bar"] = progress_bar # to use only one progress bar along the process

# """ functions """ 
set_uploaded_data(uploader_section)

set_model_visualisation(process_section)