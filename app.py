import streamlit as st

st.logo('assets/logo.png', size='large', link='https://github.com/Jeremy-Duval-PhD')

st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon='assets/logo.png',
)

home_page = st.Page("src/pages/Home.py")
doc_page = st.Page("src/pages/Documentation.py")
pages = [home_page, doc_page]

pg = st.navigation(pages)
pg.run()