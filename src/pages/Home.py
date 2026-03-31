import streamlit as st
import src.pages.Licence_and_disclaimer as legal

@st.dialog('⚠️ Medical disclaimer', dismissible=False)
def dialog_important_info():
    st.markdown(legal.get_medical_disclaimer_markdown())
    st.markdown('''
                **For licences, please see "Documentation".**
                ''')
                
    if st.button("Read and approved"):
        st.session_state['diclaimer_read'] = True
        st.rerun()

if 'diclaimer_read' not in st.session_state or not st.session_state['diclaimer_read']:
    dialog_important_info()

st.title("Brain Tumor Detector")

st.markdown('''
            ## Application goal
            The aim of this project is to develop a computer vision model. This model will be used to diagnose the presence or absence of a brain tumour, as well as its type, using MRI images.
            
            The dataset used for AI development is available on [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
            
            Further information about the project can be found in the documentation section.
            
            
            ## About me
            My name is Jérémy Duval, MSc in AI, PhD in health and I am data scientist.
            
            You can contact me on [LinkedIn](https://www.linkedin.com/in/j%C3%A9r%C3%A9my-duval-phd/)
            and follow my work on [GitHub](https://github.com/Jeremy-Duval-PhD)
            and [OrcID](https://orcid.org/0000-0001-6037-5486).
            ''')
            
st.image('assets/banner.jpg')