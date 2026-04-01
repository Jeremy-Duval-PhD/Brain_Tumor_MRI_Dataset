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


details_expander = process_section.expander('Global metrics for interpretability')
details_expander.markdown('''
            # Confidence intervals
            ## Head *tumor presence*
            ''')
                
conf_interval_pres = pd.DataFrame(
    {
        "Value": ["0.9886", "0.9927", "0.9915", "0.9921"],
        "95% CI": ["[0.9806, 0.9933]", "[0.9867, 0.9976]", "[0.9849, 0.9965]", "[0.9876, 0.9958]"],
    },
    index=["Accuracy", "Precision", "Recall", "F1"],
)
details_expander.table(conf_interval_pres)
                
details_expander.markdown('''               
                ## Head *tumor type*
            ''')       
                
conf_interval_type = pd.DataFrame(
    {
        "Value": ["0.9721", "0.9715", "0.9715", "0.9715"],
        "95% CI": ["[0.9585, 0.9813]", "[0.9601, 0.9831]", "[0.9600, 0.9832]", "[0.9601, 0.9830]"],
    },
    index=["Accuracy", "Precision", "Recall", "F1"],
)
details_expander.table(conf_interval_type)        
            
details_expander.markdown('''                    
                ## Tumor type class recall
            ''')       
                
conf_interval_class = pd.DataFrame(
    {
        "Value": ["0.9697", "0.9552", "0.9897"],
        "95% CI": ["[0.9414, 0.9846]", "[0.9234, 0.9742]", "[0.9702, 0.9965]"],
    },
    index=["glioma", "meningioma", "pituitary"],
)
details_expander.table(conf_interval_class)  
                    
details_expander.markdown('''                    
              # Confusion matrix 
              ## Head *tumor presence*
            ''')

details_expander.image('docs/docs/figures/Confusion_mtrx_presence_af_fine_tuning.png', \
             caption='Confusion matrix for type tumor head, after fine-tuning')
              
details_expander.markdown('''
            Accuracy by class:
            - no_tumor: 98.1%
            - tumor: 99.1%
            
            ## Head *tumor type*
              ''')
              
details_expander.image('docs/docs/figures/Confusion_mtrx_type_af_fine_tuning.png', \
             caption='Confusion matrix for type tumor head, after fine-tuning')
            
details_expander.warning('''"no tumor" bad score is explained by the loss mask 
                         when the presence head predicts that there is no tumour.
                         Don\'t consider it.''')
details_expander.info('''"meningioma" score is explained by the fact that, in 
                      some cases, meningiomas can resemble gliomas, and the classification 
                      of this tumour type depends more on the MRI scan.''')

details_expander.markdown('''
            Accuracy by class:
            - glioma: 97.0%
            - meningioma: 95.5%
            - pituitary: 99.0%
            
            # Calibrations
          
              ''')  
            
conf_interval_class = pd.DataFrame(
    {
        "ECE": ["0.030", "0.080", "0.144", "0.044"],
        "Brier": ["0.012", "0.072", "0.151", "0.035"],
        "Calibration quality": [ ":green[Good]", ":green[Good]", ":yellow[Average]", ":green[Good]"],
    },
    index=["Tumor presence", "Tumor type (glioma)", "Tumor type (meningioma)", "Tumor type (pituitary)"],
)
details_expander.table(conf_interval_class) 



progress_bar = process_section.progress(0.0, text="")
st.session_state["progress_bar"] = progress_bar # to use only one progress bar along the process

# """ functions """ 
set_uploaded_data(uploader_section)

set_model_visualisation(process_section)