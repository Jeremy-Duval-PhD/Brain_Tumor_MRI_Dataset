import streamlit as st


def get_files_names(files):
    names = []
    
    for file in files:
        names.append(file.name)
        
    return names


def get_uploaded_data(section):
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    
    cntr = section.container(border=True)
    
    help_msg = '''
                 You can upload MRI files at jpg format. 
              '''
    uploaded_files = cntr.file_uploader("Upload your MRI", 
                                       type=['jpg'],
                                       help=help_msg, 
                                       accept_multiple_files=True,
                                       key=f"uploader_{st.session_state.uploader_key}")
    
    if uploaded_files is not None:
        st.write(uploaded_files)
        file_names = get_files_names(uploaded_files)
        st.write(file_names)
        """
        if 'clean_files' not in st.session_state \
        or file_names != st.session_state['clean_files']:
            clean_files = preprocess_files(uploaded_files) # traiter que les nouveaux !
            init_all_session_state_var(clean_files, file_name)
        """
        
    if cntr.button("Clear", type="primary", disabled=\
    (not uploaded_files) and ('clean_files' not in st.session_state or not st.session_state['clean_files'])):
        st.session_state.uploader_key += 1
        if 'clean_files' in st.session_state:
            st.session_state['clean_files'] = []
        st.rerun()