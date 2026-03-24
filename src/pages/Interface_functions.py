import streamlit as st
import numpy as np
from pathlib import Path
import yaml
import tensorflow as tf
from PIL import Image


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing configuration file: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_config():
    if 'config' not in st.session_state:
        st.session_state['config'] = load_config(st.session_state['config_path'])

@st.cache_resource
def get_preproc_model():
    set_config()
    return tf.saved_model.load(str(st.session_state['artefact_dir']))


def get_files_names(files):
    names = []
    
    for file in files:
        names.append(file.name)
        
    return names


def session_clear():
    st.session_state['clean_files'] = []
    st.session_state['file_names'] = []
    
    
def init_session_state_var():
    if 'clean_files' not in st.session_state:
        st.session_state['clean_files'] = []
    if 'file_names' not in st.session_state:
        st.session_state['file_names'] = []
    if 'project_root' not in st.session_state:
        st.session_state['project_root'] = Path(__file__).resolve().parents[2]
        st.session_state['config_path'] = st.session_state['project_root'] / "config.yaml"
        set_config()
        st.session_state['artefact_dir'] = \
            st.session_state['project_root'] / st.session_state['config']["path"]["models_dir"] / "preproc_pipeline"
    
    
def load_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    return np.array(img)
    
    
def get_tf_dataset(preproc_model, images, labels):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(lambda img, lbl: preproc_model(img, lbl),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds


def get_clean_tfrecords(ds):
    def serialize_example(img, label):
        img_bytes = tf.io.serialize_tensor(img).numpy()

        example = tf.train.Example(
            features=tf.train.Features(feature={
                "image": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[img_bytes])
                ),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label.numpy()])
                ),
            })
        )
        return example.SerializeToString()

    # Collect TFRecords in memory
    tfrecords = []
    for img, lbl in ds:
        tfrecords.append(serialize_example(img, lbl))

    return tfrecords
    
    
def preprocess_files(uploaded_files, file_names):    
    if 'clean_files' not in st.session_state:
        new_files = file_names
    else:
        new_files = np.setdiff1d(file_names, st.session_state['clean_files'])
        
    images = [load_image(f) for f in new_files]
    images = [tf.convert_to_tensor(img) for img in images]
    labels = [-1 for i in range(0,len(images))]
    
    model = get_preproc_model()
    ds = get_tf_dataset(model, images, labels)
    preproc_img = get_clean_tfrecords(ds)
        
    return preproc_img
    

def set_uploaded_data(section):
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
        
        if 'clean_files' not in st.session_state \
        or file_names != st.session_state['file_names']:
            init_session_state_var() 
            clean_files = preprocess_files(uploaded_files, file_names)
            st.session_state['clean_files'] = clean_files
            st.session_state['file_names'] = file_names
        
        
    if cntr.button("Clear", type="primary", disabled=\
    (not uploaded_files) and ('clean_files' not in st.session_state or not st.session_state['clean_files'])):
        st.session_state.uploader_key += 1
        if 'clean_files' in st.session_state:
            session_clear()
        st.rerun()