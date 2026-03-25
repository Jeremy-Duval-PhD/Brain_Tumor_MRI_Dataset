import streamlit as st
import numpy as np
from pathlib import Path
import yaml
import tensorflow as tf
from PIL import Image
import tempfile


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
    uploaded_file.seek(0)
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    return np.array(img)
    
    
def get_tf_dataset(preproc_model, images, labels):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(lambda img, lbl: preproc_model(img, lbl),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds


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


def get_clean_tfrecords(ds):

    # Collect TFRecords in memory
    tfrecords = []
    for img, lbl in ds:
        tfrecords.append(serialize_example(img, lbl))

    return tfrecords


def save_images(section, images, nb_files, filename_prefix="img"):
    count = 0
    temp_dir = Path(tempfile.mkdtemp())
    st.session_state['temp_dir_raw'] = temp_dir

    progress_text = "Saving images. Please wait."
    progress_bar = section.progress(0.0, text=progress_text)

    for idx, img in enumerate(images):
        # security for type
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        # PIL image
        pil_img = Image.fromarray(img)

        # saving
        img_path = temp_dir / f"{filename_prefix}_{idx:04d}.jpg"
        pil_img.save(img_path, format="JPG")

        count += 1

        # update progress
        progress = count / nb_files
        progress_bar.progress(progress, text=f"{progress_text} ({count}/{nb_files})")

    progress_bar.progress(1.0, text="Done ✅")



def save_tf_records(section, ds, nb_files, batch_size, filename_prefix="data"):
    writer = None
    count = 0
    file_idx = 0
    temp_dir = Path(tempfile.mkdtemp())
    st.session_state['temp_dir_preproc'] = temp_dir
    
    progress_text = "Saving images. Please wait."
    progress_bar = section.progress(0, text=progress_text)

    lbl = tf.constant(0) # default value -> use only to compare prediction and reality during model creation
    for img, lbl in ds:
        if count % batch_size == 0:
            if writer:
                writer.close()
            record_path = temp_dir / f"{filename_prefix}_{file_idx:03d}.tfrecord"
            writer = tf.io.TFRecordWriter(str(record_path))
            file_idx += 1
        writer.write(serialize_example(img, lbl))
        count += 1
        
        # update progress bar
        progress = count / nb_files
        progress_bar.progress(progress, text=f"{progress_text} ({count}/{nb_files})")

    progress_bar.progress(1.0, text="Done ✅")

    if writer:
        writer.close()
    
    
def preprocess_files(section, uploaded_files):    
    if 'clean_files' not in st.session_state:
        new_files = uploaded_files
    else:
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state['file_names']
        ]
        
    images = [load_image(f) for f in new_files]
    images = [tf.convert_to_tensor(img) for img in images]
    labels = ["none" for i in range(0,len(images))]
    nb_files = len(images)
    
    save_images(section, images, nb_files)
    
    model = get_preproc_model()
    ds = get_tf_dataset(model, images, labels)
    
    with st.spinner("Preprosessing images in progress...", show_time=True):
        preproc_img = get_clean_tfrecords(ds)
    
    batch_size = st.session_state['config']['model']['batch_size']
    save_tf_records(section, ds, nb_files, batch_size)
    
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
    
    if uploaded_files is not None and uploaded_files:
        st.write(uploaded_files)
        file_names = get_files_names(uploaded_files)
        st.write(file_names)
        
        if 'clean_files' not in st.session_state \
        or file_names != st.session_state['file_names']:
            init_session_state_var() 
            clean_files = preprocess_files(section, uploaded_files)
            st.session_state['clean_files'] = clean_files
            st.session_state['file_names'] = file_names
        
        
    if cntr.button("Clear", type="primary", disabled=\
    (not uploaded_files) and ('clean_files' not in st.session_state or not st.session_state['clean_files'])):
        st.session_state.uploader_key += 1
        if 'clean_files' in st.session_state:
            session_clear()
        st.rerun()
        
        
        
        
        
        
        
        
def set_model_visualisation(section):
    # TODO:
    # Call model for all images
    # Call visualiation for all images
    pass