import streamlit as st
import numpy as np
from pathlib import Path
import yaml
import json
import gc
import io
import zipfile
import tensorflow as tf
from PIL import Image
import tempfile
import pandas as pd
import warnings
from src.models.model_archi import get_model_built
from src.models.commons import run_medical_XAI_one_image, get_presence_explainer, \
                               load_tfrecord_dataset, split_labels, StdSHAPWarning

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing configuration file: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
    

@st.cache_resource(show_spinner=False)
def get_preproc_model():
    init_session_state_var()
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
    
    if 'config_path' not in st.session_state:
        st.session_state['config_path'] = st.session_state['project_root'] / "config.yaml"
    
    if 'config' not in st.session_state:
        st.session_state['config'] = load_config(st.session_state['config_path'])
        
    if 'artefact_dir' not in st.session_state:
        st.session_state['artefact_dir'] = \
        st.session_state['project_root'] / st.session_state['config']["path"]["models_dir"] / "preproc_pipeline"
        
    if 'temp_dir_raw' not in st.session_state:
        temp_dir = Path(tempfile.mkdtemp())
        st.session_state['temp_dir_raw'] = temp_dir
        
    if 'temp_dir_preproc' not in st.session_state:
        temp_dir = Path(tempfile.mkdtemp())
        st.session_state['temp_dir_preproc'] = temp_dir
        
    if 'temp_dir_output' not in st.session_state:
        temp_dir = Path(tempfile.mkdtemp())
        st.session_state['temp_dir_output'] = temp_dir
    
    
def load_image(uploaded_file):
    uploaded_file.seek(0)
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    return np.array(img)


def build_df_from_uploaded(paths, labels=None):
    """
    paths : list[str] → peut contenir des dossiers OU fichiers
    """

    filepaths = []

    for p in paths:
        p = Path(p)

        if p.is_dir():
            # browse folder
            for img_file in p.iterdir():
                if img_file.is_file():
                    filepaths.append(str(img_file.resolve()))
        elif p.is_file():
            filepaths.append(str(p.resolve()))
            
    # remove duplicates
    filepaths = list(set(filepaths))
    st.write(st.session_state.file_names)
    st.write(filepaths)
    if labels is None:
        # default value -> use only to compare prediction and reality during model creation
        labels = ["notumor"] * len(filepaths)

    return pd.DataFrame({
        "filepath": filepaths,
        "label": labels
    })
    
    
def get_tf_dataset(preproc_model, nb_files):
    df = build_df_from_uploaded([st.session_state['temp_dir_raw']])
    paths = df["filepath"].astype(str).values
    labels = df["label"].astype(str).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
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


def save_images(section, images, nb_files):
    count = 0

    progress_text = "Upload images. Please wait."
    progress_bar = st.session_state.progress_bar
    progress_bar = progress_bar.progress(0.0, text=progress_text)

    for idx, (img, file) in enumerate(images):
        # security for type
        if hasattr(img, "numpy"):
            img = img.numpy()
    
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        # PIL image
        pil_img = Image.fromarray(img)

        # saving
        #img_path = st.session_state['temp_dir_raw'] / f"{filename_prefix}_{idx:04d}.jpg"
        img_path = st.session_state['temp_dir_raw'] / f"{file.split['.'][0]}.jpg"
        pil_img.save(img_path, format="PNG")

        count += 1

        # update progress
        progress = count / nb_files
        progress_bar.progress(progress, text=f"{progress_text} ({count}/{nb_files})")

    progress_bar.progress(1.0, text="Upload images. Done ✅")


def save_tf_records(section, ds, nb_files, batch_size, filename_prefix="data"):
    writer = None
    count = 0
    file_idx = 0
    
    progress_text = "Saving images. Please wait."
    progress_bar = st.session_state.progress_bar
    progress_bar = progress_bar.progress(0, text=progress_text)

    for img, lbl in ds:
        if count % batch_size == 0:
            if writer:
                writer.close()
            record_path = st.session_state['temp_dir_preproc'] / f"{filename_prefix}_{file_idx:03d}.tfrecord"
            writer = tf.io.TFRecordWriter(str(record_path))
            file_idx += 1
        writer.write(serialize_example(img, lbl))
        count += 1
        
        # update progress bar
        progress = count / nb_files
        progress_bar.progress(progress, text=f"{progress_text} ({count}/{nb_files})")

    progress_bar.progress(1.0, text="Saving images. Done ✅")

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
        
    images = [(load_image(f), f.name) for f in new_files]
    nb_files = len(images)
    save_images(section, images, nb_files)
    model = get_preproc_model()
    ds = get_tf_dataset(model, nb_files)
    
    with st.spinner("Preprosessing images in progress...", show_time=True):
        preproc_img = get_clean_tfrecords(ds)
    
    batch_size = st.session_state['config']['model']['batch_size']
    save_tf_records(section, ds, nb_files, batch_size)
    
    return preproc_img
    

def reactivate_form(rerun=True):
    st.session_state.submitted = False
    st.session_state['is_processing'] = False

    if 'clean_files' in st.session_state:
        session_clear()

    if rerun:
        st.rerun()
    
        
@st.fragment
def settings_shap():
    popover = st.popover("SHAP settings", disabled=st.session_state.is_processing)
    help_msg = """Activate if you are on cloud. It will fix SHAP samples to 5."""
    low_memory = popover.toggle("Low memory", True, help=help_msg, \
                                    key="low_memory",\
                                    disabled=st.session_state.is_processing)
    help_msg = """Use it for local applications. A high number of samples is more effective for explaining the model with SHAP, but it can use a lot of memory."""
    popover.slider("Number of semples for SHAP", 1, 200, 100, 1, help=help_msg, \
                                  key="nsamples",\
                                  disabled=(st.session_state.is_processing or low_memory))


def set_uploaded_data(section):
    # Init session state
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if "submitted" not in st.session_state:
        st.session_state.submitted = False
        
    if 'is_processing' not in st.session_state:
       st.session_state['is_processing'] = False

    with section.container(border=True):

        help_msg = """
        You can upload MRI files at jpg or png format. You can download the testing files for this app [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download).
        """
        
        # uploader form
        with st.form(key="upload_form"):
    
            uploaded_files = st.file_uploader(
                "Upload your MRI",
                type=['png', 'jpg'],
                help=help_msg,
                accept_multiple_files=True,
                key=f"uploader_{st.session_state.uploader_key}",
                disabled=st.session_state.is_processing,
            )
            
            col1, col2 = st.columns([1, 2])
    
            clear = col1.form_submit_button(
                "Clear",
                #type="primary",
                disabled=(
                    st.session_state.is_processing
                ),
                use_container_width=True
            )
    
            submit = col2.form_submit_button(
                "Submit",
                type="primary",
                disabled=st.session_state.is_processing,
                use_container_width=True
            )
       
        settings_shap()

    # Process files after submit
    if (submit or st.session_state.submitted) and uploaded_files:
        if not st.session_state.is_processing:
            st.session_state.submitted = True
            st.session_state.is_processing = True
            st.rerun()
            
        file_names = get_files_names(uploaded_files)

        if 'clean_files' not in st.session_state \
        or file_names != st.session_state.get('file_names', []):

            init_session_state_var()

            clean_files = preprocess_files(section, uploaded_files)

            st.session_state['clean_files'] = clean_files
            st.session_state['file_names'] = file_names

    # Clear form
    if clear:
        st.session_state.uploader_key += 1
        reactivate_form(rerun=True)
        
        




@st.cache_resource(show_spinner=False)
def rebuild_model(config_str):
    config = json.loads(config_str)
    
    tf.keras.backend.clear_session()
    
    img_size = config['data_preprocessing']['img_size']
    model_dir = config['path']['models_dir']
    seed = config['general']['seed']
    
    model = get_model_built(
        img_size,
        model_dir,
        freeze_backbone=False,
        seed=seed
    )
    
    model.load_weights(model_dir+"brain_tumor_heads.weights.h5")
    
    return model


@st.cache_resource(show_spinner=False)
def load_background():
    path = Path(st.session_state['config']['path']['models_dir'])
    return np.load(path / "background.npy")


def get_type_explainer_cache():
    if 'type_explainers_cache' not in st.session_state:
        st.session_state['type_explainers_cache'] = {}
    return st.session_state['type_explainers_cache']


def create_zip_from_images(image_paths):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in image_paths:
            zf.write(img_path, arcname=img_path.name)
    zip_buffer.seek(0)
    return zip_buffer


def download_images_zip(section, image_paths, session_key="zip_output_images", zip_name="output_images.zip"):
    if session_key not in st.session_state:
        st.session_state[session_key] = create_zip_from_images(image_paths)

    col1, col2, col3 = st.columns([1, 1, 1])

    reset_btn = col1.button("Reset", 
                            help="Clear images and interface",
                            on_click=reactivate_form,
                            kwargs={"rerun": False}
                            )

    col3.download_button(
        label="Download images",
        data=st.session_state[session_key],
        file_name=zip_name,
        help= f'The images will be downloaded as {zip_name}',
        mime="application/zip",
        icon=":material/download:",
        on_click=reactivate_form,
        kwargs={"rerun": False}
    )
    
    
def get_img_paths(section, valid_ext = [".jpg", ".jpeg", ".png"]):
    if "temp_dir_output" not in st.session_state:
        section.warning("No output directory found in session.")
        return
    
    img_dir = Path(st.session_state["temp_dir_output"])

    if not img_dir.exists():
        section.error(f"Directory does not exist: {img_dir}")
        return
    
    image_paths = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in valid_ext
    ])
    
    if not image_paths:
        section.info("No images found in output directory.")
        return
    
    return image_paths
    

def is_in_file_names(file):
    for name in st.session_state.file_names:
        if file in name.split('.')[0] :
            return True
    return False


def is_presence_head(file):
    if "presence" in file:
        return True
    else:
        return False
    
    
def get_pred_confidence(file, is_pres):
    nb = int(float(file.split('(')[-1].split(')')[0])*100)
    if is_pres: # if pred <= 0.5 = no tumor and > 0.5 = tumor 
        if nb > 50:
            return nb
        else:
            nb = 100 - nb
        return nb
    else:
        return nb


def get_pred_label(file):
    return file.split('__')[-1].split('_')[0]


def get_file_root(file):
    splited = file.split('_')
    return splited[0] + '_' + splited[1]


def get_file_name(path):
    return path.name.split("/")[-1]


def display_output_images(section):
    image_paths = get_img_paths(section)
    if not image_paths:
        return
    
    tumor_detected = []
    with section.container():
        for img_path in image_paths:
            file_name = get_file_name(img_path)
            root = get_file_root(file_name)
            in_file_n = is_in_file_names(root)
            
            if in_file_n:
                is_pres = is_presence_head(file_name)
                pred_conf = get_pred_confidence(file_name, is_pres)
                pred = get_pred_label(file_name)
                
                if is_pres or root in tumor_detected:
                    try:
                        img = Image.open(img_path)
                        section.image(
                            img,
                            caption=img_path.name,
                            use_column_width=True
                        )
                    except Exception as e:
                        section.warning(f"Error loading {img_path.name}: {e}")
                    
                if is_pres:
                    if pred == "tumor":
                        tumor_detected.append(root)
                    else:
                        msg = f"{root}: no tumor detected, with {pred_conf}% probability."
                        section.badge(msg, icon=":material/check:", color="green")
                else:
                    if root in tumor_detected:
                        msg= f"{root}: {pred} detected, with {pred_conf}% probability."
                        section.badge(msg, icon="🚨", color="red")    

            
            
def get_datasets_preprocs():
    path = st.session_state['temp_dir_preproc']
    
    ds = load_tfrecord_dataset(
        path,
        shuffle=True,
        batch_size=st.session_state['config']['model']['batch_size'],
        repeat=False
    ).prefetch(tf.data.AUTOTUNE)
    
    ds = ds.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds

        
def set_model_visualisation(section):
    init_session_state_var()
    
    if 'file_names' in st.session_state and st.session_state['file_names'] and st.session_state.submitted:
        model = rebuild_model(json.dumps(st.session_state['config'], sort_keys=True))
        background_images = load_background()
        
        classes = st.session_state['config']['general']['classes']
        presence_cat = st.session_state['config']['general']['presence_cat']
        type_explainers_cache = get_type_explainer_cache()
        dataset = get_datasets_preprocs()
        
        progress_text = "MRI processing in progress. Please wait."
        progress_bar = st.session_state.progress_bar
        progress_bar = progress_bar.progress(0.0, text=progress_text)
        count = 0
        nb_files = len(st.session_state['file_names'])
        for x_batch, _ in dataset:
            for img in x_batch:
                explainer_presence = get_presence_explainer(model, background_images.copy()) # in loop to avoid bug and help shap
                img_id=st.session_state['file_names'][count].split('.')[0]
                
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    #st.write(st.session_state.low_memory)
                    #st.write(st.session_state.nsamples)
                    st.session_state['type_explainers_cache'] = run_medical_XAI_one_image(\
                                              img.numpy(), \
                                              st.session_state['config']['data_preprocessing']['img_size'], \
                                              model, \
                                              background_images, \
                                              explainer_presence, \
                                              st.session_state['temp_dir_output'], \
                                              classes, \
                                              presence_cat=presence_cat,\
                                              type_explainers_cache=type_explainers_cache,
                                              low_memory=st.session_state.low_memory,
                                              nsamples=st.session_state.nsamples,
                                              img_id=img_id)
                    
                    for warning in w:
                        if issubclass(warning.category, StdSHAPWarning):
                            msg = f"SHAP failed for {img_id}. {warning.message}. This can occur if 'low_memory' is selected or if 'nsamples' is low."
                            section.warning(msg, icon="🚨")
                
                # update progress
                count += 1
                progress = count / nb_files
                progress_bar.progress(progress, text=f"{progress_text} ({count}/{nb_files})")
                
                gc.collect()
    
        progress_bar.progress(1.0, text="MRI processing done ✅")
        
        image_paths = get_img_paths(section) 
        with section.container(border=True):
            download_images_zip(section, image_paths)
            
        section.divider()
        
        img_ctnr = section.container()
        display_output_images(img_ctnr)
        






