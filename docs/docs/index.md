# Brain_Tumor_MRI_Dataset documentation

## Description

The aim of this project is to develop a computer vision model. Its role is to diagnose the presence or absence of a brain tumor and its type, using MRI images.


This project is based on the following Kaggle dataset:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download

## Commands

The Makefile contains the central entry points for common tasks related to this project.

Files runs in the following order :
1. data/data_load.py *-> download files from Kaggle*
2. data/build_preprocessing_artifact.py.py *-> create a preprocessing pipeline artefact*
3. data/preprocessing_data.py *-> preprocess all raw data and save cleaned data in the processed directory*
4. models/make_model.py *-> create the model and save it. Need to launch MLflow and ngrok before see below)*
5. models/use_model.py *-> pipeline to test and explain the model*

To launch MLflow and ngrok:

1. Outside the Git repository, in a shell: mlflow ui --host 127.0.0.1 --port 5000 --allowed-hosts="\*" --cors-allowed-origins="\*"
2. In another shell:
	1. Configure ngrok: ngrok config add-authtoken <MY_AUTHTOKEN>
	2. Start ngrok: ngrok http 5000
3. Copy the “Forwarding” link into the *secrets.json* file.