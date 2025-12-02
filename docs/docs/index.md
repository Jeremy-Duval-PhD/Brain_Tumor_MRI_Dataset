# Brain_Tumor_MRI_Dataset documentation

## Description

The aim of this project is to develop a computer vision model. Its role is to diagnose the presence or absence of a brain tumor and its type, using MRI images.


This project is based on the following Kaggle dataset:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download

## Commands

The Makefile contains the central entry points for common tasks related to this project.

Files runs in the following order :
1. data/data_load.py *-> download files from Kaggle*
2. data/preprocessing.py *-> create a preprocessing pipeline artefact and preprocess all raw data and save cleaned data in the processed directory*
