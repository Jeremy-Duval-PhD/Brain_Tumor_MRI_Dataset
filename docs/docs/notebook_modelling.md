# Notebook modelling

The notebook is here as a space to make a predicitve model.

**Project goal :** using magnetic resonance imaging (MRI) data, detect the presence of tumors and classify them.

**Metrics :**
F1 score + recall.

I prefer to use their metrics compared to accuracy because the objective is early detection, which allows for monitoring and further examinations. It's more important to focus on avoiding false negatives, even if the accuracy isn't optimal.

## Modelling

**Goal :** Create a pipeline that performs the entire modelling step and saves the optimized model as artifact in the 'models' directory.

> [!CAUTION]
> I am using VirtualBox for this project, but due to an incompatibility with the TensorFlow library, I need to use Google Collab instead of the Jupyter Notebook for the preprocessing stage.

## Data

I will use the train and validation data in the 'processed' directory.

## Model

The model is a CNN. It uses a backbone for feature extraction. Then, one head predicts the presence or absence of a tumour. If the result is positive, a second head predicts the tumour type:
- glioma
- meningioma
- pituitary

## Backbone

The backbone use is **DenseNet121** from **RadImageNet**. It is a high-performance model for extracting MRI features, and it is adapted to my dataset's characteristics.

For more informations, follow this [link](https://github.com/BMEII-AI/RadImageNet).

## Preprocessing-like steps integrate to the model pipeline

Some pre-processing steps are best implemented in the modelling stage.

### Image converstion : gray to RGB

I do it at the modelling stage because it depends on the prerequisite of the choosing model.

### Data augmentation

Implemented in a second time, the data augmentation steps are as follows:
-
-
-

## MLflow

Modelling experiments are carried out using MLflow. It provides tools for monitoring metrics and parameters, saving models, and associating datasets. 
For more informations, please follow this [link](https://mlflow.org/docs/latest/ml/).

Due to my need to use Google Collab and VirtualBox, I have to use **ngrok** as an intermediary between MLflow and the notebook during experimentation. 