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

### Shared part

- *GlobalAveragePooling1D()*: this layer is needed to flatten the backbone output. We prefer average pooling because it takes the feature position into account more evenly than *Flatten*, and is better for MRI than max pooling.
- *Dense(512, use_bias=False)*: this layer is needed to abstract and reduce the output of the backbone. The size of 512 is the half of output (1024). We don't use bias because of the subsequent normalisation.
- *BatchNormalization()*: normalise the weight before heads to stabilise it.
- *Activation('relu')*: classic and works weel.
- *Dropout(0.4)*: reduce the over-fitting risk.

### Heads part

Common for two heads :

- *Dense(128, use_bias=False)*: this layer is needed to learn. The size of 128 is a reduction compromize for efficient learning. We don't use bias because of the subsequent normalisation.
- *BatchNormalization()*: normalise the weight before heads to stabilise it.
- *Activation('relu')*: classic and works weel.
- *Dropout(0.2)*: reduce the over-fitting risk.

For *tumor presence* head:

- *Dense(1,activation='sigmoid')*: binary output.

For *tumor type* head:

- *Dense(4,activation='softmax')*: multiple class output.

### Model compile part

1. Tumour presence: in medical testing, we prefer the recall metric to accuracy. It is more important to detect false negatives. This is why we use *BinaryFocalCrossentropy*. The gamma is fixed to prioritise the detection of difficult cases, in line with the literature consensus. The alpha favours recall, but to a moderate extent, due to the imbalanced data proportions for this question (25% no tumour, 75% tumour).
2. Tumour type: we use a loss based on *SparseCategoricalCrossentropy*. The goal is to not take into account the prediction if the tumour presence prediction is "0" (no tumour).

## Backbone

The backbone use is **DenseNet121** from **RadImageNet**. It is a high-performance model for extracting MRI features, and it is adapted to my dataset's characteristics.

For more informations, follow this [link](https://github.com/BMEII-AI/RadImageNet).

## Preprocessing-like steps integrate to the model pipeline

Some pre-processing steps are best implemented in the modelling stage.

### Image convertion

Images are already in RGB and don't need convertion.

### Data augmentation

Implemented in a second time, the data augmentation steps are as follows:
-
-
-

## Fitting

1. Configure a *ReduceLROnPlateau* with a small, progressive reduction.
2. Configure an *EarlyStopping*.
3. In the fit configuration, the training and validation steps are the same size as the file numbers because the batch size is 1.

## MLflow

Modelling experiments are carried out using MLflow. It provides tools for monitoring metrics and parameters, saving models, and associating datasets. 
For more informations, please follow this [link](https://mlflow.org/docs/latest/ml/).

I need to use a cloud notebook due to an incompatibility between TensorFlow and VirtualBox. I started out on Google Collab, but due to poor performance, I need to transfer my notebook to Kaggle. Then, I have to use **ngrok** as an intermediary between MLflow and the notebook during experimentation. 

For each run, I use the *autolog* function and log model and hyperparameters. It helps with reproducibility.

## Modelling Tests:

0. The first test was out from MLflow and with the simple two-head model. It was effective at detecting tumours and slightly better than random at identifying the tumour type, on training data. High overfitting and bad performances on validation data.

1. The first test using MLflow involved the simple two-head model. It was effective at detecting tumours (with near 100% recall) and at identifying the tumour type (with near 50% accuracy), on both the training and validation data. However, it remained increasing oscillation on the validation data at the end. **End at epoch:** 13. **End with learning rate:** 0.0002500000118743628.

2. Changed EarlyStop min_delta to 0.0001. It was effective at detecting tumours (with near 100% recall) and at identifying the tumour type (with near 50% accuracy), on both the training and validation data. However, it remained increasing oscillation on the validation data at the end. The best epoch for tumor_presence was 4 but it seems to be 12 for tumor_type. **End at epoch:** 14. **End with learning rate:** 6.25000029685907e-05.