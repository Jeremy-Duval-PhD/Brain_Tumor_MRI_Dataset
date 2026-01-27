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
- horizontal flip *(randomly apply)*
- slight rotation *(randomly apply on an interval)*
- slight (de)zoom *(randomly apply on an interval)*
- slight translation *(randomly apply on an interval)*
- slight contrast variation *(randomly apply on an interval)*

Additional data augmentation steps before fine-tuning:
- slight brightness variation *(randomly apply on an interval)*
- slight gaussian noise


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

1. The first test using MLflow involved the simple two-head model. It was effective at detecting tumours (with near 100% recall) and at identifying the tumour type (with near 50% accuracy), on both the training and validation data. 
	- However, it remained increasing oscillation on the validation data at the end. 
	- **End at epoch:** 13. 
	- **End with learning rate:** 0.0002500000118743628.

2. Changed EarlyStop min_delta to 0.0001. It was effective at detecting tumours (with near 100% recall) and at identifying the tumour type (with near 64% accuracy), on both the training and validation data. 
	- However, it remained increasing oscillation on the validation data at the end. 
	- The best epoch for tumor_presence was 4 but it seems to be 12 for tumor_type. 
	- **End at epoch:** 14. 
	- **End with learning rate:** 6.25000029685907e-05. 
	- **Run** : DenseNet121freeze=True_mask=True_20260126-1551.

3. Data augmentation added. It was effective at detecting tumours (with near 90% recall) and slightly at identifying the tumour type (with near 35% accuracy), on both the training and validation data. 
	- **No more oscillation on the validation data but performances are lowers than 2.**. 
	- The best epoch was 1. 
	- **End at epoch:** 11. 
	- **End with learning rate:** 0.0002500000118743628. 
	- **Run** : DenseNet121freeze=True_mask=True_20260127-1618.

3. Data augmentation: keep only RandomFlip. It was effective at detecting tumours (with near 96% recall) and slightly at identifying the tumour type (with near 50% accuracy), on both the training and validation data. 
	- **Reduced oscillation on the validation data and lower performances than 2.**. Globally, it was between 2. and 3.
	- The best epoch seems to be 8 for presence and 12 for type. 
	- **End at epoch:** 15. 
	- **End with learning rate:** 0.0002500000118743628. 
	- **Run** : DenseNet121freeze=True_mask=True_20260127-1644.

4. Data augmentation: keep only RandomRotation. It was effective at detecting tumours (with near 96% recall) and slightly at identifying the tumour type (with near 60% accuracy), on both the training and validation data. 
	- **Increased oscillation on the validation data, longer train and performances near but lower than 2.**. Seems to be ton effective.
		- Reducing rotation from 0.09 to 0.03 -> slightly better but same conclusion.
	- The best epoch seems to be 12 for presence and 19 for type. 
	- **End at epoch:** 21. 
	- **End with learning rate:** 0.0002500000118743628
	- **Run** : DenseNet121freeze=True_mask=True_20260127-1703.

5. Data augmentation: keep only RandomZoom. It was effective at detecting tumours (with near 96% recall) and slightly at identifying the tumour type (with near 63% accuracy), on both the training and validation data. 
	- **Quickly reduced oscillation on the validation data and nearly same performances than 2.**. Globally, it was between 2. and 3.
	- The best epoch seems to be 5 for presence and 12 for type. 
	- **End at epoch:** 14. 
	- **End with learning rate:** 0.0002500000118743628
	- **Run** : DenseNet121freeze=True_mask=True_20260127-1732

6. Data augmentation: keep only RandomTranslation. It was effective at detecting tumours (with near 97% recall) and slightly at identifying the tumour type (with near 60% accuracy), on both the training and validation data. 
	- **Nearly no more oscillation on the validation data and nearly same performances than 2.**. Globally, it was between 2. and 3.
	- The best epoch seems to be 11 for presence and 11 for type. 
	- **End at epoch:** 12. 
	- **End with learning rate:** 0.0002500000118743628
	- **Run** : DenseNet121freeze=True_mask=True_20260127-1745

7. Data augmentation: keep only RandomContrast. It was effective at detecting tumours (with near 0% recall) and slightly at identifying the tumour type (with near 42% accuracy), on both the training and validation data. 
	- **No more oscillation on the validation data and but performances are always 100% for train and 0% for validation in presence detection.**. Do not use !
	- The best epoch seems to be 11 for presence and 11 for type. 
	- **End at epoch:** 11. 
	- **End with learning rate:** 0.0002500000118743628
	- **Run** : DenseNet121freeze=True_mask=True_20260127-1758

8. Data augmentation: keep only RandomBrightness. It was effective at detecting tumours (with near 25% recall) and slightly at identifying the tumour type (with near 46% accuracy), on both the training and validation data. 
	- **No more oscillation on the validation data and but performances are always 100% for train and above 0% for validation in presence detection.**. Do not use !
	- The best epoch seems to be 6 for presence and 2 for type. 
	- **End at epoch:** 17. 
	- **End with learning rate:** 0.0002500000118743628
	- **Run** : DenseNet121freeze=True_mask=True_20260127-1812

9. Data augmentation: keep only GaussianNoise. It was effective at detecting tumours (with near 100% recall) and slightly at identifying the tumour type (with near 60% accuracy), on both the training and validation data. 
	- **Always oscillation on the validation data but slightly earlier than 2, and nearly same performances than 2 but lower.**. Globally, it was between 2. and 3.
	- The best epoch seems to be 2 for presence and 8 for type. 
	- **End at epoch:** 12. 
	- **End with learning rate:** 0.0002500000118743628
	- **Run** : DenseNet121freeze=True_mask=True_20260127-1820

10. Data augmentation: keep Flip, Zoom, Translation and Noise. It was effective at detecting tumours (with near 97% recall) and slightly at identifying the tumour type (with near 63% accuracy), on both the training and validation data. 
	- **Always oscillation on the validation data and nearly same performances than 2.**. Globally, it was between 2. and 3.
	- The best epoch seems to be 9 for presence and 15 for type. 
	- **End at epoch:** 9. 
	- **End with learning rate:** 0.0002500000118743628
	- **Run** : DenseNet121freeze=True_mask=True_20260127-1745

11. Data augmentation: keep Flip, Zoom and Translation. It was effective at detecting tumours (with near 99% recall) and slightly at identifying the tumour type (with near 60% accuracy), on both the training and validation data. 
	- **Always oscillation on the validation data, more large but progressively reducing and nearly same performances than 2.**. Globally, it was between 2.
	- The best epoch seems to be 4 for presence and 14 for type. 
	- **End at epoch:** 15. 
	- **End with learning rate:** 0.0002500000118743628
	- **Run** : DenseNet121freeze=True_mask=True_20260127-1850

**Data augmentation conclusion:** fine-tuning requires data augmentation. The best performance was achieved in step 10, even though the raw metrics were slightly worse than in step 2 (i.e. without data augmentation). I will keep their parameters.

12. 