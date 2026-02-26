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

## Epoch filtering

In the modelling tests section, starting from step 15, the epoch filtering for a modelling test was selected as following:
- *presence* recall ≥ 0.94
- *type* accuracy ≥ 0.55

A personalised score is also used. It is defined as the sum of the loss and the principal head metric, weighted accordingly.

$ S = w1 \times Recall_presence_norm + w2 \times Accuracy_type_norm + w3 \times Loss_presence_norm + w4 \times Loss_type_norm $

The idea was to prioritise specific metrics for each head while also taking into account the respective loss. Also, presence has been prioritised over type. The chosen weights are:
- w1 : 0.40
- w2 : 0.35
- w3 : 0.15
- w4 : 0.10


## Modelling Tests:

### Heads optimisation

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

12. Head tumor_type: 128 to 256 neurones in the hidden dense layer. Detecting tumours with near 98% recall. Identifying the tumour type with near 64% accuracy. Performances both the training and validation data. 
	- **High oscillation on the validation data, but type accuracy don't fall at end ; nearly same performances than 11.**
	- The best epoch seems to be 7 for presence and 14 for type. 
	- **End at epoch:** 18. 
	- **End with learning rate:** 0.0001250000059371814
	- **Run** : DenseNet121freeze=True_mask=True_20260128-1442

13. Head tumor_type: back to 128 neurones + Adding loss weights to improve tumor type detection. Detecting tumours with near 97% recall. Identifying the tumour type with near 60% accuracy. Performances both the training and validation data. 
	- **High oscillation on the validation data with a stabilization on step 14 for presence and nearly same performances than 11.**
	- The best epoch seems to be 9 for presence and 15 for type. 
	- **End at epoch:** 19. 
	- **End with learning rate:** 0.0002500000118743628
	- **Run** : DenseNet121freeze=True_mask=True_20260128-1509

13. Loss weights: 1.2 to 1.3. Detecting tumours with near 97% recall. Identifying the tumour type with near 62% accuracy. Performances both the training and validation data. 
	- **Oscillation on the validation data are similar for presence with a few more time to stabilize and clearly improved oscillation for type ; nearly same performances than 11.**
	- The best epoch seems to be 7 for presence and 13 for type. 
	- **End at epoch:** 17. 
	- **End with learning rate:** 0.0002500000118743628
	- **Run** : DenseNet121freeze=True_mask=True_20260128-1532

**Keeping loss weights at 1.0 for presence and 1.3 for type.**

14. EarlyStopping and ReduceLROnPlateau : monitor on val_tumor_type_loss in place of val_tumor_presence_recall. Detecting tumours with near 97% recall. Identifying the tumour type with near 60% accuracy. Performances both the training and validation data. 
	- **Oscillation on the validation data are similar for presence with a few more time to stabilize and worst oscillation for type ; type accuracy stay longer at high performances than 13.**
	- The best epoch seems to be 13 for presence and 13 for type. 
	- **End at epoch:** 18. 
	- **End with learning rate:** 0.0002500000118743628
	- **Run** : DenseNet121freeze=True_mask=True_20260128-1606

15. Detecting tumours with near 97% recall. Identifying the tumour type with near 60% accuracy. Performances both the training and validation data. 
	- **There is some oscillation in the validation data, with a decline above epoch 10 and a gradual stabilisation over time. For loss, epoch 21 was very stable.**
	- **Personalized score best epoch:** 21. 
	- **End at epoch:** 22. 
	- **End with learning rate:** 0.0001250000059371814
	- **Run** : DenseNet121freeze=True_mask=True_20260203-1523

16. Adding checkpoint to reload best epoch's weights + monitoring correction ("max' to "min" for loss observation) + adding callback to stop on NaN value. Detecting tumours with near 98.5% recall. Identifying the tumour type with near 66% accuracy. Performances both the training and validation data. 
	- **There is some oscillation in the first half experimention. After 10 (presence) or 25 (type) epochs, recall, accuracy and all loss are very stable.**
	- **Personalized score best epoch:** 43. 
	- **End at epoch:** 52. 
	- **End with learning rate:** 7.812500371073838e-06
	- **Run** : DenseNet121freeze=True_mask=True_20260204-1641

17. New train after Sequential remove in architecture (to be grad-cam safe). Detecting tumours with near 99% recall. Identifying the tumour type with near 65% accuracy. Performances both the training and validation data. 
	- **Similar to 16th but a little bit less stable.**
	- **Personalized score best epoch:** 34. 
	- **End at epoch:** 50. 
	- **End with learning rate:** 3.125000148429535e-05
	- **Run** : DenseNet121freeze=True_mask=True_20260213-1004

18. New train after adding a proxy layer to approximate backbone output (for grad-CAM) + EarlyStop : min_delta 1e-4 1 e-5. Detecting tumours with near 97.7% recall. Identifying the tumour type with near 63.1% accuracy. Performances both the training and validation data. 
	- **Similar to 17th but a little bit more stable.**
	- **Personalized score best epoch:** 27. 
	- **End at epoch:** 53. 
	- **End with learning rate:** 3.1250e-05
	- **Run** : DenseNet121freeze=True_mask=True_20260217-1117

18. Detecting tumours with near 98.3% recall. Identifying the tumour type with near 66.4% accuracy. Performances both the training and validation data. 
	- **Similar to 18th but a little bit more stable.**
	- **Personalized score best epoch:** 40. 
	- **End at epoch:** 51. 
	- **End with learning rate:** 6.2500e-05
	- **Run** : DenseNet121freeze=True_mask=True_20260217-1156

**Base on the S score, the best epoch was 40 from 18th run. We will keep this as the best heads model optimisation.**

**NB: because the 'tumor_type' is masked when the 'tumor_presence' head does not detect a tumor, the 'no_tumor' class is never trained. Consequently, the accuracy value was incorrect. The real 'tumor_type' accuracy is 84%.** 

#### Confusion matrix for tumor presence head, before fine-tuning
![Confusion matrix for tumor presence head, before fine-tuning](/figures/Confusion_mtrx_presence_bf_fine_tuning.png)

Accuracy by class:
- no_tumor: 98.4%
- tumor: 97.9%

#### Confusion matrix for type tumor head, before fine-tuning
![Confusion matrix for type tumor head, before fine-tuning](/figures/Confusion_mtrx_type_bf_fine_tuning.png)

"no tumor" bad score is explained by the loss mask when the presence head predicts that there is no tumour.

"meningioma" score is explained by the fact that, in some cases, meningiomas can resemble gliomas, and the classification of this tumour type depends more on the MRI scan.

Accuracy by class:
- glioma: 97.7%
- meningioma: 54.5%
- pituitary: 98.6%

#### Grad-CAM
Due to Keras limitations, I was unable to recreate and connect the model, nor expose the internal layers of the model (it is integrated as a single layer). I therefore added a fixed “Conv2D” layer in order to best approximate the backbone output.

The Grad-Cam shows the focus of the AI model using the classic JET colour scale (from blue to red).

### Fine-tuning

Backbone levels:
- denseblock1–2 → low-level textures
- denseblock3 → intermediate patterns
- denseblock4 (conv5_block) → fine semantics

Strategy:
1. The lowest score is for a meningioma. Taking previous information into account, it appears that there is a lack of fine detection by the backbone. 
2. We must unfreeze the fourth level of backbone layers.
3. We need to monitor the meningioma recall to include it in the final S-score.