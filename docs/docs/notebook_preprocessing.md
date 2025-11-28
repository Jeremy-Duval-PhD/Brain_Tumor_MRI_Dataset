# Notebook EDA

The notebook is here as a space to preprocessing data.

**Project goal :** using magnetic resonance imaging (MRI) data, detect the presence of tumors and classify them.

**Metrics :**
F1 score + recall.

I prefer to use their metrics compared to accuracy because the objective is early detection, which allows for monitoring and further examinations. It's more important to focus on avoiding false negatives, even if the accuracy isn't optimal.

## Preprocessing

**Goal :** Create a pipeline that performs the entire fixed preprocessing and saves the new data in the 'processed' directory.

> [!CAUTION]
> I am using VirtualBox for this project, but due to an incompatibility with the TensorFlow library, I need to use Google Collab instead of the Jupyter Notebook for the preprocessing stage.

## Dataset split

The data to be given has already been split into two directories: *Training* and *Testing*. 

Here, I created a stratified split of the training data to generate training and validation datasets. Currently, the ratios are respectively **80% and 20%**. The class distributions in both datasets are as follows:

- no tumor : 28%
- glioma : 23%
- meningioma : 23%
- pituitary : 26%

## Clipping bounds

Based on the train set, I need to recalculate the clipping bounds. Based on the 1st and 99th percentiles, I obtained the following values: -1.3045 and 3.0091.

## Visual comparisons

The visual comparison with the EDA indicates that the pre-processing pipeline is working correctly.

## Pipeline artefact

The preprocessing pipeline was exported as an artefact. You can use it as in the following code exemple:
```python3
loaded_layer = tf.saved_model.load(ARTEFACTS_DIR+"/preproc_pipeline")
img, label = loaded_layer("path/to/img.png", "glioma")
```