# Notebook KBTD

The notebook is here as a space to explore data and protrotype the prediction model.

**Project goal :** using magnetic resonance imaging (MRI) data, detect the presence of tumors and classify them.

**Metrics :**
F1 score + recall.

I prefer to use their metrics compared to accuracy because the objective is early detection, which allows for monitoring and further examinations. It's more important to focus on avoiding false negatives, even if the accuracy isn't optimal.


## Exploratory Data Analysis

**Goal :** Understand data. EDA can be begin on the Kaggle website in the *Data* tab.

### EDA on shape

#### Target variable

The target variable is composed of four classes:

- no tumor
- glioma 
- meningioma
- pituitary

"no tumor" is the control class. The others are different types of brain tumors. 

#### Meta-data

It seems that I cannot access any metadata or connect data to patients.

#### Images shapes

The dataset consists of 5,712 MRIs. There are no dublicates.

In this notebook section, we can see that the images have different shapes.

#### Images orientation

The MRI can be taken from the front, sides, or bottom with slight rotations. The images can be complete or incomplete.

#### More ? 

### Content

#### Target visualisation

#### Classes signification

### Advenced

#### Hypothesis

### Conclusion


