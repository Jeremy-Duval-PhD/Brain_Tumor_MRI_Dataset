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

In this notebook section, we can see that the images have different shapes **=> need to redimension**

#### Images orientation

The MRI can be taken from the front (coronal view), sides (sagittal view), or top (axial view) with slight rotations. The images can be complete or incomplete.

### Content

#### Target visualisation

The classes are quite balanced.

- no tumor: 1595
- glioma: 1321
- meningioma: 1339
- pituitary: 1457

#### Classes signification

- **no tumor:**
	Control class
- **[glioma](https://en.wikipedia.org/wiki/Glioma):**
	A benign or malignant tumor that originates from glial cells (the environment of neurons). They represent **30% of all brain tumors** and **80% of malignant tumors**. There are four subtypes.
- **[meningioma](https://en.wikipedia.org/wiki/Meningioma):**
	A slow-growing tumor from the meninges. 92% of cases are benign. The remaining 8% are either atypical or malignant.
- **[pituitary]():**
	A tumor that occurs in the pituitary gland. 65% of them are benign. It affects 1/6 people, but only 1/1,000 requires surgical intervention.

#### Images orientations

Most of the MRI pictures are in square format, but the "no tumor" class is balanced between portrait (355), landscape (735), and square (505), with the majority being landscape.

### Images view

The view is almost completely axial (from the top). There are a maximum of ten cases for sagittal and coronal views in each tumor class. We might need to exclude theirs ?

#### Images intensity / contrast

* The brightness intensity was evaluated using a z-score threshold of 2.576 (1% extremes). This suggests 158 cases out of 1595 (9.91%) in 'notumor', 22 out of 1339 (1.64%) in 'meningioma', one out of 1457 (0.07%) in 'pituitary', and none out of 1321 in 'glioma'.

* Based on the results of the pairwise z-test, it does not seem that the mean intensity is significantly different between the classes.

* A comparison of class variances using the Levene test suggests high statistical differences. The next step is to retry the test after normalising the images and cropping the borders.
	* **Cropping:** I tried three cropping methods on twenty images per class. The "black crop" method, based on the image background, produced the best results.
	* **Normalization:** Ok.
	* **Clipping:** : 
		* The pixel intensity distribution shows one peak for values < 10 and another for values > 210. These seem to represent background and saturation. However, there does not appear to be any noise in the dataset.
		* I applied a clipping function to the image, setting the lower bound to the first percentile (0) and the upper bound to the 99th percentile (209). This reduces variance differences.

* Based on ten random examples by class, it seems to improve MRI clarity, with tumours always being visible.

#### Intraclass variability

* On the boxplot: 
	- the median intensities are similar between classes.
	- there is overlap between classes.
	- there are many outliers, which **need to be investigated**.

* On the violin plot:
	- for 'notumor', the three oscillations seem to indicate the use of multiple machines.
	- for 'meningioma', the long tail can make learning more difficult.

* Outliers:
	- outliers seem to be primarily due to acquisition problems and are difficult to interpret.

### Conclusion


