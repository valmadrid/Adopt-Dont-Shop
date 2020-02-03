Status: WIP

<img src="https://c.wallhere.com/photos/e7/0c/1920x1200_px_animals_cats_dog-729051.jpg!d" />
<img src="https://www.petfinder.my/images/logo-575x100.png" />

# Helping Rehome Our Pets
*Petfinder.my is a Malaysian website that hosts pet listings for adoption and for sale.  It also educates the public on how to manage their pets responsibly.  It also provides venue for animal welfare advocates.*

### Summary
This project aims to help shelters, rescuers and owners rehome their pets faster.  A classifier model was developed to predict pet adoption rate. Extreme Gradient Boosting (XGBoost) gave the highest accuracy and quadratic Cohen's kappa<sup>1</sup>  scores (42% and 36%, respectively) among all the classifiers and ensemble methods that were used.  Enhancements were recommended for Petfinder.my to help boost the adoptability of the pets.  

As a supplement, a content-based recommendation system was also developed based on the pet images.  ResNet-50<sup>2</sup> was employed to extract features from the images while cosine similarity was used to measure similarities between images.

### Dataset
<a href="https://www.kaggle.com/c/petfinder-adoption-prediction">Dataset</a> consists of 14k pet listings across Malaysia.  Each listing has the following features:

<img src="https://github.com/valmadrid/Petfinder-Malaysia-Helping-Rehome-Our-Pets-/blob/master/images/fifi.png" />

<a href="https://cloud.google.com/natural-language/">Google Cloud Natural Language API</a> was used to extract sentiment from the *description*.  Each description has score and magnitude values.

> - score is the overall emotional leaning of the text; it ranges between -1.0 (negative sentiment) and 1.0 (positive sentiment)
> - magnitude denotes the overall strength of emotion (or the score); longer text has higher magnitude

Additional information regarding the location was added to the dataset: state population, density, area and GDP per capita.

### Process
Classifer Model:

<img src="https://github.com/valmadrid/Petfinder-Malaysia-Helping-Rehome-Our-Pets-/blob/master/images/workflow%20A.png"/>

Recommender System:

<img src="https://github.com/valmadrid/Petfinder-Malaysia-Helping-Rehome-Our-Pets-/blob/master/images/workflow%20B.png"/>

### Results and Recommendations

### Important Libraries and Modules used
- Scikit-learn
- XGBoost
- Lightgbm
- Shap
- Keras

### Files

Main notebook: <a href="https://github.com/valmadrid/Petfinder-Malaysia-Helping-Rehome-Our-Pets-/blob/master/petfinder.ipynb">petfinder.ipynb</a>

Data cleaning and pre-processing: <a href="https://github.com/valmadrid/Petfinder-Malaysia-Helping-Rehome-Our-Pets-/blob/master/preprocess.py">preprocess.py</a>

Model evaluation: <a href="https://github.com/valmadrid/Petfinder-Malaysia-Helping-Rehome-Our-Pets-/blob/master/functions.py">functions.py</a>

Recommender system: <a href="https://github.com/valmadrid/Petfinder-Malaysia-Helping-Rehome-Our-Pets-/blob/master/get_similar_pets.ipynb">get_similar_pets.ipynb</a> (interactive), <a href="https://github.com/valmadrid/Petfinder-Malaysia-Helping-Rehome-Our-Pets-/blob/master/recommendation.py">recommendation.py</a>

Presentation slides: <a href="https://github.com/valmadrid/Petfinder-Malaysia-Helping-Rehome-Our-Pets-/blob/master/Petfinder.pdf">petfinder.pdf</a>

### Contributor
<a href="https://www.linkedin.com/in/valmadrid/">Grace Valmadrid</a>

### Credits
- <a href="https://www.kaggle.com/c/petfinder-adoption-prediction">Kaggle</a> for the dataset

- <a href="https://www.kaggle.com/chocozzz/petfinder-external-data">chocozzz</a> (Kaggle user) for the external data used in this project

- <a href="https://www.petfinder.my">Petfinder.my</a> and <a href="https://c.wallhere.com">Wallhere</a> for the images used here

Footnotes:

1 https://en.wikipedia.org/wiki/Cohen%27s_kappa

2 https://keras.io/applications/#resnet
