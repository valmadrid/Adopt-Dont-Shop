Status: WIP

<img src="https://c.wallhere.com/photos/e7/0c/1920x1200_px_animals_cats_dog-729051.jpg!d" />
<img src="https://www.petfinder.my/images/logo-575x100.png" />

# Helping Rehome Our Pets
*Petfinder.my is a Malaysian website that hosts pet listings for adoption and for sale.  It also educates the public on how to manage their pets responsibly.  It is also used as a venue for animal welfare advocates.*

### Summary
This project aims to help shelters, rescuers and owners rehome their pets faster.  A classifier model was developed to predict pet adoption rate. Extreme Gradient Boosting (XGBoost) gave the highest accuracy and quadratic Cohen's kappa<sup>1</sup>  scores (42% and 36%, respectively) among all the classifiers and ensemble methods that were used.  Enhancements were recommended for Petfinder.my to help boost the adoptability of the pets.  

As a supplement, a content-based recommendation system was also developed based on the pet images.  ResNet50 was employed to extract features from the images while cosine similarity was used to measure similarities between images.

### Dataset
<a href="https://www.kaggle.com/c/petfinder-adoption-prediction">Dataset</a> consists of 14k pet listings across Malaysia:

<FIFI PHOTO HERE>

<a href="https://cloud.google.com/natural-language/">Google Cloud Natural Language API</a> was used to extract sentiment analysis from the *description*.  Each description was given score and magnitude.

> - score is the overall emotional leaning of the text; it ranges between -1.0 (negative sentiment) and 1.0 (positive sentiment)
> - magnitude denotes the overall strength of emotion (or the score); longer text has higher magnitude

Pet images uploaded by the rescuers are also included.  These were used to create the recommender system.

Additional information regarding the location was added to the dataset: state population, density, area and GDP per capita.

### Process
Classifer Model:

<img src="https://github.com/valmadrid/Petfinder-Malaysia-Helping-Rehome-Our-Pets-/blob/master/workflow%20A.png"/>

Recommender System:



### Results and Recommendations

### Libraries and Modules used
- Scikit-learn
- XGBoost
- Lightgbm
- Pandas
- PandasSQL
- Matplolib
- Seaborn
- Pickle
- Shap
- OS
- Shutil
- Python Imaging Library
- OpenCV
- Regular expression operations
- Wordcloud

### Files

### Contributor
<a href="https://www.linkedin.com/in/valmadrid/">Grace Valmadrid</a>

### Credits
- <a href="https://www.kaggle.com/c/petfinder-adoption-prediction">Kaggle</a> for the dataset

- <a href="https://www.kaggle.com/chocozzz/petfinder-external-data">chocozzz</a> (Kaggle user) for the external data used in this project

- <a href="https://www.petfinder.my">Petfinder.my</a> and <a href="https://c.wallhere.com">Wallhere</a> for the images used here

Footnotes:

1 https://en.wikipedia.org/wiki/Cohen%27s_kappa
