
<img src="https://c.wallhere.com/photos/e7/0c/1920x1200_px_animals_cats_dog-729051.jpg!d" />
<img src="https://www.petfinder.my/images/logo-575x100.png" />

# Helping Rehome Our Pets
*Petfinder.my is a Malaysian website that hosts pet listings for adoption and for sale.  It also educates the public on how to manage their pets responsibly.  It is also used as a venue for animal welfare advocates.*

### Summary
This project aims to help shelters, rescuers and owners rehome their pets faster.  A classifier model was developed to predict the adoption rate of the pets. Extreme Gradient Boosting (XGBoost) gave the highest accuracy and quadratic Cohen's kappa<sup>1</sup>  scores among all the classifiers and ensemble methods that were used.  Enhancements were recommended Petfinder.my to help boost the adoptability of the pets.  As a supplement, a content-based recommendation system was also developed based on the pet images.

### Dataset
<a href="https://www.kaggle.com/c/petfinder-adoption-prediction">Dataset</a> consists of 14k pet listings across Malaysia:

* Type - Type of animal (1 = Dog, 2 = Cat)
* Name - Name of pet (Empty if not named)
* Age - Age of pet when listed, in months
* Breed1 - Primary breed of pet
* Breed2 - Secondary breed of pet, if pet is of mixed breed
* Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
* Color1 - Color 1 of pet
* Color2 - Color 2 of pet
* Color3 - Color 3 of pet
* MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
* FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
* Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
* Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
* Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
* Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
* Quantity - Number of pets represented in profile
* Fee - Adoption fee (0 = Free)
* State - State location in Malaysia
* RescuerID - Unique hash ID of rescuer
* VideoAmt - Total uploaded videos for this pet
* Description - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.
* PetID - Unique hash ID of pet profile
* PhotoAmt - Total uploaded photos for this pet
* AdoptionSpeed - Categorical speed of adoption:
    - 0 - Pet was adopted on the same day as it was listed.
    - 1 - Pet was adopted between 1 and 7 days (1st week) after being listed.
    - 2 - Pet was adopted between 8 and 30 days (1st month) after being listed.
    - 3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.
    - 4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).

<a href="https://cloud.google.com/natural-language/">Google Cloud Natural Language API</a> was used to extract sentiment analysis from the *description*.  Each description was given score and magnitude.

> - score is the overall emotional leaning of the text; it ranges between -1.0 (negative sentiment) and 1.0 (positive sentiment)
> - magnitude denotes the overall strength of emotion (or the score); longer text has higher magnitude

Pet images uploaded by the rescuers are also included.  These were used to create the recommender system.

Additional information regarding the location was added to the dataset: state population, density, area and GDP per capita.

### Process

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

### Results and Recommendations

### Files

### Contributor
<a href="https://www.linkedin.com/in/valmadrid/">Grace Valmadrid</a>

### Credits
<a href="https://www.kaggle.com/c/petfinder-adoption-prediction">Kaggle</a> for the dataset

<a href="https://www.kaggle.com/chocozzz/petfinder-external-data">chocozzz</a> (Kaggle user) for the external data used in this project

<a href="https://www.petfinder.my">Petfinder.my</a> and <a href="https://c.wallhere.com">Wallhere</a> for the images used here

Footnotes:

1 https://en.wikipedia.org/wiki/Cohen%27s_kappa
