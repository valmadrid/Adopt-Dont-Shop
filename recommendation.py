import pandas as pd
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

import preprocess as pre



def load_image(df, index, images_folder_path):
    """
    Loads an image from the filename specified in df.filename[index]
    """
    
    return cv2.imread(images_folder_path+df.filename[index])


def get_features(model, image_filename, images_folder_path):
    """
    Takes in model, image filename (string) and image folder path (string) then reshapes and converts the images into an array using the model
    """

    img = image.load_img(images_folder_path + image_filename,
                         target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).reshape(-1)


def get_similar_pet_listings(i, top=2):
    """
    Takes in an index (integer) of the pet in pets_reco table then returns top n similar pets
    """
    df = pickle.load(open("pets_reco.pkl", "rb"))
    type_1 = pickle.load(open("df_dogs.pkl", "rb"))
    type_2 = pickle.load(open("df_cats.pkl", "rb"))
    if df.type.loc[i] == 1:
        count = 0
        for j in list(type_1.index):
            if i == j:
                new_index = count
            else:
                count +=1
        dogs_reco(i, new_index, type_1, top)
    elif df.type.loc[i] == 2:
        count = 0
        for j in list(type_2.index):
            if i == j:
                new_index = count
            else:
                count +=1
        cats_reco(i, new_index, type_2, top)
    else:
        print("Index out of range.  Please select from 0 to", len(df))

def dogs_reco(pet_index, new_index, df, top):
    """
    Used by get_similar_pet_listins to get top n similar dogs
    
    """
    
    df_cosine_dogs = pickle.load(open("df_cosine_dogs.pkl", "rb"))
    images_folder_path = "dataset/petfinder-adoption-prediction/train_images/"
    top_cos = sorted(list(enumerate(df_cosine_dogs[new_index])), key=lambda x: x[1], reverse=True)[1:top+1]
    rec_list    = [i[0] for i in top_cos]
    cos_list    = [i[1] for i in top_cos]
    
    print("Meet: \n")
    get_details(pet_index, df)
    
    print("\n")
    print("\n Please check out these listings as well: \n")
    for count, rec in enumerate(rec_list):
        pet_index_rec = get_old_index(rec, df)
        get_details(pet_index_rec, df)
        print("Note: Cosine Similarity =",cos_list[count], "\n\n")
        
        
def cats_reco(pet_index, new_index, df, top):
    """
    Used by get_similar_pet_listins to get top n similar cats
    
    """
    
    df_cosine_cats = pickle.load(open("df_cosine_cats.pkl", "rb"))
    top_cos = sorted(list(enumerate(df_cosine_cats[new_index])), key=lambda x: x[1], reverse=True)[1:top+1]
    rec_list    = [i[0] for i in top_cos]
    cos_list    = [i[1] for i in top_cos]
    
    print("Meet: \n")
    get_details(pet_index, df)
    
    print("\n")
    print("Please check out these listings as well: \n")
    for count, rec in enumerate(rec_list):
        pet_index_rec = get_old_index(rec, df)
        get_details(pet_index_rec, df)
        print("Note: Cosine Similarity =",cos_list[count], "\n\n")

def get_old_index(new_index, subset):
    """
    Takes in the new_index of a pet in subset (df) then return the index of that pet in the pets_reco df
    
    """
    return list(subset.index)[new_index]   


def get_details(i, df):
    """
    Takes in index i of the pet in df then prints pets info
    
    """
    print("Name:", df.name.loc[i])
    print("Gender:", df.gender.loc[i])
    print("Age in months:", df.age.loc[i])
    print("Breed:", df.breed1_desc.loc[i].title().replace("_"," "),df.breed2_desc.loc[i].title().replace("_"," "))
    print("Color/s:", df.color1_desc.loc[i].title(), df.color2_desc.loc[i].title(), df.color3_desc.loc[i].title())
    print("Fur Length:", df.fur_length.loc[i])
    print("Vaccinated:", df.vaccinated.loc[i])
    print("Dewormed:", df.dewormed.loc[i])
    print("Spayed or Neutered:", df.sterilized.loc[i])
    print("Health:", df.health.loc[i])
    print("No. of Pets in this Listing:", df.quantity.loc[i])
    if df.fee.loc[i] == 0:
        print("Adoption Fee: FREE")
    else:
        print("Adoption Fee: MYR", round(df.fee.loc[i],2))
    print("Location :", df.state_desc.loc[i].title().replace("_"," "))
    print("Description :", df.description.loc[i])
    print_images(i, df)
    
    
    
def print_images(i, df):
    """
    Takes in index i of the pet in df then prints image
    
    """
        
    images_folder_path = "dataset/petfinder-adoption-prediction/train_images/"
    plt.imshow(cv2.cvtColor(cv2.imread(images_folder_path+df.filename[i]), cv2.COLOR_BGR2RGB),);
    plt.axis("off");
    plt.show()
    