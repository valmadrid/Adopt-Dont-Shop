import pandas as pd
import pickle
import cv2
import matplotlib.pyplot as plt

import preprocess as pre

def get_similar_pet_listings(i, top=2):
    """
    Takes in an index (integer) of the pet in pets_reco table then returns 
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
    
    """
    
    df_cosine_dogs = pickle.load(open("df_cosine_dogs_2.pkl", "rb"))
    images_folder_path = "dataset/petfinder-adoption-prediction/train_images/"
#     df_scores = list(enumerate(df_cosine_dogs[new_index]))
#     df_scores = sorted(df_scores, key=lambda x: x[1], reverse=True)
#     df_scores = df_scores[1:top+1]
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
        print("Note: Cosine Similarity =",round(cos_list[count]*100,4), "% \n\n")
        
        
def cats_reco(pet_index, new_index, df, top):
    """
    
    """
    
    df_cosine_cats = pickle.load(open("df_cosine_cats_2.pkl", "rb"))
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
        print("Note: Cosine Similarity =",round(cos_list[count]*100,4), "% \n\n")

def get_old_index(new_index, subset):
    """
    
    """
    return list(subset.index)[new_index]   


def get_details(i, df):
    """
    
    """
#     'adoption_speed', 'pet_id', 'type', 'name', 'age', 'breed1',
#        'breed1_desc', 'breed2', 'breed2_desc', 'gender', 'color1',
#        'color1_desc', 'color2', 'color2_desc', 'color3', 'color3_desc',
#        'maturity_size', 'fur_length', 'vaccinated', 'dewormed', 'sterilized',
#        'health', 'quantity', 'fee', 'state', 'state_desc', 'rescuer_id',
#        'video_count', 'photo_count', 'filename', 'description', 'desc_score',
#        'desc_magnitude', 'desc_sentences_score_sum',
#        'desc_sentences_score_avg'
    
    
#     images_folder_path = "dataset/petfinder-adoption-prediction/train_images/"
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
    
    """
        
    images_folder_path = "dataset/petfinder-adoption-prediction/train_images/"
    print("Name:", df.name.loc[i])
    plt.imshow(cv2.cvtColor(cv2.imread(images_folder_path+df.filename[i]), cv2.COLOR_BGR2RGB),);
    plt.axis("off");
    plt.show()
    