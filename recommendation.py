import pandas as pd
import pickle

import preprocess as pre

def get_reco(i, df, top=2):
    
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

def dogs_reco(old_index, new_index, df, top):
    
    df_cosine_dogs = pickle.load(open("df_cosine_dogs.pkl", "rb"))
    images_folder_path = "dataset/petfinder-adoption-prediction/train_images/"
    df_scores = list(enumerate(df_cosine_dogs[new_index]))
    df_scores = sorted(df_scores, key=lambda x: x[1], reverse=True)
    df_scores = df_scores[1:top+1]
    rec_list    = [i[0] for i in df_scores]
    sim_list    = [i[1] for i in df_scores]
    
    print("Meet.. \n")
    print("Name :", df.name.loc[old_index])
    print("Breed :", df.breed1_desc.loc[old_index].title().replace("_"," "),df.breed2_desc.loc[old_index].title().replace("_"," "))
    print("Location :", df.state_desc.loc[old_index].title().replace("_"," "))
    print("Description :", df.description.loc[old_index])
    pre.get_image(images_folder_path, df.filename.loc[old_index])
    
    print("Please check out these listings as well: \n\n")
    for count, rec in enumerate(rec_list):
        old_index_rec = get_old_index(rec, df)
        print("Name :", df.name.loc[old_index_rec])
        print("Breed :", df.breed1_desc.loc[old_index_rec].title().replace("_"," "),df.breed2_desc.loc[old_index_rec].title().replace("_"," "))
        print("Location :", df.state_desc.loc[old_index_rec].title().replace("_"," "))
        print("Description :", df.description.loc[old_index_rec])
        print("Note: Cosine similarity = ", sim_list[count])
        pre.get_image(images_folder_path, df.filename.loc[old_index_rec])
        
def cats_reco(old_index, new_index, df, top):
    
    df_cosine_cats = pickle.load(open("df_cosine_cats.pkl", "rb"))
    images_folder_path = "dataset/petfinder-adoption-prediction/train_images/"
    df_scores = list(enumerate(df_cosine_cats[new_index]))
    df_scores = sorted(df_scores, key=lambda x: x[1], reverse=True)
    df_scores = df_scores[1:top+1]
    rec_list    = [i[0] for i in df_scores]
    sim_list    = [i[1] for i in df_scores]
    
    print("Meet.. \n")
    print("Name :", df.name.loc[old_index])
    print("Breed :", df.breed1_desc.loc[old_index].title().replace("_"," "),df.breed2_desc.loc[old_index].title().replace("_"," "))
    print("Location :", df.state_desc.loc[old_index].title().replace("_"," "))
    print("Description :", df.description.loc[old_index])
    pre.get_image(images_folder_path, df.filename.loc[old_index])
    
    print("Please check out these listings as well: \n\n")
    for count, rec in enumerate(rec_list):
        old_index_rec = get_old_index(rec, df)
        print("Name :", df.name.loc[old_index_rec])
        print("Breed :", df.breed1_desc.loc[old_index_rec].title().replace("_"," "),df.breed2_desc.loc[old_index_rec].title().replace("_"," "))
        print("Location :", df.state_desc.loc[old_index_rec].title().replace("_"," "))
        print("Description :", df.description.loc[old_index_rec])
        print("Note: Cosine similarity = ", sim_list[count])
        pre.get_image(images_folder_path, df.filename.loc[old_index_rec])

def get_old_index(new_index, subset):
    return list(subset.index)[new_index]   