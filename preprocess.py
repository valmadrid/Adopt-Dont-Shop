import pandas as pd
import numpy as np
import pandas_profiling
import itertools

import seaborn as sns
import matplotlib.pyplot as plt


from IPython.display import Image

import os, json

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures


def get_breed(df, breeds, column):
    """
    Takes:
        df = left table to merge
        breeds = right table to merge
        column = column in df to merge on
    Returns df with breed description
    """
    
    breeds.BreedName = breeds.BreedName.map(lambda x: "_".join(str(x).lower().split()))
    merged = pd.merge(df, breeds, left_on=column, right_on="BreedID", how="left")
    merged.rename(mapper={"BreedName": column + "_desc"}, axis=1, inplace=True)
    merged.drop(["BreedID", "Type"], axis=1, inplace=True)
    
    return merged


def get_color(df, colors, column):
    """
    Takes in:
        df = left table to merge
        colors = right table to merge
        column = column in df to merge on
    Returns df with color description
    """
    
    colors.ColorName = colors.ColorName.map(lambda x: str(x).lower())
    merged = pd.merge(df, colors, left_on=column, right_on="ColorID", how="left")
    merged.rename(mapper={"ColorName": column + "_desc"}, axis=1, inplace=True)
    merged.drop(["ColorID"], axis=1, inplace=True)
    
    return merged


def get_state(df, states, column):
    """
    Takes in:
        df = left table to merge
        states = right table to merge
        column = column in df to merge on
    Returns df with state description
    """
    
    states.StateName = states.StateName.map(lambda x: "_".join(str(x).lower().split()))
    merged = pd.merge(df, states, left_on=column, right_on="StateID", how="left")
    merged.rename(mapper={"StateName": column + "_desc"}, axis=1, inplace=True)
    merged.drop(["StateID"], axis=1, inplace=True)
    
    return merged


def breed_dummies(df):
    """
    Converts the breed column into dummy variables.
    """
    
    df["breed1_check"] = df["breed1_desc"]
    df["breed2_check"] = df["breed2_desc"]
    df["check"] = df["breed1"] == df["breed2"]
    
    
    for i in range(len(df)):
        #if breed1 is Domestic... then change to Domestic
        if df.breed1.iloc[i] in [264, 265, 266]:
            df.breed1_check.iloc[i] = "domestic"
        #if breed2 is Domestic... then change to Domestic
        if df.breed2.iloc[i] in [264, 265, 266]:
            df.breed2_check.iloc[i] = "domestic"
        #if columns are the same then take change breed2 to nan
        if df.check.iloc[i]:
            df.breed2_check.iloc[i] = np.nan
    
    #create dummies using breed 1
    df = pd.get_dummies(df, prefix="breed", columns=['breed1_check'], dtype="float64")

    for i in range(len(df)):
        #if the breed columns are the same then skip that row
        if df.check.iloc[i]:
            continue
        #check whether breed2 has a value
        if type(df.breed2_check.iloc[i]) == str:
            #then try to check if dummy column has been created, put 1
            try:
                df["breed_" + df.breed2_check.iloc[i]].iloc[i] = 1
            #if not, then do nothing
            except:
                continue
            #if a value 1 has been placed under a dummy column then change this to nan
            else:
                df.breed2_check.iloc[i] = np.nan
    
    #create dummies using breed 2
    df = pd.get_dummies(df, prefix="breed", columns=['breed2_check'], dtype="float64")
    
    #column headers to lower case
    #df.columns = df.columns.map(lambda x: x.lower())
    
    #drop the columns created for comparison
    new_df = df.drop(["check"], axis=1)
    
    return new_df


def color_dummies(df):
    """
    Converts the breed column into dummy variables.
    """

    df["color1_check"] = df["color1_desc"]
    df["color2_check"] = df["color2_desc"]
    df["color3_check"] = df["color3_desc"]
    df["check1"] = df["color1"] == df["color2"]
    df["check2"] = df["color1"] == df["color3"]

    for i in range(len(df)):
        #if columns are the same then take change color 2 & 3 to nan
        if df.check1.iloc[i]:
            df.color2_check.iloc[i] = np.nan
        if df.check2.iloc[i]:
            df.color3_check.iloc[i] = np.nan
            
    #create dummies using color 1
    df = pd.get_dummies(df, prefix="color", columns=['color1_check'], dtype="float64")

    for i in range(len(df)):
        #check whether color2 has a value
        if type(df.color2_check.iloc[i]) == str:
            #then try to check if dummy column has been created, put 1
            try:
                df["color_" + df.color2_check.iloc[i]].iloc[i] = 1
            #if not, then do nothing
            except:
                continue
            #if a value 1 has been placed under a dummy column then change this to nan
            else:
                df.color2_check.iloc[i] = np.nan
        #check whether color3 has a value
        if type(df.color3_check.iloc[i]) == str:
            #then try to check if dummy column has been created, put 1
            try:
                df["color_" + df.color3_check.iloc[i]].iloc[i] = 1
            #if not, then do nothing
            except:
                continue
            #if a value 1 has been placed under a dummy column then change this to nan
            else:
                df.color3_check.iloc[i] = np.nan
                
    #create dummies using color 2
    df = pd.get_dummies(df, prefix="color", columns=['color2_check'], dtype="float64")
    
    #column headers to lower case
    #df.columns = df.columns.map(lambda x: x.lower())
    
    #drop the columns created for comparison
    new_df = df.drop(["color_black", "color3_check", "check1", "check2"], axis=1)

    return new_df


def drop_columns(df, cols_to_delete):
    """
    Takes in:
    df = dataframe
    cols_to_delete = list of column header names to delete
    """

    new_df = df.drop(columns=cols_to_delete, axis=1)

    return new_df


def drop_rows(df, rows_to_delete):
    """
    Takes in:
    df = dataframe
    rows_to_delete = list of indices to delete
    """

    new_df = df.drop(index=rows_to_delete, axis=0)

    return new_df


def plot_bar(df, x, y, title, figsize = False, rotate = False):
    """
    Takes in:
    df = table to plot
    x = x-axis string
    x =y-axis string
    title = title of the bar graph
    
    """
    
    if figsize:
        plt.figure(figsize = figsize);
    else:
        plt.figure(figsize = [8, 7]);
    _barplot = sns.barplot(x=x, y=y, data=df)
    show_values_on_bars(_barplot)
    plt.title(title)
    if rotate:
        plt.xticks(rotation=65, fontsize=10)
    plt.show()


def show_values_on_bars(axs):
    """
    Add labels to each bar
    
    """

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
        
        
def _show_on_single_plot(ax):
        for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                try:
                    int(p.get_height())
                except:
                    ax.text(_x, _y, "", ha="center")
                else:
                    value = int(p.get_height())
                    ax.text(_x, _y, value, ha="center")

        
def plot_hist(series, xlabel, figsize):
    """
    Takes in a series (and label for x axis and figure size) then returns a histogram.
    """
    plt.figure(figsize=figsize)
    sns.distplot(series)
    plt.xticks(fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
    
def plot_bar_hue(df, x, y, hue, title, figsize = False, rotate = False):
    """
    Takes in:
    df = table to plot
    x = x-axis string
    x =y-axis string
    title = title of the bar graph
    
    """
    
    if figsize:
        plt.figure(figsize = figsize);
    else:
        plt.figure(figsize = [8, 7]);
    _barplot = sns.barplot(x=x, y=y, hue=hue, data=df)
    show_values_on_bars(_barplot)
    plt.title(title)
    if rotate:
        plt.xticks(rotation=65, fontsize=10)
    plt.show()
    
    
def bin_breed(table):
    """
    Takes a df then bins the breed columns
    """
    
    df = table.copy()
    
    df["check"] = df.breed1 == df.breed2
    df["breed_bin"] = pd.Series()
    
    for i in range(len(df)):
        #breed1 = mixed then mixed
        if df.breed1.iloc[i] == 307:
            df.breed_bin.iloc[i] = 3
        #breed1 = domestic
        elif df.breed1.iloc[i] in [264, 265, 266]:
            #breed2 = domestic then domestic
            if df.breed2.iloc[i] in [264, 265, 266]:
                df.breed_bin.iloc[i] = 2
            #breed2 = mixed then mixed
            elif df.breed2.iloc[i] == 307:
                df.breed_bin.iloc[i] = 3
            #breed2 nan then domestic
            elif not(df.breed2.iloc[i]):
                df.breed_bin.iloc[i] = 2
            #breed2 others then pure
            else:
                df.breed_bin.iloc[i]= 1
        #breed1 others
        else:
            #breed2 = mixed then mixed
            if df.breed2.iloc[i] == 307:
                df.breed_bin.iloc[i] = 3
            #breed2 domestic then pure
            elif df.breed2.iloc[i] in [264, 265, 266]:
                df.breed_bin.iloc[i]= 1
            #breed2 nan then pure
            elif not(df.breed2.iloc[i]):
                df.breed_bin.iloc[i]= 1
            #breed2 diff then mixed
            elif df.check.iloc[i] == False:
                df.breed_bin.iloc[i] = 3
            #breed2 same then pure
            elif df.check.iloc[i] == True:
                df.breed_bin.iloc[i]= 1
                
    df["breed_bin"] = df["breed_bin"].astype("int64")
        
    return df


def get_pet_image(images_folder_path, pet_id):
    """
    Takes in folder path and pet_id (both strings), then prints the first image uploaded in Petfinder.my
    """
    
    display(Image(images_folder_path+pet_id+"-1.jpg"))

def get_image(images_folder_path, image_filename):
    """
    Takes in folder path and image_filename.jpg (both strings), then prints image_filename.jpg
    """
    
    display(Image(images_folder_path+image_filename))


def get_image_filename(df, images_folder_path):
    
    """
    Takes in df and path where the images are saved (string) and returns a df with column for image filename
    """

    image_files = []

    for r, d, f in os.walk(images_folder_path):  # r=root, d=directories, f = files
        for file in f:
            if ".jpg" in file:
                image_files.append(os.path.join(file))

    images = pd.DataFrame(data = image_files, columns = ["filename"])
    images["pet_id"] = images["filename"].map(lambda x: str(x)[:9])
    images = pd.merge(images, df, on="pet_id", how="left")

    images.to_csv("images_with_pet_details.csv", index=False)

    print("There are a total of ",len(images), " pet images.")

    unique_images = images.drop_duplicates("pet_id")[["pet_id", "filename"]]

    unique_images.to_csv("unique_images.csv", index=False)

    new_df = pd.merge(df, unique_images, on="pet_id", how="left")
    
    print(new_df.filename.isna().sum(), "pets have no images.")

    return new_df


def get_sentiment_analysis(table, sentiment_folder_path):
    """
    Takes in a table (df) and sentiment_path (folder path of the sentiment analysis) 
    then returns a df with sentiment score & magnitude and language.
    
    """
    
    df = table.copy()
#     df["desc_language"] = pd.Series()

    for i in range(len(df)):
        json_filename = df.pet_id.iloc[i] + ".json"
        
        #search for the pet_id's json file 
        try:
            with open(os.path.join(sentiment_folder_path, json_filename)) as json_file:
                sentiment = json.load(json_file)
        #if file is not existing then move to next file
        except:
            continue
        #if existing, sentiment score & magnitude and language to the df
        else: 
            score = 0
            df.at[i,"desc_score"] = sentiment["documentSentiment"]["score"]
            df.at[i,"desc_magnitude"] = sentiment["documentSentiment"]["magnitude"]
            if len(sentiment["sentences"]) > 0:
                for j in range(len(sentiment["sentences"])):
                    score += sentiment["sentences"][j]["sentiment"]["score"]
                df.at[i, "desc_sentences_score_sum"] = score
                df.at[i, "desc_sentences_score_avg"] = score / len(sentiment["sentences"])
                score = 0
#             df.desc_language.iloc[i] = sentiment["language"]
    
#     print(df.desc_score.isna().sum(),"descriptions have no sentiment analysis.")
    
    return df

def get_interaction(df, interaction_column_list, degree=3):
    """
    Takes in a df and returns a df with interaction columns based on the interaction_column_list
    """
    
    
    X_poly = df[interaction_column_list]

    poly = PolynomialFeatures(degree=degree,
                              interaction_only=True,
                              include_bias=False)
    X_interaction = poly.fit_transform(X_poly)
    new_df = pd.DataFrame(
        X_interaction,
        columns=[
            feat.replace(" ", "_")
            for feat in poly.get_feature_names(interaction_column_list)
        ],
        index=X_poly.index)

    return new_df