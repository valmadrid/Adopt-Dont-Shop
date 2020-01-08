import pandas as pd
import numpy as np
import pandas_profiling
import itertools

import seaborn as sns
import matplotlib.pyplot as plt


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
    df.drop(["check"], axis=1, inplace=True)
    
    return df


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
    df.drop(["color_black", "color3_check", "check1", "check2"], axis=1, inplace=True)

    return df


def drop_columns(df, cols_to_delete):
    """
    Takes in:
    df = dataframe
    cols_to_delete = list of column header names to delete
    """

    df.drop(columns=cols_to_delete, axis=1, inplace=True)

    return df


def drop_rows(df, rows_to_delete):
    """
    Takes in:
    df = dataframe
    rows_to_delete = list of indices to delete
    """

    df.drop(index=rows_to_delete, axis=0, inplace=True)

    return df


def plot_bar(df, x, y, title, rotate = False):
    """
    Takes in:
    df = table to plot
    x = x-axis string
    x =y-axis string
    title = title of the bar graph
    
    """

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
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = int(p.get_height())
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

        
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
    
    
def plot_bar_hue(df, x, y, hue, title, rotate = False):
    """
    Takes in:
    df = table to plot
    x = x-axis string
    x =y-axis string
    title = title of the bar graph
    
    """

    _barplot = sns.barplot(x=x, y=y, hue=hue, data=df)
    show_values_on_bars(_barplot)
    plt.title(title)
    if rotate:
        plt.xticks(rotation=65, fontsize=10)
    plt.show()