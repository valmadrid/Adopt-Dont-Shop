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
    merged = pd.merge(df, breeds, left_on=column, right_on="BreedID")
    merged.rename(mapper={"BreedName": column + "_desc"}, axis=1, inplace=True)
    merged.drop(["BreedID", "Type"], axis=1, inplace=True)
    return merged


def get_color(df, colors, column):
    """
    Takes:
        df = left table to merge
        colors = right table to merge
        column = column in df to merge on
    Returns df with color description
    """
    merged = pd.merge(df, colors, left_on=column, right_on="ColorID")
    merged.rename(mapper={"ColorName": column + "_desc"}, axis=1, inplace=True)
    merged.drop(["ColorID"], axis=1, inplace=True)
    return merged


def get_state(df, states, column):
    """
    Takes:
        df = left table to merge
        states = right table to merge
        column = column in df to merge on
    Returns df with state description
    """
    merged = pd.merge(df, states, left_on=column, right_on="StateID")
    merged.rename(mapper={"StateName": column + "_desc"}, axis=1, inplace=True)
    merged.drop(["StateID"], axis=1, inplace=True)
    return merged
