
import pandas as pd

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from statistics import mode

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, MultiLabelBinarizer
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV, LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, plot_confusion_matrix


from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def minmax_scaler(X_train, X_test, cols_to_scale):
    """
    Takes in X train and test and scale then columns specified in cols_to_scale (list)
    
    """

    X_train_cat = X_train.drop(cols_to_scale, axis=1)
    X_test_cat = X_test.drop(cols_to_scale, axis=1)

    X_train_cont = X_train[cols_to_scale]
    X_test_cont = X_test[cols_to_scale]

    scaler = MinMaxScaler()
    scaler.fit(X_train_cont)
    X_train_cont_scaled = scaler.transform(X_train_cont)
    X_test_cont_scaled = scaler.transform(X_test_cont)

    X_train_scaled = pd.DataFrame(X_train_cont_scaled,
                                  columns=X_train_cont.columns,
                                  index=X_train_cont.index)
    X_train_scaled = pd.concat([X_train_scaled, X_train_cat], axis=1)

    X_test_scaled = pd.DataFrame(X_test_cont_scaled,
                                 columns=X_test_cont.columns,
                                 index=X_test_cont.index)
    X_test_scaled = pd.concat([X_test_scaled, X_test_cat], axis=1)

    return X_train_scaled, X_test_scaled


def run_clf(classifiers_list, X_train, X_test, y_train, y_test):
    """
    Takes in a classifier list and X_train, X_test, y_train, y_test arrays then runs score_clf() for each classifier
    
    """

    score_table = pd.DataFrame()

    for count, clf in enumerate(classifiers_list):
        acc_score, cohen_kappa = score_clf(X_train, X_test, y_train, y_test, count, clf)
        score_table.at[count, "estimator"] = clf.__class__.__name__
        score_table.at[count, "accuracy_score"] = acc_score
        score_table.at[count, "quadratic_cohen_kappa_score"] = cohen_kappa

    return score_table


def score_clf(X_train, X_test, y_train, y_test, count, estimator, **kwargs):
    """
    Runs estimator on X_train, X_test, y_train, y_test arrays then prints and returns accuracy and quadratic Cohen-Kappa scores
    """

    model = estimator
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    acc_score = round(accuracy_score(y_test, y_test_pred), 4)
    cohen_kappa = round(cohen_kappa_score(y_test, y_test_pred, weights="quadratic"), 4)

    print("{}. {}:".format(count + 1, estimator.__class__.__name__))
    print("Accuracy: {}".format(acc_score))
    print("Quadratic Cohen's kappa score: {}".format(cohen_kappa))
    print("\n")
    print("Confusion Matrix: \n")
    plot_confusion_matrix(estimator, X_test, y_test, normalize="true", cmap=plt.cm.GnBu);
    plt.show();
    print("\n")
    return acc_score, cohen_kappa


def run_GridSearchCV(X_train, X_test, y_train, y_test, model, param_grid, scoring="accuracy", cv = 5, verbose=1):
    opt_model = GridSearchCV(model, param_grid = param_grid, scoring = scoring, cv = cv, n_jobs = -1, return_train_score = True)
    print("Running GridSearchCV...")
    opt_model.fit(X_train, y_train)
    best_model = opt_model.best_estimator_
    print("Done.")
    print("---------------------------------------------------------------------")
    print("Best Parameters:", opt_model.best_params_)
    
    evaluate(best_model, X_train, X_test, y_train, y_test)

    cv_results = pd.DataFrame(opt_model.cv_results_)
    columns = ["rank_test_score", "params", "mean_train_score","std_train_score", "mean_test_score", "std_test_score"]
    cv_results_top10 = cv_results[columns].sort_values("rank_test_score").head(10)
    
    return cv_results_top10, best_model


def evaluate(estimator, X_train, X_test, y_train, y_test): 
    y_test_pred = estimator.predict(X_test)
    probas = estimator.predict_proba(X_test)
    print("---------------------------------------------------------------------")
    print("Accuracy score: ", round(accuracy_score(y_test, y_test_pred), 4))
    print("Quadratic Cohen's kappa score: ", round(cohen_kappa_score(y_test, y_test_pred, weights="quadratic"), 4))
    print("---------------------------------------------------------------------")
    print("Classification Report: \n", classification_report(y_test, y_test_pred))
    print("---------------------------------------------------------------------")
    print("Confusion Matrix: \n")
    plot_confusion_matrix(estimator, X_test, y_test, normalize="true", cmap=plt.cm.GnBu);
    plt.show();