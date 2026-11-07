"""
Shared data loading, preprocessing and six-model definitions.
Each Pipeline refits preprocessing on training folds only to prevent leakage.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import sklearn
from sklearn.model_selection import StratifiedKFold

import config as C


def _target_encoder():
    """
    Keep encoder behaviour consistent across scikit-learn versions; from 1.9,
        random_state belongs to the CV splitter.
    """
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 9):
        return TargetEncoder(target_type="binary",
                             cv=StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE))
    return TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=C.RANDOM_STATE)

SVM_SUBSAMPLE = 15_000


# data
def load_data():
    df = pd.read_csv(C.ANALYTIC_FILE, low_memory=False)
    _, _, cat, high = C.feature_lists("B")
    for col in cat + high:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df


def get_xy(df, split, feature_set="A", verified_only=True):
    part = df[df["SPLIT"] == split]
    if verified_only:
        part = part[part["LABEL_VERIFIED"] == 1]
    num, ordn, cat, high = C.feature_lists(feature_set)
    return part[num + ordn + cat + high], part[C.TARGET].astype(int)


# preprocessing
def build_preprocessor(feature_set="A", missing="indicator"):
    """
    'indicator': median-fill numeric blanks and add missingness flags; give text blanks
        a MISSING category.
    'impute': fill numeric/text blanks with medians/modes, without flags (E1).
    """
    num, ordn, cat, high = C.feature_lists(feature_set)
    flag = missing == "indicator"
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=flag)),
                 ("scale", StandardScaler())]
    cat_imp = (SimpleImputer(strategy="constant", fill_value="MISSING") if flag
               else SimpleImputer(strategy="most_frequent"))
    high_imp = (SimpleImputer(strategy="constant", fill_value="MISSING") if flag
                else SimpleImputer(strategy="most_frequent"))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num + ordn),
        ("cat", Pipeline([("impute", cat_imp),
                          ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                   min_frequency=100, sparse_output=False))]), cat),
        # TargetEncoder cross-fits internally so no row's label contributes to its own
        # encoding.
        ("high", Pipeline([("impute", high_imp),
                           ("target", _target_encoder()),
                           ("scale", StandardScaler())]), high),
    ], verbose_feature_names_out=False)


# models + search spaces
def model_specs():
    rs = C.RANDOM_STATE
    return {
        "LR": (LogisticRegression(max_iter=3000),
               {"model__C": [0.01, 0.1, 1, 10],
                "model__class_weight": [None, "balanced"]}),
        "GNB": (GaussianNB(),
                {"model__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}),
        "KNN": (KNeighborsClassifier(n_jobs=-1),
                {"model__n_neighbors": [25, 75, 151],
                 "model__weights": ["uniform", "distance"]}),
        "SVM": (SVC(kernel="rbf", cache_size=1000, random_state=rs),
                {"model__C": [0.1, 0.3, 1, 3],
                 "model__gamma": ["scale", 0.01, 0.003, 0.001],
                 "model__class_weight": [None, "balanced"]}),
        "DT": (DecisionTreeClassifier(random_state=rs),
               {"model__max_depth": [4, 6, 8, 12],
                "model__min_samples_leaf": [20, 100],
                "model__class_weight": [None, "balanced"]}),
        "RF": (RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=rs),
               {"model__max_features": ["sqrt", 0.3],
                "model__min_samples_leaf": [3, 10],
                "model__class_weight": [None, "balanced_subsample"]}),
    }


def make_pipeline(estimator, feature_set="A", missing="indicator"):
    return Pipeline([("prep", build_preprocessor(feature_set, missing)), ("model", estimator)])


def scores(fitted, X):
    return fitted.predict_proba(X)[:, 1]
