"""
Shared building blocks: data loading, preprocessing, and the six models.

Every model is wrapped in ONE scikit-learn Pipeline (preprocessing + model).
This matters for leakage: during cross-validation the imputers, scalers and
target encoder are re-fitted on each training fold only, so no information
from the held-out fold ever reaches the model.
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
    """Same encoder on any scikit-learn version (the random_state argument
    moved into the cv splitter in version 1.9)."""
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 9):
        return TargetEncoder(target_type="binary",
                             cv=StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE))
    return TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=C.RANDOM_STATE)

# SVC training time grows with the square of the number of rows, so the RBF
# SVM is trained on a stratified random subsample of the training years.
SVM_SUBSAMPLE = 15_000


# ------------------------------------------------------------------ data
def load_data():
    """Read the analytic table built by 01_prepare.py."""
    df = pd.read_csv(C.ANALYTIC_FILE, low_memory=False)
    _, _, cat, high = C.feature_lists("B")
    for col in cat + high:                                # text columns: plain object dtype, NaN = missing
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df


def get_xy(df, split, feature_set="A", verified_only=True):
    """Rows of one split -> feature matrix X and label vector y."""
    part = df[df["SPLIT"] == split]
    if verified_only:
        part = part[part["LABEL_VERIFIED"] == 1]
    num, ordn, cat, high = C.feature_lists(feature_set)
    return part[num + ordn + cat + high], part[C.TARGET].astype(int)


# ------------------------------------------------------------------ preprocessing
def build_preprocessor(feature_set="A", missing="indicator"):
    """
    missing="indicator": numeric blanks -> median + a 0/1 'was missing' column;
                         text blanks -> their own category "MISSING".
                         The model can learn from the fact that a field was blank.
    missing="impute"   : numeric blanks -> median, text blanks -> most common value.
                         The model cannot tell a value was missing (experiment E1).
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
        # Target encoding replaces each category with a smoothed damage rate.
        # scikit-learn's TargetEncoder cross-fits internally (5 folds) during
        # training, so a row's own label never feeds its own encoding.
        ("high", Pipeline([("impute", high_imp),
                           ("target", _target_encoder()),
                           ("scale", StandardScaler())]), high),
    ], verbose_feature_names_out=False)


# ------------------------------------------------------------------ models + search spaces
def model_specs():
    """Name -> (estimator, hyper-parameter grid). Grid keys use the 'model__' prefix."""
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
                {"model__C": [0.3, 1, 3],
                 "model__gamma": ["scale", 0.01],
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
    """Damage probability for every row (SVC and others all expose predict_proba here)."""
    return fitted.predict_proba(X)[:, 1]
