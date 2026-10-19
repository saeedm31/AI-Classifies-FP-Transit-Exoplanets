"""Core logic for the Kepler false-positive classifier.

Shared by the Streamlit app (app.py) and the training/evaluation script (train.py),
so the model in the app is always built exactly the way it was evaluated.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "FP_new.csv"
LABEL = "info_status"          # "Confirmed" or "FP"
POSITIVE = "FP"                # the class we predict a probability for

# Raw columns, named exactly as in the NASA Exoplanet Archive KOI table,
# so a table downloaded from the archive can be scored directly.
INPUT_FEATURES = [
    "koi_period",    # orbital period [days]
    "koi_duration",  # transit duration [hours]
    "koi_depth",     # transit depth [ppm]
    "koi_impact",    # impact parameter b
    "koi_ror",       # planet-to-star radius ratio Rp/R*
    "koi_dor",       # planet-star distance over star radius a/R*
    "koi_srad",      # stellar radius [solar radii]
    "koi_smass",     # stellar mass [solar masses]
    "koi_slogg",     # stellar surface gravity log10(cm/s^2)
]

# Derived feature: b + Rp/R*. Above 1 the planet only clips the edge of the
# star (a "grazing" transit), a classic signature of eclipsing binaries.
DERIVED_FEATURES = ["transit_shape"]

MODEL_FEATURES = INPUT_FEATURES + DERIVED_FEATURES

FEATURE_INFO = {
    "koi_period": ("Orbital period", "days"),
    "koi_duration": ("Transit duration", "hours"),
    "koi_depth": ("Transit depth", "ppm"),
    "koi_impact": ("Impact parameter b", ""),
    "koi_ror": ("Planet / star radius ratio", ""),
    "koi_dor": ("Orbit distance / star radius", ""),
    "koi_srad": ("Stellar radius", "R☉"),
    "koi_smass": ("Stellar mass", "M☉"),
    "koi_slogg": ("Stellar surface gravity", "log g"),
    "transit_shape": ("Grazing indicator (b + Rp/R*)", ""),
}

RANDOM_STATE = 42


def add_features(df):
    """Return a copy of df with the derived model features added."""
    out = df.copy()
    out["transit_shape"] = out["koi_impact"] + out["koi_ror"]
    return out


def load_clean_data(path=DATA_PATH):
    """Load the labelled Kepler objects and remove duplicates.

    The raw file contains exact duplicate rows (mostly confirmed planets) and a
    few objects with identical measurements but different labels. Keeping them
    lets copies of the same object land in both training and test folds, which
    inflates accuracy, so both are dropped.
    """
    raw = pd.read_csv(path)
    df = raw[[LABEL] + INPUT_FEATURES].drop_duplicates()
    conflicting = df.groupby(INPUT_FEATURES)[LABEL].transform("nunique") > 1
    df = df[~conflicting].reset_index(drop=True)
    stats = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(df)),
        "removed_duplicates": int(len(raw) - len(df) - conflicting.sum()),
        "removed_conflicting": int(conflicting.sum()),
        "class_counts": df[LABEL].value_counts().to_dict(),
    }
    return add_features(df), stats


def build_model():
    """Random forest with class weights, since false positives outnumber planets."""
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train(df):
    """Fit the final model on all clean data."""
    model = build_model()
    model.fit(df[MODEL_FEATURES], (df[LABEL] == POSITIVE).astype(int))
    return model


def predict_fp_probability(model, df):
    """Probability that each row is a false positive (0 to 1)."""
    X = add_features(df)[MODEL_FEATURES]
    return model.predict_proba(X)[:, 1]


def red_flags(row):
    """Simple physical warning signs, shown next to the model's verdict.

    These are rules of thumb an astronomer would check by eye. They are not
    how the model decides, but they help explain typical false positives.
    """
    flags = []
    if row["koi_impact"] + row["koi_ror"] > 1:
        flags.append("Grazing transit (b + Rp/R* > 1): the object only clips the edge of the star, "
                     "which is typical of eclipsing binary stars.")
    if row["koi_ror"] > 0.3:
        flags.append("Very large radius ratio (Rp/R* > 0.3): the companion is probably too big to be a planet.")
    if row["koi_depth"] > 30000:
        flags.append("Very deep transit (> 3%): deeper than almost any planet causes; points to a stellar companion.")
    return flags
