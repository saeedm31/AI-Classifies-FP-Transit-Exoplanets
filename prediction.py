import joblib


def predict(data):
    clf = joblib.load("rf_model.sav")
    return clf.predict(data), clf.predict_proba(data)
