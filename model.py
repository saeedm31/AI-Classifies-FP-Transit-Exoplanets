import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# random seed
seed = 42

# Read original dataset
iris_df = pd.read_csv("FP_new.csv")
iris_df.sample(frac=1, random_state=seed)

# selecting features and target data
X = iris_df[["koi_period", "koi_duration", "koi_ror", "koi_srad"]]
y = iris_df["info_status"]  # Remove the extra brackets to make it 1D

# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)

# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=seed)

# train the classifier on the training data
clf.fit(X_train, y_train)

# predict on the test set
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save model parameters instead of the full model to avoid numpy issues
model_params = {
    'feature_names': list(X.columns),
    'classes': clf.classes_.tolist(),
    'n_estimators': clf.n_estimators,
    'estimators': []
}

# Extract individual tree parameters
for i, tree in enumerate(clf.estimators_):
    tree_params = {
        'tree': tree.tree_,
        'n_features': tree.n_features_,
        'n_classes': tree.n_classes_,
        'n_outputs': tree.n_outputs_
    }
    model_params['estimators'].append(tree_params)

# Save the model parameters
with open("rf_model_params.pkl", "wb") as f:
    pickle.dump(model_params, f)

print("Model parameters saved as rf_model_params.pkl")

# Also save a simple version for testing
simple_model = {
    'feature_names': list(X.columns),
    'classes': clf.classes_.tolist(),
    'predict': lambda X: clf.predict(X),
    'predict_proba': lambda X: clf.predict_proba(X)
}

with open("rf_model_simple.pkl", "wb") as f:
    pickle.dump(simple_model, f)

print("Simple model saved as rf_model_simple.pkl")
