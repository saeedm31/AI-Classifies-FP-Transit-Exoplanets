

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import joblib

# random seed
seed = 42

# Read original dataset
iris_df = pd.read_csv("FP_new.csv")
iris_df.sample(frac=1, random_state=seed)

# selecting features and target data
X = iris_df[["koi_period", "koi_duration", "koi_ror", "koi_srad"]]
y = iris_df[["info_status"]]

# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)

# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)


# train the classifier on the training data
clf.fit(X_train, y_train)

# predict on the test set
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# recall_score = recall_score(y_test, y_pred) 
# print(f"Recall: {recall_score}")  

# save the model to disk
joblib.dump(clf, "rf_model.sav")
