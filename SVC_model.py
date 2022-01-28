""" Buidling a classical ML model for text categorisation. 
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy import stats
import joblib
import os

#
PATH_TO_METADATA_FOLDER = "data/metatdata/"
PATH_TO_METADATA = PATH_TO_METADATA_FOLDER + "df_maths.json"
PATH_TO_METADATA_MSC = PATH_TO_METADATA_FOLDER + "df_maths_msc.json"
DICT_TYPE = {"id": str}

df_maths = pd.read_json(PATH_TO_METADATA, dtype=DICT_TYPE)

print("Data Loaded")
print("The shape of the data is", df_maths.shape)


X = df_maths["abstract"]
Y_text = df_maths["categories"]

mlb = MultiLabelBinarizer()
Y_multi = mlb.fit_transform(Y_text)

print("Binarized labels")

X_train, X_int, y_train, y_int = train_test_split(
    X, Y_multi, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X, Y_multi, test_size=0.5, random_state=42
)

print("Data splitted")

pipe_parameters = [
    {'vectorizer__max_features': (None, 1500),
     'clf__estimator__C': [ 0.1, ], 
     'clf__estimator__gamma': [0.0001, 1],
     'clf__estimator__kernel': ['rbf']},
    {'vectorizer__max_features': (None, 1500),
     'clf__estimator__C': [0.1, 1],
     'clf__estimator__kernel': ['linear']}
]

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(SVC())),
    ])


classifier.fit(X_train, y_train)

print("Classifier fitted")

grid = GridSearchCV(classifier, pipe_parameters, cv=2)

grid.fit(X_train, y_train)

print("The best parameters are", grid.best_params_)

mean_acc = classifier.score(X_val, y_val)

print("The mean accuracy is ", mean_acc)


PATH_TO_MODEL_FOLDER = "model/"
PATH_TO_MODEL = PATH_TO_MODEL_FOLDER + "model_SVC.sav"

if not os.path.exists(PATH_TO_MODEL_FOLDER):
    os.mkdir(PATH_TO_MODEL_FOLDER)

joblib.dump(grid, PATH_TO_MODEL)

print("Model_saved")
