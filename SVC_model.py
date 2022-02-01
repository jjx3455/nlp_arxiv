""" Buidling a classical ML model for text categorisation. 
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.metrics import multilabel_confusion_matrix


# Starting the looging

PATH_TO_LOG = "log/"
if not os.path.exists(PATH_TO_LOG):
    os.mkdir(PATH_TO_LOG)


logging.basicConfig(
    filename=PATH_TO_LOG + "log_SVM.log",
    filemode="a",
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


logging.info("New training")

# Loading the data
PATH_TO_METADATA_FOLDER = "data/metadata/"
PATH_TO_METADATA = PATH_TO_METADATA_FOLDER + "df_maths.json"
PATH_TO_METADATA_MSC = PATH_TO_METADATA_FOLDER + "df_maths_msc.json"
DICT_TYPE = {"id": str}

df_maths = pd.read_json(PATH_TO_METADATA, dtype=DICT_TYPE)

print("Data loaded")
print("The shape of the data is", df_maths.shape)
logging.info(f"The shape of the data is {df_maths.shape}")

# train, val test split
X = df_maths[["abstract"]]
targets = ["main_math_categories", "math_categories"]
Y_text = df_maths[targets]
X_train, X_int, y_train_text, y_int = train_test_split(
    X, Y_text, test_size=0.4, random_state=42
)
X_val, X_text, y_val_text, y_test_text = train_test_split(
    X_int, y_int, test_size=0.5, random_state=42
)
print("Data splitted")

# resampling to take into account the imbalance between the different classes.
# The strategy adopted here is to maximally resampled all labels: I consider the
# most frequent class, and upsample all other classes to get the same amount of samples
# for each label.

df_train = X_train
df_train[targets] = y_train_text
frequencies = df_train["main_math_categories"].value_counts()
lowest = frequencies[-1]
highest = frequencies[0]
labels = df_train["main_math_categories"].unique()
df_downsampled = pd.DataFrame()
df_upsampled = pd.DataFrame()
for label in labels:
    mask_label = df_train["main_math_categories"] == label
    df_label = df_train.loc[mask_label, :]
    df_down = resample(df_label, n_samples=lowest)
    df_up = resample(df_label, n_samples=highest)
    df_downsampled = pd.concat([df_downsampled, df_down], axis=0)
    df_upsampled = pd.concat([df_upsampled, df_up], axis=0)


print("Data resampled")


# Selecting upsampled data
X_train = df_upsampled["abstract"]
y_train_text_multi = df_upsampled["math_categories"]

logging.info(f"Train size: {X_train.shape}")

# Multilabel target encoding
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train_text_multi)
logging.info(f"Label list: {mlb.classes_}")


# Defining the classifier
# CountVectorizer: Convert a collection of text documents to a matrix of token counts.
# TFID : Transform a count matrix to a normalized tf or tf-idf representation. Tf means term-frequency
# while tf-idf means term-frequency times inverse document-frequency. This is a common term weighting
# scheme in information retrieval, that has also found good use in document classification.
# OneVsRestClassifier: One-vs-the-rest (OvR) multiclass strategy. Also known as one-vs-all, this strategy consists
# in fitting one classifier per class. For each classifier, the class is fitted against all
# the other classes. In addition to its computational efficiency (only n_classes classifiers are
# needed), one advantage of this approach is its interpretability. Since each class is represented
# by one and one classifier only, it is possible to gain knowledge about the class by inspecting its
# corresponding classifier. This is the most commonly used strategy for multiclass classification and
# is a fair default choice.
# LinearSVC : Linear Support Vector Classification.
classifier = Pipeline(
    [
        ("vectorizer", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", OneVsRestClassifier(LinearSVC())),
    ]
)

# pipeline parameters prepared for a grid search
pipe_parameters = [
    {
        "vectorizer__max_features": [1500, 2000, 3000, 5000, 10000, 25000, 50000],
        "clf__estimator__C": [
            1,
        ],
    },
    {
        "vectorizer__max_features": [
            None,
        ],
        "clf__estimator__C": [0.1, 1, 5, 10],
    },
]
logging.info(f"Pipe parameters: {pipe_parameters}")


# Gridsearch definition
grid = GridSearchCV(classifier, pipe_parameters, cv=2)

grid.fit(X_train, y_train)

print("The best parameters are", grid.best_params_)
logging.info(f"The best parameters are {grid.best_params_}")


PATH_TO_MODEL_FOLDER = "model/"
PATH_TO_MODEL = PATH_TO_MODEL_FOLDER + "model_SVC.sav"

if not os.path.exists(PATH_TO_MODEL_FOLDER):
    os.mkdir(PATH_TO_MODEL_FOLDER)

joblib.dump(grid, PATH_TO_MODEL)

print("Model saved")


# getting the metrics per class. I redefine here the basic metrics based on the multi-label confusion matrix.
y_val = mlb.transform(y_val_text["math_categories"])
y_pred = grid.predict(X_val.squeeze())
cm = multilabel_confusion_matrix(y_val, y_pred)


def accuracy(matrix):
    return np.trace(matrix) / np.sum(matrix)


def precision(matrix):
    return matrix[1, 1] / (matrix[0, 1] + matrix[1, 1])


def recall(matrix):
    return matrix[1, 1] / (matrix[1, 0] + matrix[1, 1])


def F1(matrix):
    return 2 / (1 / precision(matrix) + 1 / recall(matrix))


dict_results = dict()
for i in range(len(mlb.classes_)):
    matrix = cm[i, :, :]
    dict_results[f"{mlb.classes_[i]}"] = {
        "Accuracy": round(accuracy(matrix), 2),
        "Precision:": round(precision(matrix), 2),
        "Recall": round(recall(matrix), 2),
        "F1": round(F1(matrix), 2),
    }

list_dict = list(dict_results.values())
list_metrics = list(list_dict[0].keys())
dict_results["global"] = dict()
for metric in list_metrics:
    list_metric = []
    for dict in list_dict:
        list_metric.append(dict[f"{metric}"])
    dict_results["global"][f"{metric}"] = np.mean(list_metric)
    print(f"The average {metric} is", dict_results["global"][f"{metric}"])

logging.info(f"Metrics of the model {dict_results}")
