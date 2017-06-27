# XGBoost

# Install xgboost following the instructions on this link: http://xgboost.readthedocs.io/en/latest/build.html#

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy as cp

np.set_printoptions(threshold='nan')

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset = dataset.fillna(dataset.median())
dataset['Embarked'] = dataset['Embarked'].fillna('S')
dataset.drop('Ticket', axis=1, inplace=True)
dataset.drop('Cabin', axis=1, inplace=True)
dataset.drop('Name', axis=1, inplace=True)
X = dataset.iloc[:, 2:9].values
y = dataset.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
T = cp.copy(X)

X[:, 6] = X[:, 2]
X[:, 2] = T[:, 6]

labelencoder_X_3 = LabelEncoder()
X[:, 2] = labelencoder_X_3.fit_transform(X[:, 2])

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 0] = labelencoder_X_2.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0,2], sparse=True)
X = onehotencoder.fit_transform(X).toarray()

X = np.delete(X, 0, 1)
X = np.delete(X, 2, 1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set
import xgboost as xgb
classifier = xgb.XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

testset = pd.read_csv('test.csv')
testset = testset.fillna(dataset.median())
testset['Embarked'] = testset['Embarked'].fillna('S')
testset.drop('Ticket', axis=1, inplace=True)
testset.drop('Cabin', axis=1, inplace=True)
testset.drop('Name', axis=1, inplace=True)
TT = testset.iloc[:, 1:].values
W = cp.copy(TT)

TT[:, 6] = TT[:, 2]
TT[:, 2] = W[:, 6]

labelencoder_TT_3 = LabelEncoder()
TT[:, 2] = labelencoder_TT_3.fit_transform(TT[:, 2])

labelencoder_TT_1 = LabelEncoder()
TT[:, 1] = labelencoder_TT_1.fit_transform(TT[:, 1])

labelencoder_TT_2 = LabelEncoder()
TT[:, 0] = labelencoder_TT_2.fit_transform(TT[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0,2], sparse=True)
TT = onehotencoder.fit_transform(TT).toarray()

TT = np.delete(TT, 0, 1)
TT = np.delete(TT, 2, 1)
prediction = classifier.predict(TT)