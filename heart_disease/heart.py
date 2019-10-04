# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv('heart.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# K-Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2)
classifier.fit(x_train, y_train)

y_knn = classifier.predict(x_test) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_knn)

acc_knn = (cm[0][0] + cm[1][1])*100/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

# Logistic regression
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(x_train, y_train)

y_lr = classifier_lr.predict(x_test) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_lr)

acc_lr = (cm[0][0] + cm[1][1])*100/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

# Support Vector Machine
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'rbf', random_state = 0)
classifier_svm.fit(x_train, y_train)

y_svm = classifier_svm.predict(x_test) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_svm)

acc_svm = (cm[0][0] + cm[1][1])*100/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

# Random Forest Classifier 
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_rf.fit(x_train, y_train)

y_rf = classifier_rf.predict(x_test) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_rf)

acc_rf = (cm[0][0] + cm[1][1])*100/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

# Artificial Neural Network
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classifier_ann = Sequential()
classifier_ann.add(Dense(units = 8, activation = 'relu', input_dim = 13))
# classifier_ann.add(Dense(units = 5, activation = 'relu'))
classifier_ann.add(Dense(units = 1, activation = 'sigmoid'))
classifier_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier_ann.fit(x_train, y_train, batch_size = 8, epochs = 100)

y_ann = classifier_ann.predict(x_test) 
y_ann = (y_ann > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_ann)

acc_ann = (cm[0][0] + cm[1][1])*100/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
