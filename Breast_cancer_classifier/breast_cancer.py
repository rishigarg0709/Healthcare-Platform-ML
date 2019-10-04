import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('data.csv')
x = dataset.iloc[:, 2: 32].values
y = dataset.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

# K-Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2)
classifier.fit(x_train, y_train)

y_knn = classifier.predict(x_test) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_knn)

acc_knn = (cm[0][0] + cm[1][1])*100/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

# Artificial Neural Network
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classifier_ann = Sequential()
classifier_ann.add(Dense(units = 15, activation = 'relu', input_dim = 30))
classifier_ann.add(Dense(units = 15, activation = 'relu'))
classifier_ann.add(Dense(units = 1, activation = 'sigmoid'))
classifier_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier_ann.fit(x_train, y_train, batch_size = 8, epochs = 100)

y_ann = classifier_ann.predict(x_test) 
y_ann = (y_ann > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_ann)

acc_ann = (cm[0][0] + cm[1][1])*100/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
