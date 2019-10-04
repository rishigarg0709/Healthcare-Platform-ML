import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('diabetes.csv')
x = dataset.iloc[:, 0: 8].values
y = dataset.iloc[:, 8].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

# Artificial Neural Network
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classifier_ann = Sequential()
classifier_ann.add(Dense(units = 7, activation = 'relu', input_dim = 8))
# classifier_ann.add(Dense(units = 5, activation = 'relu'))
classifier_ann.add(Dense(units = 1, activation = 'sigmoid'))
classifier_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier_ann.fit(x_train, y_train, batch_size = 8, epochs = 100)

y_ann = classifier_ann.predict(x_test) 
y_ann = (y_ann > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_ann)

acc_ann = (cm[0][0] + cm[1][1])*100/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])


# K- Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2)
classifier_knn.fit(x_train, y_train)

y_knn = classifier_knn.predict(x_test) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_knn)

acc_knn = (cm[0][0] + cm[1][1])*100/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier_lg = LogisticRegression()
classifier_lg.fit(x_train, y_train)

y_lg = classifier_lg.predict(x_test) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_lg)

acc_lg = (cm[0][0] + cm[1][1])*100/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

import pickle

if acc_ann >= acc_knn and acc_ann >= acc_lg:
    with open("AAAAAAAAAAAAA", "wb") as f:
        pickle.dump(classifier_ann, f)
elif acc_knn >= acc_ann and acc_knn >= acc_lg:
     with open("Diabetes", "wb") as f:
        pickle.dump(classifier_knn, f)
elif acc_lg >= acc_ann and acc_lg >= acc_knn:
     with open("Diabetes", "wb") as f:
        pickle.dump(classifier_lg, f)        
      


# Bar plot
x = np.arange(3)
plt.bar(x, height = [acc_ann, acc_knn, acc_lg])
plt.xticks(x, ['ANN', 'K-nn', 'LG'])

# Parallel Plot
from pandas.plotting import parallel_coordinates

f = (
    dataset.iloc[:, 0:8]
        .loc[dataset['Outcome'].isin([1, 0])]
        .applymap(lambda v: int(v) if v else np.nan)
        .dropna()
)

f['Outcome'] = dataset['Outcome']
f = f.sample(50)

parallel_coordinates(f, 'Outcome', colormap = 'cool')