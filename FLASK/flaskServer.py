#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:31:22 2019

@author: sid
"""
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('healthcare_predictor.html')


import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('diabetes.csv')
x = dataset.iloc[:, 0: 8].values

sc_x = StandardScaler()
x = sc_x.fit_transform(x)

with open("Diabetes_Pickle","rb") as f:
    model = pickle.load(f)


def pred(pregnancies, glucose, bp, skinThickness, Insulin, BMI, DPF, age):
    test = np.array([[pregnancies, glucose, bp, skinThickness, Insulin, BMI, DPF, age]])
    test = pd.DataFrame(test)
    test = sc_x.transform(test)
    ans = str(model.predict(test)[0])
    return ans

    
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        pregnancies = request.form.get('pregnancies')
        glucose = request.form.get('glucose')
        bp = request.form.get('bp')
        skinThickness = request.form.get('skinThickness')
        Insulin = request.form.get('Insulin')
        BMI = request.form.get('BMI')
        DPF = request.form.get('DPF')
        age = request.form.get('age')

        return render_template('diabetes.html', results = pred(pregnancies, glucose, bp, skinThickness, Insulin, BMI, DPF, age))
    
    return render_template('diabetes.html')


if __name__ == '__main__':
    app.run(debug=True)
    
#app.run(host="127.0.0.1", port = 5000)