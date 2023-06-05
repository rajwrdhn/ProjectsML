import keras
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from math import sqrt

def preprocess():
    df = pd.read_csv("hcv.csv")
    df = df.drop(columns=["Unnamed: 0"])

    #retrieve the array
    data = df.values

    # split into input and output elements
    X, y = data[:,:-1], data[:,-1]  

    #X, y = df.iloc[:,:-1], df.iloc[:,-1]

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #print(X_train.shape, y_train.shape)
    #identify outliers in the training set
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X_train)
    mask = yhat != -1
    X_train, y_train = X_train[mask, :], y_train[mask]
    #print(X_train.shape, y_train.shape)
    # fit the model
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(X_train, y_train)
    # evaluate the model
    evaluate(model, X_test, y_test)
    yhat = model.predict(X_test)
    # evaluate predictions
    #mae = mean_absolute_error(y_test, yhat)
    #print('MAE: %.3f' % mae)

def evaluate(model, X_test, y_test):
    predicted = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predicted)
    mae = mean_absolute_error(y_test, predicted)
    r2 = r2_score(y_test, predicted)
    print('=== Result for ===')
    print('MSE: ', mse)
    print('RMSE:', sqrt(mse))
    print('MAE: ', mae)
    print('R2:  ', r2)
        
    return {'MSE': mse, 'RMSE': sqrt(mse), 'MAE': mae, 'R2': r2}


if __name__ == '__main__':
    preprocess()