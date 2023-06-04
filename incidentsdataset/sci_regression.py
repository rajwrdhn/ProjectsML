import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from scipy.sparse import load_npz
import argparse
import os
from datetime import datetime
import time
from math import sqrt


def load_data():
    data = load_npz('./npz_incidents.csv.npz')
    
    X = data[:,:-1].todense()
    y = data[:,-1].todense()

    return X, y

def evaluate(model, X_test, y_test):
    predicted = model.predict(np.asarray(X_test))
    
    mse = mean_squared_error(np.asarray(y_test), np.asarray(predicted))
    mae = mean_absolute_error(np.asarray(y_test), np.asarray(predicted))
    r2 = r2_score(np.asarray(y_test), np.asarray(predicted))
    print('=== Result for ===')
    print('MSE: ', mse)
    print('RMSE:', sqrt(mse))
    print('MAE: ', mae)
    print('R2:  ', r2)
        
    return {'MSE': mse, 'RMSE': sqrt(mse), 'MAE': mae, 'R2': r2}

def main():
    parser = argparse.ArgumentParser(description='Trains a regressor using one of the existing linear models of scipy, prints evaluation metrics for the predictions.')
    parser.add_argument('regressor', nargs='?', default='LIN', help='One of LIN, RF, SVR, or CONST, denoting the regressor to be used.')  
    args = parser.parse_args()
    
    X, y = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    if args.regressor.upper() in ['LIN', 'CONST', 'RF', 'SVR']:
        import scipy_wrap
        model = scipy_wrap.ScipyModel(args.regressor)
    else:
        print('Unknown regressor type! Exiting...')
        return
    
    starttime = time.time()

    model.train(X_train, y_train)
    res = evaluate(model, X_test, y_test)

if __name__ == '__main__':
    main()