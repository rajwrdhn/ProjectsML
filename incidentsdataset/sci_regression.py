import pandas as pd

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from scipy.sparse import load_npz
import argparse
import os
from datetime import datetime
import time

def load_data():
    data = load_npz('incidentsdataset/npz_incidents.csv.npz')
    
    X = data[:,:-1].todense()
    y = data[:,-1].todense()

    return X, y

def main():
    parser = argparse.ArgumentParser(description='Trains a regressor using either keras, or one of the existing linear models of scipy, prints evaluation metrics and plots the predictions.')
    parser.add_argument('regressor', nargs='?', default='LIN', help='One of LIN, LOG, RF, SVR, or CONST, denoting the regressor to be used.')  
    args = parser.parse_args()
    
    X, y = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    if args.regressor.upper() in ['LIN', 'LOG', 'CONST', 'RF', 'SVR']:
        import scipy_wrap
        model = scipy_wrap.ScipyModel(args.regressor)
    else:
        print('Unknown regressor type! Exiting...')
        return
    
    starttime = time.time()


if __name__ == '__main__':
    main()