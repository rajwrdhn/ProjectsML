import pandas as pd

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from scipy.sparse import load_npz
import argparse
import os
from datetime import datetime
import time
from math import sqrt


def load_data():
    data = load_npz('incidentsdataset/npz_incidents.csv.npz')
    
    X = data[:,:-1].todense()
    y = data[:,-1].todense()

    return X, y

def evaluate(model, X_test, y_test):
    predicted = model.predict(X_test)
    
    mse = metrics.mean_squared_error(y_test, predicted)
    mae = metrics.mean_absolute_error(y_test, predicted)
    r2 = metrics.r2_score(y_test, predicted)
    print('=== Result for ===')
    print('MSE: ', mse)
    print('RMSE:', sqrt(mse))
    print('MAE: ', mae)
    print('R2:  ', r2)
    
    prediction_df = pd.DataFrame(data={'truth': y_test.A1, 'predicted': predicted})
    prediction_df.to_csv(os.path.join('predictions.csv'))
        
    return {'MSE': mse, 'RMSE': sqrt(mse), 'MAE': mae, 'R2': r2}

def main():
    parser = argparse.ArgumentParser(description='Trains a regressor using one of the existing linear models of scipy, prints evaluation metrics for the predictions.')
    parser.add_argument('regressor', nargs='?', default='LIN', help='One of LIN, RF, SVR, or CONST, denoting the regressor to be used.')  
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

    model.train(X_train, y_train)
    res = evaluate(model, X_test, y_test)

if __name__ == '__main__':
    main()