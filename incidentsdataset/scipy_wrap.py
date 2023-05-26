import pandas as pd

import sklearn.linear_model as lin_models
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
import sklearn.dummy
import pickle

import matplotlib.pyplot as plt
import shutil
import os

class ScipyModel():
    def __init__(self, log_basepath, regressor_suffix, **kwargs):
        self.log_basepath = log_basepath
        self.suffix = regressor_suffix.upper()

        if self.suffix == 'LIN':
            self.model = lin_models.LinearRegression(**kwargs)
        elif self.suffix == 'LOG':
            self.model = lin_models.LogisticRegression(**kwargs)
        elif self.suffix == 'CONST':
            self.model = sklearn.dummy.DummyRegressor(**kwargs)
        elif self.suffix == 'RF':
            self.model = RandomForestRegressor(n_estimators = 100, random_state = 42, max_depth = 30, **kwargs)
        elif self.suffix == 'SVR':
            self.model = SVR(epsilon=0.1, tol=0.0001, **kwargs)
        else:
            self.model = None