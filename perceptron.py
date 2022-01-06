#!/usr/bin/python3

"""
Feed-forward neural networks
"""

# TODO Enable batch mode

import pandas as pd
import numpy as np

def mean_squared_error_loss(y_pred, y_true):
    return (y_pred - y_true) ** 2 

def fit(X, y, epoches=50):
    W = np.random.uniform(-1, 1, 4)
    eta = 0.0001
    losses = []
    for epoch in range(epoches):
        loss = 0
        for i in range(0, len(X)):
            y_pred = np.sum(W * X[i])
            y_true = y[i]
            loss += mean_squared_error_loss(y_pred, y_true)
            gradient = 2 * (y_pred - y_true) * X[i]
            W = W - eta * gradient
        losses.append(loss)
    return W, losses

if __name__ == "__main__":
    df = pd.read_csv('../data/regression_data.csv')
    X = df[['x1', 'x2', 'x3']].values
    y = df['y'].values
    bias = np.ones((len(X), 1))
    X = np.append(X, bias, axis=1)
    w, losses = fit(X, y)
    for i in range(0, 50):
        if i % 2 == 0:
            print (losses[i])
