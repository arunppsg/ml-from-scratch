#!/usr/bin/python3
import pandas as pd
import numpy as np

def sigmoid(x):
    d = 1 + np.exp(-1*x)
    return 1 / d

def cross_entropy_loss(y_pred, y_true):
    return -1 * ((y_true * np.log(y_pred)) + ((1 - y_true) * np.log(1 - y_pred)))

def fit(X, y, epoches=50):
    w = np.random.uniform(-1, 1, 3)
    eta = 0.01
    losses = []
    for epoch in range(epoches): 
        loss = 0
        for i in range(0, len(X)):
            y_pred = sigmoid(np.sum(X[i] * w[:-1]) + w[-1])
            y_true = y[i]
            loss += cross_entropy_loss(y_pred, y_true)
            gradient = (y_pred - y_true) * np.append(X[i], 1)
            w = w - eta * gradient
        losses.append(loss) 
    return w, losses

if __name__ == "__main__":
    df = pd.read_csv('../data/classification_data.csv')
    X = df[['x1', 'x2']].values
    y = df['y'].values
    w, losses = fit(X, y)
    print (losses)
