import numpy as np
import pandas as pd

import copy
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from models import LSTMModel, LinearModel, Simple, Deep, Deeper

from sklearn.model_selection import StratifiedKFold

def model_train(model, X_train, y_train, X_val, y_val):
        # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.01)
 
    n_epochs = 2000   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
 
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
 
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc

if __name__ == "__main__":

    time_series_data = pd.read_csv(r'data\time_series_data.csv')

    X = time_series_data.iloc[:, 26:31]
    y = time_series_data.iloc[:, -1]

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)

    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    cv_scores = []
    for train, test in kfold.split(X, y):
    # create model, train, and get accuracy
        model = Deep()
        acc = model_train(model, X[train], y[train], X[test], y[test])
        print("Accuracy (wide): %.2f" % acc)
        cv_scores.append(acc)

    acc = np.mean(cv_scores)
    std = np.std(cv_scores)
    print("Model accuracy: %.2f%% (+/- %.2f%%)" % (acc*100, std*100))

    print('test')


    