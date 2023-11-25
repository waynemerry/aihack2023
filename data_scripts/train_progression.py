import copy
import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_curve
from models import LSTMModel, LinearModel, Simple, Deep, Deeper
from sklearn.model_selection import StratifiedKFold, train_test_split

def model_train(model, X_train, y_train, X_val, y_val):
        # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.01)
 
    n_epochs = 1000   # number of epochs to run
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

    X = time_series_data.iloc[:, 6:11]
    y = time_series_data.iloc[:, -2:-1]

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True)

    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    # Training
    cv_scores = []
    for train, test in kfold.split(X_train, y_train):
    # create model, train, and get accuracy
        model = Deep()
        acc = model_train(model, X[train], y[train], X[test], y[test])
        print("Accuracy: %.2f" % acc)
        cv_scores.append(acc)

    acc = np.mean(cv_scores)
    std = np.std(cv_scores)
    print("Model accuracy: %.2f%% (+/- %.2f%%)" % (acc*100, std*100))

    # Model refining

    acc = model_train(model, X_train, y_train, X_test, y_test)
    print(f"Final model accuracy: {acc*100:.2f}%")

    model.eval()
    threshold = 0.5
    with torch.no_grad():
        # Test out inference with 5 samples
        for i in range(X_test.shape[0]):
            y_pred = model(X_test[i:i+1])
            print(f"{X_test[i].numpy()} -> {int(y_pred[0].numpy() > threshold)} (expected {y_test[i].numpy()})")
    
        # Plot the ROC curve
        y_pred = model(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr) # ROC curve = TPR vs FPR
        plt.title("Receiver Operating Characteristics")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show(block=False)

    torch.save(model.state_dict(), r'models\progression_model.pt')

    