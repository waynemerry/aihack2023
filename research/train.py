import torch
import numpy as np
import pandas as pd

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models import LSTMModel

class TimeSeriesDataset(Dataset):
    def __init__(self, data_file, device = 'cpu', mode = "train"):
        self.data   = pd.read_csv(data_file)
        self.device = device
        self.mode   = mode
        
    def __len__(self):

        if self.mode == "train":
            return int(self.data.shape[0]*0.9)
        elif self.mode == "val":
            return self.data.shape[0] - int(self.data.shape[0]*0.9)
        else:
            return self.data.shape[0]
        

    def __getitem__(self, idx):
        
        if self.mode == "val":
            index = idx + int(self.data.shape[0]*0.9)
        else:
            index = idx

        features = np.array(self.data.iloc[index, 1:-2])

        x = torch.tensor(features.astype(np.float32), device = self.device)
        y = torch.tensor(np.array(self.data.iloc[index, -2:]).astype(np.float32), device = self.device)

        return x, y


if __name__ == "__main__":

    time_series_data = r'data\time_series_data.csv'
    train_data = TimeSeriesDataset(time_series_data, mode="train")
    val_data   = TimeSeriesDataset(time_series_data, mode="val")

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(training_data,   batch_size=64, shuffle=True)

    model = LSTMModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    n_epochs = 2000
    
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

    print('test')

    