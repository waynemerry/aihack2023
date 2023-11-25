import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

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
