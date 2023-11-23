import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=40, hidden_size=20, num_layers=1, batch_first=True)
        self.linear = nn.Linear(20, 2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x