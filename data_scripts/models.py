import torch.nn as nn
input = 5

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=40, hidden_size=20, num_layers=1, batch_first=True)
        self.linear = nn.Linear(20, 2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(40, 2, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


class Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(input, 10)
        self.relu = nn.ReLU()
        self.output = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(input, 20)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(20, 20)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(20, 20)
        self.act3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.output = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
    
class Deeper(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(input, 20)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(20, 20)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(20, 20)
        self.act3 = nn.ReLU()
        self.layer4 = nn.Linear(20, 20)
        self.act4 = nn.ReLU()
        self.layer5 = nn.Linear(20, 20)
        self.act5 = nn.ReLU()
        self.output = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act4(self.layer4(x))
        x = self.act5(self.layer5(x))
        x = self.sigmoid(self.output(x))
        return x
    
class DeeperDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(input, 20)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(20, 20)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(20, 20)
        self.act3 = nn.ReLU()
        self.layer4 = nn.Linear(20, 20)
        self.act4 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.layer5 = nn.Linear(20, 20)
        self.act5 = nn.ReLU()
        self.output = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act4(self.layer4(x))
        x = self.dropout2(x)
        x = self.act5(self.layer5(x))
        x = self.sigmoid(self.output(x))
        return x