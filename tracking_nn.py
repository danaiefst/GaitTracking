import torch
from torch.nn import LSTM, Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout

torch.set_default_dtype(torch.double)

class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_layers = Sequential(
            Conv2d(1, 16, kernel_size=7),
            BatchNorm2d(16),
            #Dropout(0.5),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 16, kernel_size=3),
            BatchNorm2d(16),
            #Dropout(0.5),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 16, kernel_size=3),
            BatchNorm2d(16),
            #Dropout(0.5),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 16, kernel_size=3),
            BatchNorm2d(16),
            #Dropout(0.5),
            ReLU(inplace=True),
            Conv2d(16, 16, kernel_size=3),
            BatchNorm2d(16),
            Dropout(0.5),
            ReLU(inplace=True),
        )

        self.linear_layers = Sequential(
            Linear(784, 4),
            ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.to(torch.double)
        x = x.reshape(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = x.view(x.size(0), 4)
        return x

    def loss(self, yh, y):
        return ((yh[:, :2] - y[:, 0]) ** 2).sum() + ((yh[:, 2:] - y[:, 1]) ** 2).sum()

class RNN(Module):
    def init_hidden(self, batch_size, device):
        self.h = (torch.zeros(self.num_of_layers, batch_size, 4).to(device), torch.zeros(self.num_of_layers, batch_size, 4).to(device))

    def __init__(self):
        super(RNN, self).__init__()
        self.num_of_layers = 1
        self.rnn_layers = LSTM(input_size = 4, hidden_size = 4, num_layers = self.num_of_layers, batch_first = True)

    def forward(self, x):
        x = x.view(1, x.size(0), -1)
        x, self.h = self.rnn_layers(x, self.h)
        x = x.view((x.size(1), 4))
        return x

class Net(Module):

    def init_hidden(self, batch_size):
        self.rnn.init_hidden(batch_size, self.device)

    def __init__(self, device, cnn, rnn):
        super(Net, self).__init__()
        self.cnn = cnn
        self.rnn = rnn
        self.device = device

    def forward(self, x):
        return self.rnn(self.cnn(x))

    def loss(self, yh, y):
        return ((yh[:, :2] - y[:, 0]) ** 2).sum() + ((yh[:, 2:] - y[:, 1]) ** 2).sum()
