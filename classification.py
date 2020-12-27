import torch
from torch.nn import Sigmoid, LSTM, Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

torch.set_default_dtype(torch.double)

class Net(Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.grid = 7
        self.num_of_layers = 1
        self.device = device
        self.cnn_layers = Sequential(
            Conv2d(1, 16, kernel_size=7, stride=2),
            Dropout(0.5),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size = 2, stride = 2),
            Conv2d(16, 16, kernel_size=3),
            Dropout(0.5),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.linear_layers = Sequential(
            Linear(2304, 98),         #98 = 2*7*7
            Dropout(0.5),
            Sigmoid()
        )

    def loss(self, yh, y):
        #Probability loss
        probh = yh
        prob = torch.zeros(y.shape[0], 2, self.grid, self.grid).to(self.device)
        prob[torch.arange(y.shape[0]), 0, y[:, 0, 0].long(), y[:, 0, 1].long()] = 1
        prob[torch.arange(y.shape[0]), 1, y[:, 1, 0].long(), y[:, 1, 1].long()] = 1
        prob_loss = ((prob - probh) ** 2 * ((1 - prob) * 0.5 + prob)).sum()

        return prob_loss + detect_loss

    def forward(self, x):
        x = x.to(torch.double)
        x = x.reshape(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        #x = x.view(1, x.size(0), -1)
        #x, self.h = self.rnn_layers(x, self.h)
        #x = x.view((x.size(1), 6, self.grid, self.grid))
        x = x.view(x.size(0), 2, self.grid, self.grid)
        return x
