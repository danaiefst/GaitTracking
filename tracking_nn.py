import torch
from torch.nn import LSTM, Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout

torch.set_default_dtype(torch.double)

class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.device = device
        self.cnn_layers = Sequential(
            Conv2d(1, 16, kernel_size=7),
            BatchNorm2d(16),
            #Dropout(0.5),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 32, kernel_size=3),
            BatchNorm2d(32),
            #Dropout(0.5),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 32, kernel_size=3),
            BatchNorm2d(32),
            #Dropout(0.5),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 32, kernel_size=3),
            BatchNorm2d(32),
            #Dropout(0.5),
            ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=3),
            BatchNorm2d(32),
            Dropout(0.5),
            ReLU(inplace=True),
        )

        self.linear_layers = Sequential(
            Linear(1568, 294),
            ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.to(torch.double)
        x = x.reshape(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

class RNN(Module):
    def init_hidden(self, batch_size):
        self.h = (torch.zeros(self.num_of_layers, batch_size, self.grid * self.grid * 6).to(self.device), torch.zeros(self.num_of_layers, batch_size, self.grid * self.grid * 6).to(self.device))

    def __init__(self):
        super(RNN, self).__init__()
        self.grid = 7
        self.num_of_layers = 1
        self.rnn_layers = LSTM(input_size = 6 * self.grid * self.grid, hidden_size = 6 * self.grid * self.grid, num_layers = self.num_of_layers, batch_first = True)

    def forward(self, x):
        x = x.view(1, x.size(0), -1)
        x, self.h = self.rnn_layers(x, self.h)
        x = x.view((x.size(1), 6, self.grid, self.grid))
        return x

class Net(Module):

    def init_hidden(self, batch_size):
        self.rnn.init_hidden(batch_size)

    def __init__(self, device, cnn, rnn):
        super(Net, self).__init__()
        self.grid = 7
        self.cnn = cnn
        self.rnn = rnn

    def loss(self, yh, y):
        #Probability loss
        probh = yh[:, [0, 3], :, :]
        prob = torch.zeros(y.shape[0], 2, self.grid, self.grid).to(self.device)
        prob[torch.arange(y.shape[0]), 0, y[:, 0, 0].long(), y[:, 0, 1].long()] = 1
        prob[torch.arange(y.shape[0]), 1, y[:, 1, 0].long(), y[:, 1, 1].long()] = 1
        prob_loss = ((prob - probh) ** 2 * ((1 - prob) * 0.5 + prob)).sum()

        #Detection loss
        rlegh = yh[torch.arange(yh.shape[0]), 1:3, y[:, 0, 0].long(), y[:, 0, 1].long()]
        llegh = yh[torch.arange(yh.shape[0]), 4:, y[:, 1, 0].long(), y[:, 1, 1].long()]

        detect_loss = 5 * (((rlegh - y[:, 0, 2:]) ** 2).sum() + ((llegh - y[:, 1, 2:]) ** 2).sum())

        return prob_loss + detect_loss

    def forward(self, x):
        return self.rnn(self.cnn(x))
