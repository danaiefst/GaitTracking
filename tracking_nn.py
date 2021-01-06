import torch
from torch.nn import Sigmoid, LSTM, Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

torch.set_default_dtype(torch.double)

class CNN(Module):
    def __init__(self, device):
        super(CNN, self).__init__()
        self.grid = 7
        self.device = device
        self.cnn_layers = Sequential(
            Conv2d(1, 16, kernel_size=7, stride=2),
            BatchNorm2d(16),
            ReLU(inplace=True),
            Conv2d(16, 16, kernel_size=3, padding = 2),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 16, kernel_size=3, padding = 1),
            BatchNorm2d(16),
            ReLU(inplace=True),
            Conv2d(16, 16, kernel_size=3),
            BatchNorm2d(16),
            ReLU(inplace=True),
            Conv2d(16, 16, kernel_size=3),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 16, kernel_size=3),
            BatchNorm2d(16),
            ReLU(inplace=True),
            Conv2d(16, 16, kernel_size=3),
            BatchNorm2d(16),
            ReLU(inplace=True)
        )

        self.linear_layers = Sequential(
            Linear(784, 512),
            Dropout(0.5),
            ReLU(inplace=True),
            Linear(512, 294),    #384 = 6*7*7
            ReLU(inplace=True)
        )

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

        detect_loss = ((rlegh - y[:, 0, 2:]) ** 2).sum() + ((llegh - y[:, 1, 2:]) ** 2).sum()

        return 5 * prob_loss + 5 * detect_loss

    def forward(self, x):
        x = x.to(torch.double)
        x = x.reshape(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = x.view(x.size(0), 6, self.grid, self.grid)
        return x

class RNN(Module):
    def init_hidden(self, batch_size):
        self.h = (torch.zeros(self.num_of_layers, batch_size, self.grid * self.grid * 6).to(self.device), torch.zeros(self.num_of_layers, batch_size, self.grid * self.grid * 6).to(self.device))

    def __init__(self, device):
        super(RNN, self).__init__()
        self.grid = 7
        self.num_of_layers = 1
        self.device = device
        self.rnn_layers = LSTM(input_size = 6 * self.grid * self.grid, hidden_size = 6 * self.grid * self.grid, num_layers = self.num_of_layers, batch_first = True)

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

        detect_loss = ((rlegh - y[:, 0, 2:]) ** 2).sum() + ((llegh - y[:, 1, 2:]) ** 2).sum()

        return prob_loss + 5 * detect_loss

    def forward(self, x):
        x = x.view(1, x.size(0), -1)
        x, self.h = self.rnn_layers(x, self.h)
        x = x.view((x.size(1), 6, self.grid, self.grid))
        return x

class Net(Module):
    def init_hidden(self, batch_size):
        self.rnn_model.h = (torch.zeros(self.num_of_layers, batch_size, self.grid * self.grid * 6).to(self.device), torch.zeros(self.num_of_layers, batch_size, self.grid * self.grid * 6).to(self.device))

    def __init__(self, device, cnn_model, rnn_model):
        super(Net, self).__init__()
        self.grid = 7
        self.device = device
        self.cnn_model = cnn_model
        self.rnn_model = rnn_model

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

        detect_loss = ((rlegh - y[:, 0, 2:]) ** 2).sum() + ((llegh - y[:, 1, 2:]) ** 2).sum()

        return prob_loss + 5 * detect_loss

    def forward(self, x):
        x = x.to(torch.double)
        x = x.reshape(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn_model.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.cnn_model.linear_layers(x)
        x = self.rnn_model(x)
        return x
