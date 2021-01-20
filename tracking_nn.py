import torch
from torch.nn import LSTM, Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout

torch.set_default_dtype(torch.double)
grid = 7

class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
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
            #Conv2d(16, 16, kernel_size=3),
            #BatchNorm2d(16),
            #ReLU(inplace=True),
        )

        self.linear_layers = Sequential(
            Linear(1296, 512),
            Dropout(0.5),
            ReLU(inplace=True),
            Linear(512, 294),    #294 = 6*7*7
            ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.to(torch.double)
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = x.view(x.size(0), 6, grid, grid)
        return x

class RNN(Module):
    def init_hidden(self, device):
        self.h = (torch.zeros(self.num_of_layers, 1, 6 * grid * grid).to(device), torch.zeros(self.num_of_layers, 1, 6 * grid * grid).to(device))

    def __init__(self):
        super(RNN, self).__init__()
        self.num_of_layers = 1
        self.rnn_layers = LSTM(input_size = 6 * grid * grid, hidden_size = 6 * grid * grid, num_layers = self.num_of_layers, batch_first = True)

    def forward(self, x):
        x = x.view(1, x.size(0), -1)
        x, self.h = self.rnn_layers(x, self.h)
        x = x.view((x.size(1), 6, grid, grid))
        return x

class Net(Module):

    def init_hidden(self):
        self.rnn.init_hidden(self.device)

    def __init__(self, device, cnn, rnn):
        super(Net, self).__init__()
        self.cnn = cnn
        self.rnn = rnn
        self.device = device

    def forward(self, x):
        #return self.cnn(x)
        return self.rnn(self.cnn(x))

    def loss(self, yh, y):
        #Probability loss
        probh = yh[:, [0, 3], :, :]
        prob = torch.zeros(y.shape[0], 2, grid, grid).to(self.device)
        prob[torch.arange(y.shape[0]), 0, y[:, 0, 0].long(), y[:, 0, 1].long()] = 1
        prob[torch.arange(y.shape[0]), 1, y[:, 1, 0].long(), y[:, 1, 1].long()] = 1
        prob_loss = ((prob - probh) ** 2).sum()

        #Detection loss
        rposh = probh[:, 0, :, :].view(prob.shape[0], -1).argmax(axis=1)
        rlegx, rlegy = (rposh // grid).to(self.device), (rposh % grid).to(self.device)
        rlegh = yh[torch.arange(yh.shape[0]), 1:3, rlegx, rlegy]

        lposh = probh[:, 1, :, :].view(prob.shape[0], -1).argmax(axis=1)
        llegx, llegy = (lposh // grid).to(self.device), (lposh % grid).to(self.device)
        llegh = yh[torch.arange(yh.shape[0]), 4:, llegx, llegy]

        detect_loss = ((rlegh[:, 0] + rlegx.double() - y[:, 0, 0]) ** 2 + (rlegh[:, 1] + rlegy.double() - y[:, 0, 1]) ** 2 + (llegh[:, 0] + llegx.double() - y[:, 1, 0]) ** 2 + (llegh[:, 1] + llegy.double() - y[:, 1, 1]) ** 2).sum()

        #Association loss
        assoc_loss = (rlegh[1:, 0] + rlegx.double()[1:] - rlegh[:-1, 0] + rlegx.double()[:-1]) ** 2 + (rlegh[1:, 1] + rlegy.double()[1:] - rlegh[:-1, 1] + rlegy.double()[:-1]) ** 2 + ((llegh[1:, 0] + llegx.double()[1:] - llegh[:-1, 0] + llegx.double()[:-1]) ** 2 + (llegh[1:, 1] + llegy.double()[1:] - llegh[:-1, 1] + llegy.double()[:-1]) ** 2 

        return prob_loss + detect_loss + assoc_loss

    #def loss(self, yh, y):
    #    #Probability loss
    #    probh = yh[:, [0, 3], :, :]
    #    prob = torch.zeros(y.shape[0], 2, grid, grid).to(self.device)
    #    prob[torch.arange(y.shape[0]), 0, y[:, 0, 0].long(), y[:, 0, 1].long()] = 1
    #    prob[torch.arange(y.shape[0]), 1, y[:, 1, 0].long(), y[:, 1, 1].long()] = 1
    #    prob_loss = ((prob - probh) ** 2 * ((1 - prob) * 0.5 + prob)).sum()

        #Detection loss
    #    rlegh = yh[torch.arange(yh.shape[0]), 1:3, y[:, 0, 0].long(), y[:, 0, 1].long()]
    #    llegh = yh[torch.arange(yh.shape[0]), 4:, y[:, 1, 0].long(), y[:, 1, 1].long()]

    #    detect_loss = 5 * (((rlegh - y[:, 0] % 1) ** 2).sum() + ((llegh - y[:, 1] % 1) ** 2).sum())

    #    return 5 * prob_loss + detect_loss
