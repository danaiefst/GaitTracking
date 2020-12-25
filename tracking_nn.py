import torch
from torch.nn import Sigmoid, LSTM, Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

torch.set_default_dtype(torch.double)

class Net(Module):
    def init_hidden(self, batch_size):
        self.h = (torch.zeros(self.num_of_layers, batch_size, self.grid * self.grid * 6).to(self.device), torch.zeros(self.num_of_layers, batch_size, self.grid * self.grid * 6).to(self.device))

    def __init__(self, device):
        super(Net, self).__init__()
        self.grid = 7
        self.num_of_layers = 1
        self.device = device
        self.cnn_layers = Sequential(
            Conv2d(1, 16, kernel_size=7),
            BatchNorm2d(16),
            Dropout(0.5),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 32, kernel_size=3),
            BatchNorm2d(32),
            Dropout(0.5),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 32, kernel_size=3),
            BatchNorm2d(32),
            Dropout(0.5),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 32, kernel_size=3),
            BatchNorm2d(32),
            Dropout(0.5),
            ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=3),
            BatchNorm2d(32),
            Dropout(0.5),
            ReLU(inplace=True),
        )

        self.linear_layers = Sequential(
            Linear(1568, 512),
            Dropout(0.5),
            ReLU(inplace=True),
            Linear(512, 294),
            Dropout(0.5),        #384 = 6*7*7
            Sigmoid(),
        )


        self.rnn_layers = LSTM(input_size = 6 * self.grid * self.grid, hidden_size = 6 * self.grid * self.grid, num_layers = self.num_of_layers, batch_first = True)

    def loss(self, yh, y):
        #Probability loss
        prob = yh[:, [0, 3], :, :]
        probh = torch.zeros(y_h.shape[0], 2, self.grid, self.grid).to(self.device)
        probh[torch.arange(y.shape[0]), 0, y[:, 0, 0].long(), y[:, 0, 1].long()] = 1
        probh[torch.arange(y.shape[0]), 1, y[:, 1, 0].long(), y[:, 1, 1].long()] = 1
        prob_loss = ((prob - probh) ** 2).sum()

        #Detection loss
        rposh = prob[:, 0, :, :].view(prob.shape[0], -1).argmax(axis=1)
        rlegx, rlegy = rposh // self.grid, rposh % self.grid
        rlegh = yh[torch.arange(yh.shape[0]), 1:3, rlegx, rlegy]

        lposh = prob[:, 3, :, :].view(prob.shape[0], -1).argmax(axis=1)
        llegx, llegy = lposh // self.grid, lposh % self.grid
        llegh = yh[torch.arange(yh.shape[0]), 4:, llegx, llegy]

        detect_loss = ((rlegh[:, 0] + rlegx - y[:, 0, 0] - y[:, 0, 2]) ** 2 + (rlegh[:, 1] + rlegy - y[:, 0, 1] - y[:, 0, 3]) ** 2 + (llegh[:, 0] + llegx - y[:, 1, 0] - y[:, 1, 2]) ** 2 + (llegh[:, 1] + llegy - y[:, 1, 1] - y[:, 1, 3]) ** 2).sum()

    def forward(self, x):
        x = x.to(torch.double)
        x = x.reshape(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        #x = x.view(1, x.size(0), -1)
        #x, self.h = self.rnn_layers(x, self.h)
        #x = x.view((x.size(1), 6, self.grid, self.grid))
        x = x.view(x.size(0), 6, self.grid, self.grid)
        return x
