import torch
from torch.nn import Sigmoid, LSTM, Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

torch.set_default_dtype(torch.double)

class Net(Module):
    def init_hidden(self, batch_size):
        self.h = (torch.zeros(1, batch_size, 256).to(self.device), torch.zeros(1, batch_size, 256).to(self.device))

    def __init__(self, device):
        super(Net, self).__init__()
        self.grid = 8
        self.device = device
        self.cnn_layers = Sequential(
            Conv2d(1, 16, kernel_size=13),
            #BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 32, kernel_size=5),
            #BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 32, kernel_size=5),
            #BatchNorm2d(32),
            ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=5),
            #BatchNorm2d(32),
            ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=5),
            #BatchNorm2d(32),
            ReLU(inplace=True)

        )

        self.linear_layers = Sequential(
            Linear(2048, 512),
            Linear(512, 384),         #384 = 6*8*8
            Sigmoid(),
        )

        self.rnn_layers = LSTM(input_size = 256, hidden_size = 256, batch_first = True)

    def loss(self, y_h, y):
        y = y.to(torch.double)
        y_h = y_h.to(torch.double)
        """y are the data labels (format: [[[x_cell1, y_cell1, x_center1, y_center1], [x_cell2, y_cell2, x_center2, y_center2]], ...]), y_h (y hat) is the nn's output"""
        #print(y_h[:, 0, :, :])
        p1_h = y_h[:, 0, :, :]
        p2_h = y_h[:, 3, :, :]
        p1 = torch.zeros(p1_h.shape)
        p2 = torch.zeros(p2_h.shape)
        p1[torch.arange(y.shape[0]), y[:, 0, 0].long(), y[:, 0, 1].long()] = 1
        p2[torch.arange(y.shape[0]), y[:, 1, 0].long(), y[:, 1, 1].long()] = 1
        prob_loss = ((p1.to(self.device) - p1_h) ** 2).sum() + ((p2.to(self.device) - p2_h) ** 2).sum()
        detect_cell1 = p1_h.reshape((p1.size(0), -1)).argmax(axis = 1)
        detect_cell2 = p2_h.reshape((p2.size(0), -1)).argmax(axis = 1)
        detect_cell1 = torch.stack((detect_cell1 // 8, detect_cell1 % 8), dim = 1).to(self.device)
        detect_cell2 = torch.stack((detect_cell2 // 8, detect_cell2 % 8), dim = 1).to(self.device)
        #print(y, detect_cell1, detect_cell2)
        detect_loss = ((y_h[torch.arange(p1.size(0)), 1:3, detect_cell1[:, 0], detect_cell1[:, 1]] - y[:, 0, 2:]) ** 2).sum() + ((y_h[torch.arange(p1.size(0)), 4:, detect_cell2[:, 0], detect_cell2[:, 1]] - y[:, 1, 2:]) ** 2).sum() + ((detect_cell1.double() - y[:, 0, :2]) ** 2).sum() + ((detect_cell2.double() - y[:, 1, :2]) ** 2).sum()
        #print(prob_loss / y.shape[0], detect_loss / y.shape[0])
        return prob_loss + detect_loss

    def forward(self, x):
        x = x.to(torch.double)
        self.init_hidden(x.size(0))
        x = x.reshape(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = x.reshape((x.size(0), 6, self.grid, self.grid))
        #y = x[:, [1, 2, 4, 5], :, :]
        #y = y.view(y.size(0), 1, -1)
        #y, self.h = self.rnn_layers(y, self.h)
        #y = y.reshape((x.size(0), 4, self.grid, self.grid))
        #z = torch.empty(x.shape, dtype=torch.double).to(self.device)
        #z[:, [0, 3], :, :] = x[:, [0, 3], :, :]
        #z[:, [1, 2, 4, 5], :, :] = y
        return x
