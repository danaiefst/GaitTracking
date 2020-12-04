import torch
from torch.nn import Sigmoid, LSTM, Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

torch.set_default_dtype(torch.double)

class Net(Module):
    def init_hidden(self):
        self.h = (torch.zeros(self.num_of_layers, 1, self.grid * self.grid * 6).to(self.device), torch.zeros(self.num_of_layers, 1, self.grid * self.grid * 6).to(self.device))

    def __init__(self, device):
        super(Net, self).__init__()
        self.grid = 8
        self.num_of_layers = 1
        self.device = device
        self.cnn_layers = Sequential(
            Conv2d(1, 16, kernel_size=7),
            #BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 32, kernel_size=4),
            #BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 64, kernel_size=5),
            #BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, 128, kernel_size=5),
            #BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=5),
            #BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=3),
            #BatchNorm2d(128),
            ReLU(inplace=True)

        )

        self.linear_layers = Sequential(
            Linear(8192, 1024),
            Linear(1024, 384),         #384 = 6*8*8
            Sigmoid(),
        )

        self.rnn_layers = LSTM(input_size = 6 * self.grid * self.grid, hidden_size = 6 * self.grid * self.grid, num_layers = self.num_of_layers, batch_first = True)

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
        detect_loss1 = ((detect_cell1.double() - y[:, 0, :2]) ** 2).sum() + ((detect_cell2.double() - y[:, 1, :2]) ** 2).sum()
        detect_loss2 = ((y_h[torch.arange(p1.size(0)), 1:3, detect_cell1[:, 0], detect_cell1[:, 1]] - y[:, 0, 2:]) ** 2).sum() + ((y_h[torch.arange(p1.size(0)), 4:, detect_cell2[:, 0], detect_cell2[:, 1]] - y[:, 1, 2:]) ** 2).sum()
        #print(prob_loss, detect_loss1, detect_loss2)
        return 2 * prob_loss + 10 * detect_loss2 + detect_loss1

    def forward(self, x):
        x = x.to(torch.double)
        x = x.reshape(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = x.view(x.size(0), 1, -1)
        y = torch.empty(x.shape).to(self.device)
        for i, img in enumerate(x):
            img = img.view(1, 1, -1)
            img, self.h = self.rnn_layers(img, self.h)
            y[i] = img
        y = y.reshape((y.size(0), 6, self.grid, self.grid))
        return y
