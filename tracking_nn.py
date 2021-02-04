import torch
from torch.nn import LSTM, Linear, ReLU, Sequential, Conv1d, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout
from torch.nn.utils import weight_norm

torch.set_default_dtype(torch.double)
#torch.manual_seed(1)
#torch.cuda.manual_seed(1)
grid = 7

def find_center(out):
    p1_h = out[:, 0, :, :]
    p2_h = out[:, 3, :, :]
    detect_cell1 = p1_h.reshape(out.shape[0], -1).argmax(axis = 1)
    detect_cell2 = p2_h.reshape(out.shape[0], -1).argmax(axis = 1)
    x1, y1 = detect_cell1 // grid, detect_cell1 % grid
    x2, y2 = detect_cell2 // grid, detect_cell2 % grid
    return torch.stack([x1 + out[torch.arange(p1_h.shape[0]), 1, x1, y1], y1 + out[torch.arange(p1_h.shape[0]), 2, x1, y1], x2 + out[torch.arange(p1_h.shape[0]), 1, x2, y2], y2 + out[torch.arange(p1_h.shape[0]), 2, x2, y2]], dim=1).double() / grid

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
        y = find_center(x)
        return y

class RNN(Module):
    def init_hidden(self, device):
        self.h = (torch.zeros(self.num_of_layers, 1, 4).to(device), torch.zeros(self.num_of_layers, 1, 4).to(device))

    def detach_hidden(self):
        self.h = (self.h[0].detach(), self.h[1].detach())
        
    def __init__(self):
        super(RNN, self).__init__()
        self.num_of_layers = 1
        self.rnn_layers = LSTM(input_size = 4, hidden_size = 4, num_layers = self.num_of_layers, batch_first = True)

    def forward(self, x):
        x = x.view(1, x.size(0), -1)
        x, self.h = self.rnn_layers(x, self.h)
        x = x.view((x.size(1), 4))
        return x * grid

class Net(Module):

    def init_hidden(self):
        self.rnn.init_hidden(self.device)

    def detach_hidden(self):
        self.rnn.detach_hidden()
        
    def __init__(self, device, cnn, rnn):
        super(Net, self).__init__()
        self.cnn = cnn
        self.rnn = rnn
        self.device = device

    def forward(self, x):
        return self.rnn(self.cnn(x))

    def loss(self, yh, y):
        
        detect_loss = ((yh - y.view(y.size(0), -1)) ** 2).sum()

        assoc_loss = ((yh[1:] - yh[:-1]) ** 2).sum()
        return detect_loss# + assoc_loss / 10
