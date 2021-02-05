import torch
from torch.nn import LSTM, Linear, ReLU, Sequential, Conv1d, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout
from torch.nn.utils import weight_norm

torch.set_default_dtype(torch.double)
#torch.manual_seed(1)
#torch.cuda.manual_seed(1)
grid = 7

def find_center(out):
    probh = out[:, [0, 3], :, :]
    rposh = probh[:, 0, :, :].view(probh.shape[0], -1).argmax(axis=1)
    rlegx, rlegy = rposh // grid, rposh % grid
    rlegh = out[torch.arange(out.shape[0]), 1:3, rlegx, rlegy]
    
    lposh = probh[:, 1, :, :].view(probh.shape[0], -1).argmax(axis=1)
    llegx, llegy = lposh // grid, lposh % grid
    llegh = out[torch.arange(out.shape[0]), 4:, llegx, llegy]

    return torch.stack([rlegh[:, 0] + rlegx.double(), rlegh[:, 1] + rlegy.double(), llegh[:, 0] + llegx.double(), llegh[:, 1] + llegy.double()], dim=1) / grid

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
        x = find_center(x)
        return x

class RNN(Module):
    def init_hidden(self, device):
        self.h = (torch.rand(self.num_of_layers, 1, 300).to(device), torch.rand(self.num_of_layers, 1, 300).to(device))

    def detach_hidden(self):
        self.h = (self.h[0].detach(), self.h[1].detach())
        
    def __init__(self):
        super(RNN, self).__init__()
        self.num_of_layers = 1
        self.rnn_layers = LSTM(input_size = 4, hidden_size = 300, num_layers = self.num_of_layers, batch_first = True)

    def forward(self, x):
        x = x.view(1, x.size(0), -1)
        x, self.h = self.rnn_layers(x, self.h)
        x = x.view((x.size(1), -1))
        return x[:, :4] * grid

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
        return 5 * detect_loss# + assoc_loss
