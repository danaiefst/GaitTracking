import torch
from torch.nn import LSTM, Linear, ReLU, Sequential, Conv1d, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout, CrossEntropyLoss, Sigmoid, Softmax, LeakyReLU
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

    return torch.stack([rlegh[:, 0] + rlegx.double(), rlegh[:, 1] + rlegy.double(),  llegh[:, 0] + llegx.double(), llegh[:, 1] + llegy.double()], dim=1)

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
            Conv2d(16, 16, kernel_size=3),
            BatchNorm2d(16),
            ReLU(inplace=True),
        )

        self.linear_layers = Sequential(
            Linear(784, 512),
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
        self.h = (torch.zeros(1, 1, 6 * grid * grid).to(device), torch.zeros(1, 1, 6 * grid * grid).to(device))

    def detach_hidden(self):
        self.h = (self.h[0].detach(), self.h[1].detach())

    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = LSTM(input_size = 6 * grid * grid, hidden_size = 6 * grid * grid, batch_first = True)

    def forward(self, x):
        x = x.view(1, x.size(0), -1)
        x, self.h = self.rnn(x, self.h)
        #x = x[:, :, :int(x.shape[2] / 2)] + x[:, :, int(x.shape[2] / 2):]
        x = x.view((x.size(1), 6, grid, grid))
        return x

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
        assoc_loss = ((rlegh[1:, 0] + rlegx.double()[1:] - rlegh[:-1, 0] + rlegx.double()[:-1]) ** 2 + (rlegh[1:, 1] + rlegy.double()[1:] - rlegh[:-1, 1] + rlegy.double()[:-1]) ** 2 + (llegh[1:, 0] + llegx.double()[1:] - llegh[:-1, 0] + llegx.double()[:-1]) ** 2 + (llegh[1:, 1] + llegy.double()[1:] - llegh[:-1, 1] + llegy.double()[:-1]) ** 2).sum()
        #print(detect_loss * 5, assoc_loss/150)
        #return prob_loss + 5 * detect_loss + assoc_loss / 10
        return prob_loss + 5 * detect_loss

class GNet(Module):

    def init_hidden(self):
        self.h1 = (torch.zeros((self.bi + 1) * self.num_layers, 1, self.hidden).to(self.device), torch.zeros((self.bi + 1) * self.num_layers, 1, self.hidden).to(self.device))
        self.h2 = (torch.zeros((self.bi + 1) * self.num_layers, 1, self.hidden).to(self.device), torch.zeros((self.bi + 1) * self.num_layers, 1, self.hidden).to(self.device))
        self.h3 = (torch.zeros((self.bi + 1) * self.num_layers, 1, self.hidden).to(self.device), torch.zeros((self.bi + 1) * self.num_layers, 1, self.hidden).to(self.device))


    def detach_hidden(self):
        self.h1 = (self.h1[0].detach(), self.h1[1].detach())
        self.h2 = (self.h2[0].detach(), self.h2[1].detach())
        self.h3 = (self.h3[0].detach(), self.h3[1].detach())

        
    def __init__(self, device):
        super(GNet, self).__init__()
        self.bi = 1
        self.num_layers = 1
        self.input_size = 4
        self.hidden = 200
        self.linears = Sequential(
            Linear(self.hidden, 4),
            LeakyReLU(inplace=True))

        self.rnn1 = LSTM(input_size = self.input_size, hidden_size = self.hidden, num_layers = self.num_layers, batch_first = True, bidirectional = (self.bi == True))
        self.rnn2 = LSTM(input_size = self.input_size, hidden_size = self.hidden, num_layers = self.num_layers, batch_first = True, bidirectional = (self.bi == True))
        self.rnn3 = LSTM(input_size = self.hidden, hidden_size = self.hidden, num_layers = self.num_layers, batch_first = True)
        self.l = CrossEntropyLoss()
        self.device = device

    def forward(self, x):
        #print(x.shape)
        x = find_center(x)
        #x = self.linears(x)
        x = x.view(1, x.size(0), -1)
        x1, self.h1 = self.rnn1(x, self.h1)
        x1 = x1[:, :, :int(x1.shape[2] / 2)] + x1[:, :, int(x1.shape[2] / 2):]
        #y = Dropout(0.2)(x)
        x2, self.h2 = self.rnn2(x, self.h2)
        x2 = x2[:, :, :int(x2.shape[2] / 2)] + x2[:, :, int(x2.shape[2] / 2):]
        z = x1 + x2
        z = z.view(z.size(1), -1)
        z = self.linears(z)
        #y = y[:, :, :int(y.shape[2] / 2)] + y[:, :, int(y.shape[2] / 2):]
        #z = Dropout(0.2)(y)
        #z, self.h3 = self.rnn3(z, self.h3)
        #y, self.h2 = self.rnn2(y, self.h2)
        #x, self.h3 = self.rnn3(y + x, self.h3)
        return z

    def loss(self, yh, y):
        #print(yh.shape, y.shape)
        return self.l(yh, y)
