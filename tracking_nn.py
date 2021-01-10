import torch
from torch.nn import Conv1d, LSTM, Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.nn.utils import weight_norm

torch.set_default_dtype(torch.double)

class Chomp1d(Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = ReLU()
        self.dropout1 = Dropout(dropout)

        self.conv2 = weight_norm(Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = ReLU()
        self.dropout2 = Dropout(dropout)

        self.net = Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CNN(Module):
    def __init__(self, device):
        super(CNN, self).__init__()
        self.grid = 7
        self.device = device
        self.cnn_layers = Sequential(
            Conv2d(1, 16, kernel_size=7, stride=2),
            BatchNorm2d(16),
            ReLU(inplace=True),
            #Dropout(0.5),
            Conv2d(16, 16, kernel_size=3, padding = 2),
            BatchNorm2d(16),
            ReLU(inplace=True),
            #Dropout(0.5),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 16, kernel_size=3, padding = 1),
            BatchNorm2d(16),
            ReLU(inplace=True),
            #Dropout(0.5),
            Conv2d(16, 16, kernel_size=3),
            BatchNorm2d(16),
            ReLU(inplace=True),
            #Dropout(0.5),
            Conv2d(16, 16, kernel_size=3),
            BatchNorm2d(16),
            ReLU(inplace=True),
            #Dropout(0.5),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 16, kernel_size=3),
            BatchNorm2d(16),
            ReLU(inplace=True),
            Dropout(0.5),
            #Conv2d(16, 16, kernel_size=3),
            #BatchNorm2d(16),
            #ReLU(inplace=True)
        )

        self.linear_layers = Sequential(
            Linear(1296, 4),
            ReLU(inplace=True),
            #Dropout(0.5),
            #Linear(128, 4),    #294 = 6*7*7
            #ReLU(inplace=True)
        )

        self.TCN = TemporalConvNet(6 * self.grid * self.grid, [6 * self.grid * self.grid])

    def loss(self, yh, y):
        """#Probability loss
        probh = yh[:, [0, 3], :, :]
        prob = torch.zeros(y.shape[0], 2, self.grid, self.grid).to(self.device)
        prob[torch.arange(y.shape[0]), 0, y[:, 0, 0].long(), y[:, 0, 1].long()] = 1
        prob[torch.arange(y.shape[0]), 1, y[:, 1, 0].long(), y[:, 1, 1].long()] = 1
        prob_loss = ((prob - probh) ** 2 * ((1 - prob) * 0.5 + prob)).sum()

        #Detection loss
        rlegh = yh[torch.arange(yh.shape[0]), 1:3, y[:, 0, 0].long(), y[:, 0, 1].long()]
        llegh = yh[torch.arange(yh.shape[0]), 4:, y[:, 1, 0].long(), y[:, 1, 1].long()]

        detect_loss = ((rlegh - y[:, 0, 2:]) ** 2).sum() + ((llegh - y[:, 1, 2:]) ** 2).sum()

        return 5 * prob_loss + 5 * detect_loss"""
        return ((yh[:, :2] - y[:, 0]) ** 2).sum() + ((yh[:, 2:] - y[:, 1]) ** 2).sum()

    def forward(self, x):
        x = x.to(torch.double)
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

class RNN(Module):
    def init_hidden(self, batch_size):
        self.h = (torch.zeros(self.num_of_layers, batch_size, self.grid * self.grid * 6).to(self.device), torch.zeros(self.num_of_layers, batch_size, self.grid * self.grid * 6).to(self.device))

    def __init__(self, device):
        super(RNN, self).__init__()
        self.num_of_layers = 1
        self.device = device
        self.rnn_layers = LSTM(input_size = 4, hidden_size = 4, num_layers = self.num_of_layers, batch_first = True)

    def forward(self, x):
        x = x.view(1, x.size(0), -1)
        x, self.h = self.rnn_layers(x, self.h)
        x = x.view((x.size(1), 4))
        return x

class Net(Module):
    def init_hidden(self, batch_size):
        self.rnn_model.init_hidden(batch_size)

    def __init__(self, device, cnn_model, rnn_model):
        super(Net, self).__init__()
        self.device = device
        self.cnn_model = cnn_model
        self.rnn_model = rnn_model

    def loss(self, yh, y):
        return ((yh[:, :2] - y[:, 0]) ** 2).sum() + ((yh[:, 2:] - y[:, 1]) ** 2).sum()


    def forward(self, x):
        x = self.cnn_model(x)
        x = self.rnn_model(x)
        return x
