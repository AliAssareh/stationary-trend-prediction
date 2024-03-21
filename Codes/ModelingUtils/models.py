import torch.nn.functional as F
from torch import nn
import torch


class ConvNet221(nn.Module):
    def __init__(self):
        super(ConvNet221, self).__init__()
        self.conv0_1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.conv0_2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0), groups=8)
        self.max0 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=(0, 0))
        self.batch0 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 42, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.max1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.batch1 = nn.BatchNorm2d(42)
        self.conv2 = nn.Conv2d(42, 1, kernel_size=(3, 1), stride=(3, 1), padding=(0, 0))
        self.batch2 = nn.BatchNorm2d(1)

    @staticmethod
    def get_output_dim(input_dim1, input_dim2):
        return int((int((input_dim1 / 3)) / 3)) * int((((input_dim2 - 4) / 3) / 3))

    def forward(self, x):
        x = F.silu(self.conv0_1(x))
        x = F.silu(self.conv0_2(x))
        x = self.max0(x)
        x = self.batch0(x)
        x = F.silu(self.conv1_2(x))
        x = self.max1(x)
        x = self.batch1(x)
        x = F.silu(self.conv2(x))
        x = self.batch2(x)
        x = x.view(x.size(0), x.size(2) * x.size(3))
        return x


class WideNet0_1(nn.Module):
    def __init__(self, input_dim):
        super(WideNet0_1, self).__init__()
        self.layer1 = nn.Linear(input_dim, 8)

    def forward(self, x):
        x = self.layer1(x)
        return x


class ConvFC(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, device):
        super(ConvFC, self).__init__()
        self.conv_net = ConvNet221()
        input_dim = self.conv_net.get_output_dim(input_dim_1, input_dim_2)
        self.wide = WideNet0_1(input_dim)
        self.batch1 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(8, 4)
        self.batch2 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.conv_net(x)
        x = self.wide(x)
        x = F.silu(x)
        x = self.batch1(x)
        x = F.silu(self.fc1(x))
        x = self.batch2(x)
        x = self.fc2(x)
        return x


class FCN0(nn.Module):
    def __init__(self, input_dim1, input_dim2):
        super(FCN0, self).__init__()
        n = 81
        self.input_dim = self.get_output_dim(input_dim1, input_dim2)
        self.layer1 = nn.Linear(self.input_dim, n)
        self.batch1 = nn.BatchNorm1d(n)
        self.layer2 = nn.Linear(n, 15)
        self.batch2 = nn.BatchNorm1d(15)
        self.layer3 = nn.Linear(15, 8)

    @staticmethod
    def get_output_dim(input_dim1, input_dim2):
        return int(input_dim1) * int(input_dim2)

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = F.silu(self.layer1(x))
        x = self.batch1(x)
        x = F.silu(self.layer2(x))
        x = self.batch2(x)
        x = self.layer3(x)
        return x


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, device):
        super(FullyConnectedNetwork, self).__init__()
        self.fcn0 = FCN0(input_dim_1, input_dim_2)
        self.batch1 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(8, 4)
        self.batch2 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fcn0(x)
        x = F.silu(x)
        x = self.batch1(x)
        x = F.silu(self.fc1(x))
        x = self.batch2(x)
        x = self.fc2(x)
        return x


class LSTM0(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(LSTM0, self).__init__()
        self.device = device
        self.number_of_cells = 81
        self.lstm = nn.LSTM(input_dim, self.number_of_cells, batch_first=True)
        self.fc = nn.Linear(self.number_of_cells, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), x.size(2), x.size(3))
        h0 = torch.zeros(1, x.size(0), self.number_of_cells, dtype=torch.double).requires_grad_().to(self.device)
        c0 = torch.zeros(1, x.size(0), self.number_of_cells, dtype=torch.double).requires_grad_().to(self.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class FullyLSTMNetwork(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, device):
        super(FullyLSTMNetwork, self).__init__()
        self.lstm = LSTM0(int(input_dim_2), 8, device)
        self.batch1 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(8, 4)
        self.batch2 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.lstm(x)
        x = F.silu(x)
        x = self.batch1(x)
        x = F.silu(self.fc1(x))
        x = self.batch2(x)
        x = self.fc2(x)
        return x


class GRU0(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(GRU0, self).__init__()
        self.device = device
        self.number_of_cells = 81
        self.gru = nn.GRU(input_dim, self.number_of_cells, batch_first=True)
        self.fc = nn.Linear(self.number_of_cells, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), x.size(2), x.size(3))
        h0 = torch.zeros(1, x.size(0), self.number_of_cells, dtype=torch.double).requires_grad_().to(self.device)

        out, hn = self.gru(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out


class FullyGRUNetwork(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, device):
        super(FullyGRUNetwork, self).__init__()
        self.gru = GRU0(int(input_dim_2), 8, device)
        self.batch1 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(8, 4)
        self.batch2 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.gru(x)
        x = F.silu(x)
        x = self.batch1(x)
        x = F.silu(self.fc1(x))
        x = self.batch2(x)
        x = self.fc2(x)
        return x


class RNN0(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(RNN0, self).__init__()
        self.device = device
        self.number_of_cells = 81
        self.gru = nn.RNN(input_dim, self.number_of_cells, batch_first=True)
        self.fc = nn.Linear(self.number_of_cells, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), x.size(2), x.size(3))
        h0 = torch.zeros(1, x.size(0), self.number_of_cells, dtype=torch.double).requires_grad_().to(self.device)

        out, hn = self.gru(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out


class FullyRNNNetwork(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, device):
        super(FullyRNNNetwork, self).__init__()
        self.rnn = RNN0(int(input_dim_2), 8, device)
        self.batch1 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(8, 4)
        self.batch2 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.rnn(x)
        x = F.silu(x)
        x = self.batch1(x)
        x = F.silu(self.fc1(x))
        x = self.batch2(x)
        x = self.fc2(x)
        return x


class ConvNet220(nn.Module):
    def __init__(self):
        super(ConvNet220, self).__init__()
        self.conv0_1 = nn.Conv2d(1, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0))
        self.conv0_2 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0), groups=8)
        self.max0 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.batch0 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 42, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.max1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.batch1 = nn.BatchNorm2d(42)
        self.conv2 = nn.Conv2d(42, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.batch2 = nn.BatchNorm2d(1)

    @staticmethod
    def get_output_dim(input_dim):
        return int((((input_dim - 4) / 3) / 3))

    def forward(self, x):
        x = F.silu(self.conv0_1(x))
        x = F.silu(self.conv0_2(x))
        x = self.max0(x)
        x = self.batch0(x)
        x = F.silu(self.conv1_2(x))
        x = self.max1(x)
        x = self.batch1(x)
        x = F.silu(self.conv2(x))
        x = self.batch2(x)
        x = x.view(x.size(0), x.size(2), x.size(3))
        return x


class CLSTMNet0(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(CLSTMNet0, self).__init__()
        self.device = device
        self.number_of_cells = input_dim
        self.lstm = nn.LSTM(input_dim, self.number_of_cells, batch_first=True)
        self.fc = nn.Linear(self.number_of_cells, output_dim)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.number_of_cells, dtype=torch.double).requires_grad_().to(self.device)
        c0 = torch.zeros(1, x.size(0), self.number_of_cells, dtype=torch.double).requires_grad_().to(self.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class ConvLSTMNet(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, device):
        super(ConvLSTMNet, self).__init__()
        self.conv_net0 = ConvNet220()
        input_dim = self.conv_net0.get_output_dim(input_dim_2)
        self.wide_0 = CLSTMNet0(input_dim, 2, device)
        self.batch1 = nn.BatchNorm1d(2)
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.conv_net0(x)
        x = self.wide_0(x)
        x = self.batch1(x)
        x = self.fc1(x)
        return x



