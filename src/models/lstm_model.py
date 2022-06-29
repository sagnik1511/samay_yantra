import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, device):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.sigmoid(self.fc(h_out))
        return out


def test():
    b, s, c = 4, 12, 10
    rand_value = torch.rand(b, s, c)

    model = LSTM(num_classes=c, input_size=c, hidden_size=2, num_layers=1)
    op = model(rand_value)
    print(op.shape)


if __name__ == "__main__":
    test()
