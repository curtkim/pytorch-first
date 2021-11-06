from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


if __name__ == '__main__':
    net = Net()

    for param in net.parameters():
        print(type(param), param.size())

    # <class 'torch.nn.parameter.Parameter'> torch.Size([10, 1, 5, 5])       conv1
    # <class 'torch.nn.parameter.Parameter'> torch.Size([10])
    # <class 'torch.nn.parameter.Parameter'> torch.Size([20, 10, 5, 5])      conv2
    # <class 'torch.nn.parameter.Parameter'> torch.Size([20])
    # <class 'torch.nn.parameter.Parameter'> torch.Size([50, 320])           fc1
    # <class 'torch.nn.parameter.Parameter'> torch.Size([50])
    # <class 'torch.nn.parameter.Parameter'> torch.Size([10, 50])            fc2
    # <class 'torch.nn.parameter.Parameter'> torch.Size([10])
