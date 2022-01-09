# https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, size):
        self.size = size

        # create dummy data for training
        x_values = [i for i in range(self.size)]
        x_train = np.array(x_values, dtype=np.float32)
        self.x_train = x_train.reshape(-1, 1)

        y_values = [2 * i + 1 for i in x_values]
        y_train = np.array(y_values, dtype=np.float32)
        self.y_train = y_train.reshape(-1, 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


dataset = MyDataset(10)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
# for x, y in dataloader:
#     print(x, y)


inputDim = 1        # takes variable 'x'
outputDim = 1       # takes variable 'y'
learningRate = 0.01
epochs = 10

model = linearRegression(inputDim, outputDim)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)


for epoch in range(epochs):

    for x, y in dataloader:
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        print(epoch, x)
        # Converting inputs and labels to Variable
        inputs = Variable(x)
        labels = Variable(y)

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        #print(epoch, loss)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))


with torch.no_grad(): # we don't need gradients in the testing phase
    predicted = model(Variable(torch.from_numpy(dataset.x_train))).data.numpy()
    print(predicted)


plt.clf()
plt.plot(dataset.x_train, dataset.y_train, 'go', label='True data', alpha=0.5)
plt.plot(dataset.x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
