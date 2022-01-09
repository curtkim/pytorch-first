# https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
import os
import logging
import numpy as np
import torch
from pytorch_lightning.lite import LightningLite
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


class Lite(LightningLite):
    def run(self, model, dataset, epochs):

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"debug_cpu_{self.global_rank}.log", 'w')
            ]
        )

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        learningRate = 0.01

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

        model, optimizer = self.setup(model, optimizer)
        dataloader = self.setup_dataloaders(dataloader)


        for epoch in range(epochs):
            for inputs, labels in dataloader:
                # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
                optimizer.zero_grad()

                logging.info(f"{os.getpid()}, {self.global_rank} {epoch}, {inputs}")
                # Converting inputs and labels to Variable
                # inputs = Variable(x)
                # labels = Variable(y)

                # get output from the model, given the inputs
                outputs = model(inputs)

                # get loss for the predicted output
                loss = criterion(outputs, labels)
                #print(epoch, loss)
                # get gradients w.r.t to parameters
                #loss.backward()
                self.backward(loss)

                # update parameters
                optimizer.step()

            print('epoch {}, loss {}'.format(epoch, loss.item()))


def predict_and_visualize(model, dataset):
    with torch.no_grad():  # we don't need gradients in the testing phase
        predicted = model(torch.from_numpy(dataset.x_train)).data.numpy()
        print(predicted)

    plt.clf()
    plt.plot(dataset.x_train, dataset.y_train, 'go', label='True data', alpha=0.5)
    plt.plot(dataset.x_train, predicted, '--', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    inputDim = 1        # takes variable 'x'
    outputDim = 1       # takes variable 'y'
    epochs = 10

    dataset = MyDataset(10)
    model = linearRegression(inputDim, outputDim)

    module = Lite()
    module.run(model=model, dataset=dataset, epochs=epochs)
    predict_and_visualize(model, dataset)
