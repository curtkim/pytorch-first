# https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
import os
import logging
import numpy as np
import torch
from pytorch_lightning.lite import LightningLite
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
    def run(self, model, dataset, epochs:int, batch_size:int, learningRate:float):

        # self.global_rank를 사용한다.

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"debug_dp_{self.global_rank}.log", 'w')
            ]
        )

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)


        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

        model, optimizer = self.setup(model, optimizer)

        #replace_sampler=False를 적용하면 전체 dataset을 처리하고
        # 없으면 gpu개수 만큼 partition해서 처리한다.
        dataloader = self.setup_dataloaders(dataloader)

        for epoch in range(epochs):
            for inputs, labels in dataloader:
                logging.info(f"{os.getpid()} {epoch}, {inputs.transpose(0, 1)} {labels.transpose(0,1)}")

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                self.backward(loss)
                optimizer.step()

            print('epoch {}, loss {}'.format(epoch, loss.item()))


def predict_and_visualize(model, dataset):
    with torch.no_grad():  # we don't need gradients in the testing phase
        predicted = model(torch.from_numpy(dataset.x_train)).data.numpy()
        #print(predicted)

    plt.clf()
    #plt.plot(dataset.x_train, dataset.y_train, 'go', label='True data', alpha=0.5)
    plt.plot(dataset.x_train, predicted, '--', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    print(os.getpid())
    inputDim = 1        # takes variable 'x'
    outputDim = 1       # takes variable 'y'
    epochs = 20
    batch_size = 10
    learningRate = 0.00005

    dataset = MyDataset(100)
    model = linearRegression(inputDim, outputDim)

    #ddp 2개의 process가 뜨고. data가 섞여서 출력된다. replace_sampler=False를 주지 않는 이상.
    #dp는 1개의 process가 뜬다. data가 순서대로 출력된다.
    module = Lite(strategy="dp", devices=2, accelerator="gpu")
    module.run(model=model, dataset=dataset, epochs=epochs, batch_size=batch_size, learningRate=learningRate)

    model_cpu = model.cpu()
    predict_and_visualize(model_cpu, dataset)
