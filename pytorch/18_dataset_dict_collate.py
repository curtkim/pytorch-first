import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(10, 3, 24, 24)
        self.target = torch.randint(0, 10, (10,))   # (low, hight, shape)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return {'data': x, 'target': y}

    def __len__(self):
        return len(self.data)


dataset = MyDataset()
assert 10 == len(dataset)

loader = DataLoader(dataset, batch_size=2, num_workers=2)

batch = next(iter(loader))
data = batch['data']
target = batch['target']
assert torch.Size([2, 3, 24, 24]) == data.shape
assert torch.Size([2]) == target.shape
