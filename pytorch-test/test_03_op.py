import torch


def test_squeeze():
    '''removes any axes that are dimension 1 '''
    a = torch.ones((1, 6, 4, 1))
    assert torch.Size([6, 4]) == a.squeeze().shape


def test_trapz():
    Y = torch.tensor([1, 4, 9])
    assert (1 + 3 / 2) + (4 + 5 / 2) == torch.trapz(Y)
