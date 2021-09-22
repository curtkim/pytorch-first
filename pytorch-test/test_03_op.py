import torch


def test_squeeze():
    '''removes any axes that are dimension 1 '''
    a = torch.ones((1, 6, 4, 1))
    assert torch.Size([6, 4]) == a.squeeze().shape

