import torch
from torch import nn


def test_batchnorm():
    m = nn.BatchNorm1d(5, affine=False)
    input = torch.randn(3, 5)
    output = m(input)
    print(input)
    print(output)

    print(output.sum(axis=0))
    print(output.sum(axis=1))
