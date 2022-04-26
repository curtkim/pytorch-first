import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def test_conv1d():
    m = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=2)
    input = Variable(torch.randn(10, 128, 5))  # 10: batch_size, 128: embedding_dim, 5 = seq_len
    feature_maps = m(input)  # feature_maps size = [10, 32, 4=5-2+1] (bs, out_channels, out_dim)
    assert torch.Size([10, 32, 5 - 2 + 1]) == feature_maps.shape


def test_conv2d():
    # With square kernels and equal stride
    m = nn.Conv2d(in_channels=16, out_channels=33, kernel_size=3, stride=2)

    input = torch.randn(20, 16, 50, 100)
    output = m(input)
    print(output.shape)
    assert torch.Size([20, 33, int((50 - 3 + 1) / 2), int((100 - 3 + 1) / 2)]) == output.shape
    # (W - F + 2P) / S + 1
    # W: input_volume_size
    # F: kernel_size
    # P: padding_size
    # S: strides


def test_conv2d_non_square_kernel():
    m2 = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    input = torch.randn(20, 16, 50, 100)
    output = m2(input)
    assert torch.Size([20, 33, int((50 - 3 + 8)/2)+1, int((100 - 5 + 2*2))+1]) == output.shape


def test_conv2d_functional():
    # input: (batch_size, in_channels, height, width)
    # weight: (out_channels, in_channels, kernel_height, kernel_width)

    input = torch.reshape(torch.arange(0, 25, dtype=torch.float32), (1, 1, 5, 5))
    filter = torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9
    # print(input)
    # print(F.conv2d(input, filter, padding=1))
    torch.testing.assert_allclose(
        F.conv2d(input, filter, padding=0),
        torch.tensor([[[
            [6, 7, 8],
            [11, 12, 13],
            [16, 17, 18],
        ]]], dtype=torch.float32)
    )
