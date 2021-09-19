import torch
from torch import tensor


def test_indexing():
    points = tensor([
        [4.0, 1.0],
        [5.0, 3.0],
        [2.0, 1.0]
    ])

    tensor([
        [5.0, 3.0],
        [2.0, 1.0]
    ]).equal(points[1:])

    tensor([
        [5.0],
        [2.0]
    ]).equal(points[1:, 0])

    assert torch.Size([1, 3, 2]) == points[None].shape


def test_named_tensor():
    #                  -3 -2 -1
    img_t = torch.randn(3, 5, 5)  # shape [channels, rows, columns]
    assert torch.Size([5, 5]) == img_t.mean(-3).shape
    assert torch.Size([5, 5]) == img_t.mean(axis=-3).shape

    batch_t = torch.randn(2, 3, 5, 5) # shape [batch, channels, rows, columns]
    assert torch.Size([2, 5, 5]) == batch_t.mean(-3).shape
