import torch


def test_float_tensor():
    torch.equal(
        torch.FloatTensor([1, 2]),
        torch.tensor([1, 2], dtype=torch.float32)
    )

    torch.equal(
        torch.DoubleTensor([1, 2]),
        torch.tensor([1, 2], dtype=torch.float64)
    )
