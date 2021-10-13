import torch


def test_no_grad():
    x = torch.tensor([1.0], requires_grad=True)

    with torch.no_grad():
        y = x * 2

    assert not y.requires_grad


    @torch.no_grad()
    def doubler(x):
        return x * 2

    z = doubler(x)
    assert not z.requires_grad
