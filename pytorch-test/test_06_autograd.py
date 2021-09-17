import torch


def test_1():
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    z = y * y * 3
    out = z.mean()
    torch.tensor(27.0).equal(out)

    out.backward() #  is equivalent to out.backward(torch.tensor(1.))
    assert torch.tensor([
        [4.5, 4.5],
        [4.5, 4.5],
    ]).equal(x.grad)


def test_2():
    x = torch.arange(4, requires_grad=True, dtype=float)
    print(x)
    y = 2 * torch.dot(x, x)
    # y' = 4x

    assert torch.equal(y, torch.tensor(2*(0+1+4+9), dtype=float))

    y.backward()
    assert torch.equal(x.grad, torch.tensor([0, 4, 8, 12], dtype=float))
