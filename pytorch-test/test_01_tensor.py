import torch
from torch import tensor


def test_scatter_add_():
    a = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    b = torch.ones(2, 4).scatter_add_(0, torch.tensor([
        [0, 1, 1, 0]
    ]), a)

    assert b.equal(torch.Tensor([
        [1, 1, 1, 4],
        [1, 2, 3, 1],
    ]).type(torch.float32))


def test_matrix_multiply():
    a = torch.ones((2, 2))
    b = tensor([
        [1, 2],
        [3, 4]
    ], dtype=torch.float32)

    assert torch.Tensor([
        [4, 6],
        [4, 6],
    ]).type(torch.float32).equal(a.mm(b))


def test_index_select():
    a = torch.arange(12).reshape(3, 4)
    indices = torch.tensor([0, 2])
    assert torch.Tensor([
        [0, 1, 2, 3],
        [8, 9, 10, 11]
    ]).type(torch.int64).equal(torch.index_select(a, 0, indices))


def test_cat():
    a = torch.arange(6).reshape(2, 3)
    torch.testing.assert_close(torch.tensor([
        [0, 1, 2],
        [3, 4, 5],
        [0, 1, 2],
        [3, 4, 5],
    ]), torch.cat((a, a), dim=0), rtol=1e-5, atol=1e-8)


def test_meshgrid():
    x = torch.tensor([1, 2])
    y = torch.tensor([3, 4])
    grid_x, grid_y = torch.meshgrid(x, y)

    assert torch.tensor([
        [1, 1],
        [2, 2]
    ]).equal(grid_x)
    assert torch.tensor([
        [3, 4],
        [3, 4]
    ]).equal(grid_y)


def test_arange():
    assert torch.tensor([0, 1, 2]).equal(torch.arange(3))


def test_sum_axis():
    r = torch.ones((4, 5))
    g = torch.ones((4, 5)) * 2
    b = torch.ones((4, 5)) * 3

    image = torch.stack((r, g, b))
    assert torch.Size([3, 4, 5]) == image.shape

    # r g b별로 합을 구한다.
    pixel_sum = image.sum(axis=[1, 2])
    assert torch.tensor([20, 40, 60], dtype=torch.float32).equal(pixel_sum)


def test_transpose():
    a = torch.ones(3, 2)
    a_t = torch.transpose(a, 0, 1)
    a_t2 = a.transpose(0, 1)
    assert torch.Size([2, 3]) == a_t.shape
    assert torch.Size([2, 3]) == a_t2.shape


def test_zero_():
    a = torch.ones(3, 2)
    a.zero_()
    tensor([
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])


def test_view():
    x = torch.range(1, 16).reshape(4, 4)
    y = x.view(16)
    assert torch.Size([16]) == y.shape

    z = x.view(-1, 8)
    assert torch.Size([2, 8]) == z.shape


def test_view2():
    x = torch.range(1, 6).reshape(2, 3)
    y = x.view(3, 2)
    assert tensor([
        [1, 2],
        [3, 4],
        [5, 6],
    ], dtype=torch.float32).equal(y)


def test_index_fill():
    x = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], dtype=torch.int)

    index = torch.tensor([0, 2], dtype=torch.long)

    assert torch.tensor([
        [-1, -1, -1],
        [4, 5, 6],
        [-1, -1, -1]
    ], dtype=torch.int).equal(x.index_fill(0, index, -1))

    assert torch.tensor([
        [-1, 2, -1],
        [-1, 5, -1],
        [-1, 8, -1],
    ], dtype=torch.int).equal(x.index_fill(1, index, -1))
