import torch


def test_storage():
    points = torch.tensor([
        [4.0, 1.0],
        [5.0, 3.0],
        [2.0, 1.0]
    ])
    # print(points.storage())

    points_storage = points.storage()
    # print(points_storage[0])

    points_storage[0] = 2.0

    torch.tensor([
        [2.0, 1.0],
        [5.0, 3.0],
        [2.0, 1.0]
    ]).equal(points)


def test_storage_offset_and_stride():
    points = torch.tensor([
        [4.0, 1.0],
        [5.0, 3.0],
        [2.0, 1.0]
    ])
    assert (2, 1) == points.stride()

    second_point = points[1]
    assert second_point.storage_offset() == 2
    assert torch.Size([2]) == second_point.shape
    # has  one  less  dimension,
    assert (1, ) == second_point.stride()


def test_clone():
    points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
    second_point = points[1].clone()
    second_point[0] = 10.0
    # not changed
    torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]]).equal(points)


def test_transpose_contiguous():
    points = torch.tensor([
        [4.0, 1.0],
        [5.0, 3.0],
        [2.0, 1.0]
    ])
    points_t = points.t()
    points_t[0, 0] = 9.0
    # print(points)
    # print(points.storage()[0])
    # print(points_t.storage()[0])
    #assert id(points.storage()) == id(points_t.storage())

    assert points.is_contiguous()
    assert not points_t.is_contiguous()

    assert (2, 1) == points.stride()
    assert (1, 2) == points_t.stride()

    points_t_cont = points_t.contiguous()
    assert (3, 1) == points_t_cont.stride()



