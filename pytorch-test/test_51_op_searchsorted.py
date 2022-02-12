import torch

def test_searchsorted():
    sorted_sequence = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
    values = torch.tensor([[3, 6, 9], [3, 6, 9]])

    torch.tensor([
        [1,3,4],
        [1,2,4]
    ]).equal(torch.searchsorted(sorted_sequence, values))
