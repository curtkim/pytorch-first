import torch
from torchvision.models.detection._utils import SSDMatcher


# find for each default box the best candidate ground truth
def test_ssd_matcher():
    # match_quality_matrix (Tensor[float]):
    # an MxN tensor, containing the pairwise quality
    # between M ground-truth elements and N predicted elements.

    match_quality_matrix = torch.tensor([
        [0.0, 0.0, 0.1, 0.6],
        [0.0, 0.5, 0.1, 0.0],
    ])

    matcher = SSDMatcher(0.4)
    assert torch.tensor(
        [-1, 1, -1, 0]  # prediction에 대해서 몇번째 GT에 매칭 되는지를 반환하는 것 같다.
        , dtype=torch.int64).equal(matcher(match_quality_matrix))

