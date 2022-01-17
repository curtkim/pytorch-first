import torch
import torch.nn.functional as F


def test_softmax():
    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    torch.testing.assert_close(
        torch.tensor([0.0900, 0.2447, 0.6652], dtype=torch.float32),
        F.softmax(a),
        rtol=1e-3, atol=1e-4
    )

    b = torch.tensor([
        [1, 2, 3],
        [4, 5, 6]
        ], dtype=torch.float32)

    torch.testing.assert_close(
        torch.tensor([
            [0.0900, 0.2447, 0.6652],
            [0.0900, 0.2447, 0.6652],
        ], dtype=torch.float32),
        F.softmax(b),
        rtol=1e-3, atol=1e-4
    )

    torch.testing.assert_close(
        torch.tensor([
            [0.0474, 0.0474, 0.0474],
            [0.9526, 0.9526, 0.9526],
        ], dtype=torch.float32),
        F.softmax(b, dim=0),            # dim을 주지 않으면 default로 마지막 dim으로 처리되는 것 같다.
        rtol=1e-3, atol=1e-4
    )
