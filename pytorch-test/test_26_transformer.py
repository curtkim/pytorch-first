import torch


def test_transformer():
    transformer_model = torch.nn.Transformer(nhead=16, num_encoder_layers=12)
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    out = transformer_model(src, tgt)
    assert torch.Size([20, 32, 512]) == out.shape
