import torch
from effdet.config import get_fpn_config
from effdet.efficientdet import ResampleFeatureMap


def test_resample_feature_map_downsample():
    inp = torch.randn(1, 40, 64, 64)
    resample = ResampleFeatureMap(in_channels=40, out_channels=112, reduction_ratio=2)
    out = resample(inp)
    assert torch.Size([1, 112, 32, 32]) == out.shape


def test_resample_feature_map_upsample():
    inp = torch.randn(1, 40, 64, 64)
    resample = ResampleFeatureMap(in_channels=40, out_channels=112, reduction_ratio=0.5)
    out = resample(inp)
    assert torch.Size([1, 112, 128, 128]) == out.shape


def test_get_fpn_config():
    fpn_config = get_fpn_config(None)
    # {'nodes': [{'reduction': 64, 'inputs_offsets': [3, 4], 'weight_method': 'fastattn'},
    #            {'reduction': 32, 'inputs_offsets': [2, 5], 'weight_method': 'fastattn'},
    #            {'reduction': 16, 'inputs_offsets': [1, 6], 'weight_method': 'fastattn'},
    #            {'reduction': 8, 'inputs_offsets': [0, 7], 'weight_method': 'fastattn'},
    #            {'reduction': 16, 'inputs_offsets': [1, 7, 8], 'weight_method': 'fastattn'},
    #            {'reduction': 32, 'inputs_offsets': [2, 6, 9], 'weight_method': 'fastattn'},
    #            {'reduction': 64, 'inputs_offsets': [3, 5, 10], 'weight_method': 'fastattn'},
    #            {'reduction': 128, 'inputs_offsets': [4, 11], 'weight_method': 'fastattn'}]}