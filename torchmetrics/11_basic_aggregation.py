import torch
from torchmetrics import CatMetric, MaxMetric, MeanMetric


def test_cat_metric():
    metric = CatMetric()
    metric.update(1)
    metric.update(torch.tensor([2, 3]))
    assert torch.tensor([1, 2, 3], dtype=torch.float32).equal(metric.compute())


def test_max_metric():
    metric = MaxMetric()
    metric.update(1.)
    metric.update(torch.tensor([2., 3.]))
    assert torch.allclose(torch.tensor([3.], dtype=torch.float32), metric.compute())


def test_mean_metric():
    metric = MeanMetric()
    metric.update(1.)
    metric.update(torch.tensor([2., 3.]))
    assert torch.allclose(torch.tensor([2.], dtype=torch.float32), metric.compute())



