import torch
from catalyst import metrics


def test_additive_metric_update():
    metric = metrics.AdditiveMetric(True)
    metric.update(1, 1)
    metric.update(2, 1)
    metric.update(3, 1)
    assert (2.0, 1.0) == metric.compute()


def test_additive_metric_update_batch():
    metric = metrics.AdditiveMetric(True)
    metric.update(1, 2)
    metric.update(2, 2)
    metric.update(3, 2)
    mean, std = metric.compute()
    assert 2.0 == mean
