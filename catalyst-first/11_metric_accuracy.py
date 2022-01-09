import torch
from catalyst import metrics


def test_accuracy_metric():
    outputs = torch.tensor([
        [0.2, 0.5, 0.0, 0.3],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.1, 0.6, 0.3],
        [0.0, 0.8, 0.2, 0.0],
    ])
    targets = torch.tensor([3, 0, 2, 2])
    metric = metrics.AccuracyMetric(topk_args=(1, 3))

    metric.reset()
    metric.update(outputs, targets)

    assert (
     (0.5, 1.0),  # top1, top3 mean
     (0.0, 0.0),  # top1, top3 std
    ) == metric.compute()

    assert {
        'accuracy01': 0.5,
        'accuracy01/std': 0.0,
        'accuracy03': 1.0,
        'accuracy03/std': 0.0,
    } == metric.compute_key_value()

    metric.reset()
    assert (
        (0.5, 1.0),  # top1, top3 mean
        (0.0, 0.0),  # top1, top3 std
    ) == metric(outputs, targets)


def test_accuracy_metric_1():
    outputs = torch.tensor([
        [0.2, 0.5, 0.0, 0.3],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.1, 0.6, 0.3],
        [0.0, 0.8, 0.2, 0.0],
    ])
    targets = torch.tensor([3, 0, 2, 2])
    metric = metrics.AccuracyMetric(topk_args=(1,))

    metric.reset()
    metric.update(outputs, targets)

    assert (
     (0.5,),  # top1 mean
     (0.0,),  # top1 std
    ) == metric.compute()
