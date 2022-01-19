import torch
from torch import tensor
from torchmetrics.detection.map import MeanAveragePrecision


def test_map2():
    preds = [
        # train_idx, class_prediction, prob_score, x, y, width, height
        dict(boxes=torch.Tensor([[0.55, 0.2, 0.3, 0.2]]), scores=torch.Tensor([0.9]), labels=torch.IntTensor([0])),
        dict(boxes=torch.Tensor([[0.35, 0.6, 0.3, 0.2]]), scores=torch.Tensor([0.8]), labels=torch.IntTensor([0])),
        dict(boxes=torch.Tensor([[0.8, 0.7, 0.2, 0.2]]), scores=torch.Tensor([0.7]), labels=torch.IntTensor([0])),
    ]
    targets = [
        dict(boxes=torch.Tensor([[0.55, 0.2, 0.3, 0.2]]), labels=torch.IntTensor([0])),
        dict(boxes=torch.Tensor([[0.35, 0.6, 0.3, 0.2]]), labels=torch.IntTensor([0])),
        dict(boxes=torch.Tensor([[0.8, 0.7, 0.2, 0.2]]), labels=torch.IntTensor([0])),
    ]
    metric = MeanAveragePrecision(box_format="cxcywh", iou_thresholds=[0.5])
    metric.update(preds, targets)

    assert {
               'map': tensor(1.),
               'map_50': tensor(1.),
               'map_75': tensor(-1.),
               'map_large': tensor(-1.),
               'map_medium': tensor(-1.),
               'map_per_class': tensor(-1.),
               'map_small': tensor(1.),
               'mar_1': tensor(1.),
               'mar_10': tensor(1.),
               'mar_100': tensor(1.),
               'mar_100_per_class': tensor(-1.),
               'mar_large': tensor(-1.),
               'mar_medium': tensor(-1.),
               'mar_small': tensor(1.)
           } == metric.compute()
    # from pprint import pprint
    # pprint(metric.compute())


def test_map1():
    preds = [
        dict(
            boxes=torch.Tensor([[258.0, 41.0, 606.0, 285.0]]), scores=torch.Tensor([0.536]), labels=torch.IntTensor([0]))
    ]
    target = [
        dict(boxes=torch.Tensor([[214.0, 41.0, 562.0, 285.0]]), labels=torch.IntTensor([0]))
    ]

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    assert {
               'map': tensor(0.6000),
               'map_50': tensor(1.),
               'map_75': tensor(1.),
               'map_large': tensor(0.6000),
               'map_medium': tensor(-1.),
               'map_per_class': tensor(-1.),
               'map_small': tensor(-1.),
               'mar_1': tensor(0.6000),
               'mar_10': tensor(0.6000),
               'mar_100': tensor(0.6000),
               'mar_100_per_class': tensor(-1.),
               'mar_large': tensor(0.6000),
               'mar_medium': tensor(-1.),
               'mar_small': tensor(-1.)
           } == metric.compute()

# from pprint import pprint
# pprint(metric.compute())
