import torch
from torchvision.models.detection.image_list import ImageList


def test():
    images = torch.zeros(2, 3, 15, 15)
    image_shapes = [i.shape[-2:] for i in images]
    image_list = ImageList(images, image_shapes)

    assert [torch.Size([15, 15]), torch.Size([15, 15])] == image_list.image_sizes