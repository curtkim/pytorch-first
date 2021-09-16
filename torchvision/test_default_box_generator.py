import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator, DefaultBoxGenerator
from torchvision.models.detection.image_list import ImageList


def test_default_box_generator():
    images = torch.zeros(2, 3, 15, 15)
    features = [torch.zeros(2, 8, 1, 1)]
    image_shapes = [i.shape[-2:] for i in images]
    images = ImageList(images, image_shapes)

    aspect_ratios = [[2]]
    dbox_generator = DefaultBoxGenerator(aspect_ratios)
    # DefaultBoxGenerator(aspect_ratios=[[2]], clip=True, scales=[0.15, 0.9], steps=None)

    dbox_generator.eval()
    dboxes = dbox_generator(images, features)

    dboxes_output = torch.tensor([
        [6.3750, 6.3750, 8.6250, 8.6250],
        [4.7443, 4.7443, 10.2557, 10.2557],
        [5.9090, 6.7045, 9.0910, 8.2955],
        [6.7045, 5.9090, 8.2955, 9.0910]
    ])

    assert len(dboxes) == 2
    assert tuple(dboxes[0].shape) == (4, 4)
    assert tuple(dboxes[1].shape) == (4, 4)
    torch.testing.assert_close(dboxes[0], dboxes_output, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(dboxes[1], dboxes_output, rtol=1e-5, atol=1e-8)