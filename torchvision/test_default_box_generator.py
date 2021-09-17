import torch
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.image_list import ImageList


def test_scales():
    aspect_ratios = [[2]]
    dbox_generator = DefaultBoxGenerator(aspect_ratios)
    assert [0.15, 0.9] == dbox_generator.scales


def test__wh_pairs_minimal():
    aspect_ratios = [[2]]
    dbox_generator = DefaultBoxGenerator(aspect_ratios)

    torch.testing.assert_close([torch.tensor([
        [0.1500, 0.1500],
        [0.3674, 0.3674],
        [0.2121, 0.1061],
        [0.1061, 0.2121]])], dbox_generator._wh_pairs, rtol=1e-5, atol=1e-4)


def test__wh_pairs_ssd():
    # assert len(aspect_ratios) == len(scales)+1

    aspect_ratios = [
        [2],
        [2, 3],
        [2, 3],
        [2, 3],
        [2],
        [2]
    ]
    dbox_generator = DefaultBoxGenerator(aspect_ratios,
                                         scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                         steps=[8, 16, 32, 64, 100, 300])

    # aspect ratio의 개수만큼 tensor가 생성된다.
    torch.testing.assert_close([
        torch.tensor([
            [0.0700, 0.0700],
            [0.1025, 0.1025],
            [0.0990, 0.0495],
            [0.0495, 0.0990]]),
        torch.tensor([
            [0.1500, 0.1500],
            [0.2225, 0.2225],
            [0.2121, 0.1061],
            [0.1061, 0.2121],
            [0.2598, 0.0866],
            [0.0866, 0.2598]]),
        torch.tensor([
            [0.3300, 0.3300],
            [0.4102, 0.4102],
            [0.4667, 0.2333],
            [0.2333, 0.4667],
            [0.5716, 0.1905],
            [0.1905, 0.5716]]),
        torch.tensor([
            [0.5100, 0.5100],
            [0.5932, 0.5932],
            [0.7212, 0.3606],
            [0.3606, 0.7212],
            [0.8833, 0.2944],
            [0.2944, 0.8833]]),
        torch.tensor([
            [0.6900, 0.6900],
            [0.7748, 0.7748],
            [0.9758, 0.4879],
            [0.4879, 0.9758]]),
        torch.tensor([
            [0.8700, 0.8700],
            [0.9558, 0.9558],
            [1.2304, 0.6152],
            [0.6152, 1.2304]])], dbox_generator._wh_pairs, rtol=1e-5, atol=1e-4)


def test_grid_default_boxes_minimal():
    aspect_ratios = [[2]]
    dbox_generator = DefaultBoxGenerator(aspect_ratios)
    # DefaultBoxGenerator(aspect_ratios=[[2]], clip=True, scales=[0.15, 0.9], steps=None)

    default_boxes = dbox_generator._grid_default_boxes([[1, 1]], [15, 15], dtype=torch.float32)

    torch.testing.assert_close(torch.tensor([
        [0.5000, 0.5000, 0.1500, 0.1500],
        [0.5000, 0.5000, 0.3674, 0.3674],
        [0.5000, 0.5000, 0.2121, 0.1061],
        [0.5000, 0.5000, 0.1061, 0.2121]], dtype=torch.float32), default_boxes, rtol=1e-5, atol=1e-4)


def test_grid_default_boxes_ssd():
    aspect_ratios = [
        [2],
        [2, 3],
        [2, 3],
        [2, 3],
        [2],
        [2]
    ]
    dbox_generator = DefaultBoxGenerator(aspect_ratios,
                                         scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                         steps=[8, 16, 32, 64, 100, 300])
    print(dbox_generator._wh_pairs)

    default_boxes = dbox_generator._grid_default_boxes([[1, 1]], [15, 15], dtype=torch.float32)

    torch.testing.assert_close(torch.tensor([
        [0.2667, 0.2667, 0.0700, 0.0700],
        [0.2667, 0.2667, 0.1025, 0.1025],
        [0.2667, 0.2667, 0.0990, 0.0495],
        [0.2667, 0.2667, 0.0495, 0.0990]], dtype=torch.float32), default_boxes, rtol=1e-5, atol=1e-4)


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
