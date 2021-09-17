import torch
import utils.utils as utils

from fudan_dataset import PennFudanDataset
from fudan_model import get_transform


def main():
    data_dir = '/data/datasets/PennFudanPed'
    dataset = PennFudanDataset(data_dir, get_transform(train=True))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn     # image들의 size가 다른 경우에 필요하다.
    )

    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    for images, targets in data_loader:
        #print(images)
        image = images[0]
        print(image.shape)
        print(image.sum(axis=[1, 2]))

        #target = targets[0]
        #print(image.dtype, image.shape, target['boxes'].shape)


main()
