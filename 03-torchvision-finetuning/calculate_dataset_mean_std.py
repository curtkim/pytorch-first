# https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
# https://www.thoughtco.com/sum-of-squares-formula-shortcut-3126266
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

    pixel_count = 0
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    for images, targets in data_loader:
        for image in images:
            h, w = image.shape[1], image.shape[2]
            pixel_count += h*w
            psum += image.sum(axis=[1, 2])
            psum_sq += (image ** 2).sum(axis=[1, 2])

    # mean and std
    total_mean = psum / pixel_count
    total_var = (psum_sq / pixel_count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    print('mean: ' + str(total_mean))
    print('std:  ' + str(total_std))


main()
