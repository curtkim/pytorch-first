from typing import Dict
import torch
from torchvision import transforms
import webdataset as wds
from itertools import islice

def add_noise(source, noise=0.01):
    for inputs, targets in source:
        inputs = inputs + noise * torch.randn_like(inputs)
        yield inputs, targets


#url = "http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar"
#url = f"pipe:curl -L -s {url} || true"

url = "openimages-train-000000.tar"


def chain_long():
    dataset = wds.SimpleShardList(url)
    dataset = wds.Processor(dataset, wds.url_opener)
    dataset = wds.Processor(dataset, wds.tar_file_expander)
    dataset = wds.Processor(dataset, wds.group_by_keys)
    dataset = wds.Processor(dataset, wds.shuffle, 100)
    dataset = wds.Processor(dataset, wds.decode, wds.imagehandler("torchrgb"))
    dataset = wds.Processor(dataset, wds.to_tuple, "png;jpg;jpeg", "json")
    noisy_dataset = wds.Processor(dataset, add_noise, noise=0.02)

    print('type(noisy_dataset)', type(noisy_dataset))

    images, targets = next(iter(noisy_dataset))
    #print(images.shape)     # torch.Size([3, 683, 1024])


def chain_short():
    #short version
    dataset = wds.WebDataset(url)\
        .shuffle(100)\
        .decode("torchrgb")\
        .to_tuple("png;jpg;jpeg", "json")
    noisy_dataset = wds.Processor(dataset, add_noise, noise=0.02)
    print('type(noisy_dataset)', type(noisy_dataset))
    images, targets = next(iter(noisy_dataset))
    print(images.shape)     # torch.Size([3, 683, 1024])

    for target in targets:
        print(target)

    # {'ImageID': 'e7e826761f32f769', 'Source': 'activemil', 'LabelName': '/m/01g317', 'Confidence': '1', 'XMin': '0.551919', 'XMax': '0.976298', 'YMin': '0.025381', 'YMax': '0.910321', 'IsOccluded': '0', 'IsTruncated': '0', 'IsGroupOf': '0', 'IsDepiction': '0', 'IsInside': '0'}
    # {'ImageID': 'e7e826761f32f769', 'Source': 'activemil', 'LabelName': '/m/09j2d', 'Confidence': '1', 'XMin': '0.283296', 'XMax': '0.985327', 'YMin': '0.282572', 'YMax': '0.969543', 'IsOccluded': '0', 'IsTruncated': '0', 'IsGroupOf': '0', 'IsDepiction': '0', 'IsInside': '0'}
    # {'ImageID': 'e7e826761f32f769', 'Source': 'xclick', 'LabelName': '/m/0dzct', 'Confidence': '1', 'XMin': '0.629797', 'XMax': '0.832957', 'YMin': '0.130288', 'YMax': '0.467005', 'IsOccluded': '1', 'IsTruncated': '0', 'IsGroupOf': '0', 'IsDepiction': '0', 'IsInside': '0'}

print('long')
chain_long()

print('short')
chain_short()
