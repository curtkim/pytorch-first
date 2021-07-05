import json
import os
import pathlib
import torch
from torchvision.ops import box_convert, box_area
import numpy as np

from src.metrics.bounding_box import BoundingBox
from src.metrics.enumerators import BBFormat, BBType


def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """
    Returns a list of files in a directory/path. Uses pathlib.
    """
    filenames = [file for file in path.glob(ext) if file.is_file()]
    assert len(filenames) > 0, f'No files found in path: {path}'
    return filenames


def read_json(path: pathlib.Path):
    with open(str(path), 'r') as fp:  # fp is the file pointer
        file = json.loads(s=fp.read())

    return file


def collate_double(batch):
    """
    collate function for the ObjectDetectionDataSet.
    Only used by the dataloader.
    """
    x = [sample['x'] for sample in batch]
    y = [sample['y'] for sample in batch]
    x_name = [sample['x_name'] for sample in batch]
    y_name = [sample['y_name'] for sample in batch]
    return x, y, x_name, y_name


def stats_dataset(dataset, transform: torch.nn.Module = False):
    """
    Iterates over the dataset and returns some stats.
    Can be useful to pick the right anchor box sizes.
    """
    stats = {
        'image_height': [],
        'image_width': [],
        'image_mean': [],
        'image_std': [],
        'boxes_height': [],
        'boxes_width': [],
        'boxes_num': [],
        'boxes_area': []
    }
    for batch in dataset:
        # Batch
        x, y, x_name, y_name = batch['x'], batch['y'], batch['x_name'], batch['y_name']

        # Transform
        if transform:
            x, y = transform([x], [y])
            x, y = x.tensors, y[0]

        # Image
        stats['image_height'].append(x.shape[-2])
        stats['image_width'].append(x.shape[-1])
        stats['image_mean'].append(x.mean().item())
        stats['image_std'].append(x.std().item())

        # Target
        wh = box_convert(y['boxes'], 'xyxy', 'xywh')[:, -2:]
        stats['boxes_height'].append(wh[:, -2])
        stats['boxes_width'].append(wh[:, -1])
        stats['boxes_num'].append(len(wh))
        stats['boxes_area'].append(box_area(y['boxes']))

    stats['image_height'] = torch.tensor(stats['image_height'], dtype=torch.float)
    stats['image_width'] = torch.tensor(stats['image_width'], dtype=torch.float)
    stats['image_mean'] = torch.tensor(stats['image_mean'], dtype=torch.float)
    stats['image_std'] = torch.tensor(stats['image_std'], dtype=torch.float)
    stats['boxes_height'] = torch.cat(stats['boxes_height'])
    stats['boxes_width'] = torch.cat(stats['boxes_width'])
    stats['boxes_area'] = torch.cat(stats['boxes_area'])
    stats['boxes_num'] = torch.tensor(stats['boxes_num'], dtype=torch.float)

    return stats


def from_file_to_boundingbox(file_name: pathlib.Path, groundtruth: bool = True):
    """Returns a list of BoundingBox objects from groundtruth or prediction."""
    file = torch.load(file_name)
    labels = file['labels']
    boxes = file['boxes']
    scores = file['scores'] if not groundtruth else [None] * len(boxes)

    gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED

    return [BoundingBox(image_name=file_name.stem,
                        class_id=l,
                        coordinates=tuple(bb),
                        format=BBFormat.XYX2Y2,
                        bb_type=gt,
                        confidence=s) for bb, l, s in zip(boxes, labels, scores)]


def from_dict_to_boundingbox(file: dict, name: str, groundtruth: bool = True):
    """Returns list of BoundingBox objects from groundtruth or prediction."""
    labels = file['labels']
    boxes = file['boxes']
    scores = np.array(file['scores'].cpu()) if not groundtruth else [None] * len(boxes)

    gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED

    return [BoundingBox(image_name=name,
                        class_id=int(l),
                        coordinates=tuple(bb),
                        format=BBFormat.XYX2Y2,
                        bb_type=gt,
                        confidence=s) for bb, l, s in zip(boxes, labels, scores)]


