from fudan_dataset import PennFudanDataset
from fudan_model import get_transform


def test_fudan_dataset():
    data_dir = '/data/datasets/PennFudanPed'
    dataset = PennFudanDataset(data_dir, get_transform(train=True))

    image, target = dataset.__getitem__(0)
    print(image.shape)
    print(target['boxes'])
    print(target['labels'])

