# https://mingrammer.com/understanding-the-asterisk-of-python/
from utils.utils import collate_fn


def test_zip_asterisk():
    [[1, 3, 5], [2, 4, 6]] == (item for item in zip(*[[1, 2], [3, 4], [5, 6]]))


def test_zip_asterisk_tuple():
    ([1, 3, 5], [2, 4, 6]) == tuple(zip(*[[1, 2], [3, 4], [5, 6]]))


def test_collate_fn():
    batch = [
        [1, 2],
        [3, 4],
        [5, 6]
    ]
    ([1, 3, 5], [2, 4, 6]) == collate_fn(batch)
