from itertools import chain


def test_chain():
    """
    flatten과 같은 효과
    """
    mylist = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    assert [1, 2, 3, 4, 5, 6] == list(chain(*mylist))
