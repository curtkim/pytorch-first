import torch
import collections

Point = collections.namedtuple('Point', ['x', 'y'])


@torch.jit.script
def total(point):
    # type: (Point) -> Tensor
    return point.x + point.y


p = Point(x=torch.rand(3), y=torch.rand(3))
print(total(p))
