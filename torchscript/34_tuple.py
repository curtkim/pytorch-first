# 근데 NamedTuple인 Opt를 C++에서 어떻게 만들지?
#from typing import NamedTuple
from typing import Tuple
import torch
import torch.jit

#Opt = namedtuple('Config', ['int_value', 'float_value', 'bool_value'])
# class MyOption(NamedTuple):
#     int_value: int
#     float_value: float
#     bool_value: bool


@torch.jit.script
def doit(a: torch.Tensor, b: torch.Tensor, option: Tuple[int, float, bool]):
    if option[2]:
        return a + b
    else:
        return a - b


if __name__ == '__main__':
    doit.save("34_tuple.torchscript")
    print(doit.code)

    #opt = MyOption(int_value=1, float_value=0.5, bool_value=False)
    result = doit(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), (1, 0.5, False))
    print(result)
