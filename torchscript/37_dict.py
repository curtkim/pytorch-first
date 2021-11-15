# Dict에 value type이 static 이어야 하는 것 같다.
from typing import Tuple, Dict, Any
import torch

@torch.jit.script
def doit(input, weight, config: Dict[str, int]):
    output = weight.mv(input)

    # 여기서 아래와 같은 에러가 발생함.
    # attribute lookup is not defined on python value of type 'MyConfig':
    if config['a'] >= 1:
        # This calls the `forward` method of the `nn.Linear` module, which will
        # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
        output = output + 1

    return output


weight = torch.rand(2, 3)
print(doit.code)
