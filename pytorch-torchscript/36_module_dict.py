from typing import Tuple, Dict
import torch


class MyModule(torch.nn.Module):
    def __init__(self, weight, N: int, M: int, config: Dict[str, int]):
        super(MyModule, self).__init__()

        self.config = config
        # This parameter will be copied to the new ScriptModule
        self.weight = torch.nn.Parameter(weight)

        # When this submodule is used, it will be compiled
        self.linear = torch.nn.Linear(N, M)

    def forward(self, input):
        output = self.weight.mv(input)

        # 여기서 아래와 같은 에러가 발생함.
        # attribute lookup is not defined on python value of type 'MyConfig':
        if self.config['a'] >= 1:
            # This calls the `forward` method of the `nn.Linear` module, which will
            # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
            output = self.linear(output)

        return output


weight = torch.rand(2, 3)
print(weight)
scripted_module = torch.jit.script(MyModule(weight, 2, 3, {'a': 1}))
print(scripted_module.code)
print(scripted_module.weight)
print(scripted_module.config)
