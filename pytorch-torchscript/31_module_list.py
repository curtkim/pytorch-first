import torch
from torch import nn


class SubModule(torch.nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()
        self.weight = nn.Parameter(torch.randn(2))

    def forward(self, input):
        return self.weight + input


class MyModule1(torch.nn.Module):
    #__constants__ = ['mods']

    def __init__(self):
        super(MyModule1, self).__init__()
        self.mods = torch.nn.ModuleList([SubModule() for i in range(10)])

    def forward(self, v):
        # 성공함.
        for module in self.mods:
            v = module(v)


class MyModule2(torch.nn.Module):
    # __constants__ = ['mods']

    def __init__(self):
        super(MyModule2, self).__init__()
        self.mods1 = torch.nn.ModuleList([SubModule() for i in range(5)])
        self.mods2 = torch.nn.ModuleList([SubModule() for i in range(5)])

    def forward(self, v):
        # 성공함.
        for module in self.mods1:
            v = module(v)
        for module in self.mods2:
            v = module(v)


class MyModule3Fail(torch.nn.Module):
    # __constants__ = ['mods']

    def __init__(self):
        super(MyModule3Fail, self).__init__()
        self.mods = torch.nn.ModuleList([SubModule() for i in range(10)])

    def forward(self, v):
        # 실패함.
        for module in self.mods[0:1]:
            v = module(v)
        return v


m = torch.jit.script(MyModule1())
m = torch.jit.script(MyModule2())
#m = torch.jit.script(MyModule3Fail())
