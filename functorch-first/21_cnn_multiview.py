# https://www.sscardapane.it/tutorials/functorch/  
from functools import partial
import torch
from torch import nn
import functorch


tinycnn = nn.Sequential(
        nn.Conv2d(3,5, 3),
        nn.Conv2d(5,5, 3)
)


x = torch.randn((1, 3, 16, 16))
y = tinycnn(x)
print(y.shape)

x = torch.randn((1, 3, 16, 16, 5))
model_fcn, params = functorch.make_functional(tinycnn)
model_fnc_multiview = functorch.vmap(partial(model_fcn, params), in_dims=4, out_dims=4)
print(model_fnc_multiview(x).shape)

