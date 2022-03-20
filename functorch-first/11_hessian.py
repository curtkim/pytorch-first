import torch 
from functorch import hessian

def f(x):
    return x.sin().sum()

x = torch.randn(5)
hess = hessian(f)(x)
print(hess)

