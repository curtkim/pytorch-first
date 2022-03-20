#import numpy as np
#import jax.numpy as np
#from jax import grad, vmap
from functorch import grad, vmap
import torch 
import matplotlib.pyplot as plt

def f(x):
    return x**2

x = torch.linspace(-10, 10, 500)
y = f(x)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121)
ax.plot(x.numpy(),y.numpy())
ax.set_title("y=f(x)")


y2 = vmap(grad(f))(x)

ax = fig.add_subplot(122)
ax.plot(x.numpy(), y2.numpy(), c='b', alpha=0.5, label='functorch')
ax.set_title("y=f'(x)")
ax.legend()

plt.show()
