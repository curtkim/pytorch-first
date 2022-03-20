#import numpy as np
import jax.numpy as np
from jax import grad, vmap
import matplotlib.pyplot as plt

def f(x):
    return x**2

x = np.linspace(-10, 10, 500)
y = f(x)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121)
ax.plot(x,y)
ax.set_title("y=f(x)")


delta_x = 0.0001
y2 = vmap(grad(f))(x)
# vmap을사용하지 않으면, Gradient only defined for scalar-output function. error가 발생한다

ax = fig.add_subplot(122)
ax.plot(x, y2, c='b', alpha=0.5, label='jax')
ax.set_title("y=f'(x)")
ax.legend()

plt.show()
