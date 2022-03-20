import jax.numpy as np
from jax import grad, vmap
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 + y**3

df_dx = grad(f, argnums=0)
df_dy = grad(f, argnums=1)

x = np.linspace(-10, 10, 20)
xv, yv = np.meshgrid(x, x, indexing='ij')

dfx = vmap(vmap(df_dx))(xv, yv)
dfy = vmap(vmap(df_dy))(xv, yv)
size = np.sqrt(dfx**2 + dfy**2)
dir_x = dfx/size
dir_y = dfy/size


plt.figure(figsize=(6,6))
plt.quiver(xv, yv, dir_x, dir_y, size, cmap="viridis")
plt.show()

