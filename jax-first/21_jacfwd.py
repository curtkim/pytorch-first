from jax import jacfwd
import jax.numpy as jnp


def mapping(v):
    x = v[0]
    y = v[1]
    z = v[2]
    return jnp.array([x * x, y * z])


# 3 inputs, 2 outputs
# [d/dx x^2, d/dy x^2, d/dz x^2]
# [d/dx yz,  d/dy yz,  d/dz yz]

# [2x, 0, 0]
# [0, z, y]

f = jacfwd(mapping)
v = jnp.array([4., 5., 9.])
print(f(v))
