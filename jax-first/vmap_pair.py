# https://twitter.com/cgarciae88/status/1511763733676277766

import jax
import jax.numpy as jnp


def distance(a, b):
    return jnp.linalg.norm(a - b)

# vmap based combinator to operate on all pairs
def all_pairs(f):
    f = jax.vmap(f, in_axes=(None, 0))
    f = jax.vmap(f, in_axes=(0, None))
    return f

distances = all_pairs(distance)


# create some test data
A = jnp.array([ [0,0], [1,1], [2,2]])
B = jnp.array([ [-10, -10], [-20,-20]])

d00 = distance(A[0], B[0])
D = distances(A, B)
print(D)

