from timeit import timeit
import jax.numpy as jnp
from jax import jacfwd, jacrev, jit
import torch

X = torch.randn((1000, ))


def jax_fn(x):
    return jnp.sum(jnp.square(x))


jit_jax_fn = jit(jacfwd(jacrev(jax_fn)))
X = jnp.array(X)
print(jit_jax_fn(X))
