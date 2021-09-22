from emlp.groups import SO, S
import emlp.reps as reps
import emlp
import haiku as hk
import emlp.nn.haiku as ehk
import jax.numpy as jnp
import numpy as np


so3_rep = reps.V(SO(3))
# grab a random group element
sampled_g = SO(3).sample()
dense_rep = so3_rep.rho(sampled_g)

# check its a member of SO(3)
# g @ g.T = I
print(dense_rep @ dense_rep.T)


point = np.array([0, 0, 1])
print('new point', dense_rep @ point.T)
print('norm', np.sqrt(np.sum((dense_rep @ point)**2)))


input_rep = 5 * so3_rep**0 + 5 * so3_rep**1
print('input rep', input_rep)
print('output rep', so3_rep)

input_point = np.random.randn(5 + 5 * 3)
print('input features', input_point[:5])
print('input positions\n', input_point[5:].reshape(5, 3))


model = emlp.nn.EMLP(input_rep, so3_rep, group=SO(3), num_layers=1)
output_point = model(input_point)
print('output', output_point)


trans_input_point = input_rep.rho_dense(sampled_g) @ input_point
print('transformed input features', trans_input_point[:5])
print('transformed input positions\n', trans_input_point[5:].reshape(5, 3))


print(model(trans_input_point), sampled_g @ output_point)
