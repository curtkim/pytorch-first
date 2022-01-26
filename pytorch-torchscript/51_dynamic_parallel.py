import torch

def foo(x):
    return torch.neg(x)

@torch.jit.script
def example(x):
    # Call `foo` using parallelism:
    # First, we "fork" off a task. This task will run `foo` with argument `x`
    future = torch.jit.fork(foo, x)

    # Call `foo` normally
    x_normal = foo(x)

    # Second, we "wait" on the task. Since the task may be running in
    # parallel, we have to "wait" for its result to become available.
    # Notice that by having lines of code between the "fork()" and "wait()"
    # call for a given Future, we can overlap computations so that they
    # run in parallel.
    x_parallel = torch.jit.wait(future)

    return x_normal, x_parallel


print(example(torch.ones(1)))               # (-1., -1.)
print(example(torch.ones(1).to('cuda')))    # (-1., -1.)
print(example(torch.ones(1)))               # (-1., -1.)
