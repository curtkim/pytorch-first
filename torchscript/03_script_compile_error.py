import torch


@torch.jit.script
def foo1():
    # allowed
    for n in range(0, 5):
        print(n)

foo1()

@torch.jit.script
def foo2():
    # not allowed (raises compile error)
    ns = [1, 2, 3, 4]
    for n in ns:
        print(n)
        ns.pop()

foo2()
