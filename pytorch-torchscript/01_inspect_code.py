import torch


@torch.jit.script
def foo(len: int) -> torch.Tensor:
    rv = torch.zeros(3, 4)
    for i in range(len):
        if i < 10:
            rv = rv - 1.0
        else:
            rv = rv + 1.0
    return rv


print(foo(1))
foo.save('foo.pt')
loaded_foo = torch.jit.load('foo.pt')
print(loaded_foo(1))

assert torch.allclose(foo(1), loaded_foo(1))

# print code
print(foo.code)

# def foo(len: int) -> Tensor:
#   rv = torch.zeros([3, 4])
#   rv0 = rv
#   for i in range(len):
#     if torch.lt(i, 10):
#       rv1 = torch.sub(rv0, 1.)
#     else:
#       rv1 = torch.add(rv0, 1.)
#     rv0 = rv1
#   return rv0


# print ir
# print(foo.graph)


