import torch

# from IPython.display import display, Math
# Define the graph a,b,c,d are leaf nodes and e is the root node
# The graph is constructed with every line since the
# computational graphs are dynamic in PyTorch
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)
c = torch.tensor([5.0], requires_grad=True)
d = torch.tensor([10.0], requires_grad=True)

u = a * b
t = torch.log(d)
v = t * c
t.retain_grad()
e = u + v

print('a', a.is_leaf, a.grad_fn, a.grad)
print('b', b.is_leaf, b.grad_fn, b.grad)
print('c', c.is_leaf, c.grad_fn, c.grad)
print('d', d.is_leaf, d.grad_fn, d.grad)
print('t', t.is_leaf, t.grad_fn, t.grad)
print('u', u.is_leaf, u.grad_fn, u.grad)
print('v', v.is_leaf, v.grad_fn, v.grad)
print('e', e.is_leaf, e.grad_fn, e.grad)

e.backward()

print('a', a.is_leaf, a.grad_fn, a.grad)
print('b', b.is_leaf, b.grad_fn, b.grad)
print('c', c.is_leaf, c.grad_fn, c.grad)
print('d', d.is_leaf, d.grad_fn, d.grad)
print('t', t.is_leaf, t.grad_fn, t.grad)
print('u', u.is_leaf, u.grad_fn, u.grad)
print('v', v.is_leaf, v.grad_fn, v.grad)
print('e', e.is_leaf, e.grad_fn, e.grad)
