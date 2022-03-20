import numpy as np
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
y1 = (f(x+delta_x) - f(x))/delta_x
y2 = 2*x

ax = fig.add_subplot(122)
ax.plot(x, y1, c='r', alpha=0.5, label='rate')
ax.plot(x, y2, c='b', alpha=0.5, label='rule')
ax.set_title("y=f'(x)")
ax.legend()

plt.show()
