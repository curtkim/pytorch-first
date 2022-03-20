import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
xv, yv = np.meshgrid(x, x, indexing='ij')

zv = xv**2 + yv**3

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xv, yv, zv, cmap='viridis')
plt.show()

