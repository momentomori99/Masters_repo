import numpy as np
import matplotlib.pyplot as plt

x_vec = np.linspace(0, 10, 100)
y_vec = np.linspace(0, 10, 100)

X, Y = np.meshgrid(x_vec, y_vec)
u = np.ones_like(X)
v = X - Y
Z = np.sqrt(X**2 + Y**2)

plt.figure(figsize=(8, 6))
plt.quiver(X, Y, u, v)
plt.show()