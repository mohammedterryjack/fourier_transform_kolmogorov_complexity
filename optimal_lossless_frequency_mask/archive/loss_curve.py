import torch
import numpy as np
import matplotlib.pyplot as plt

# Dummy example: loss = (m1 - 1)^2 + (m2 + 2)^2  (simple quadratic)
def loss_fn(m1, m2):
    return (m1 - 1)**2 + (m2 + 2)**2

# Create grid of m1, m2 values
m1_vals = np.linspace(-3, 3, 100)
m2_vals = np.linspace(-3, 3, 100)
M1, M2 = np.meshgrid(m1_vals, m2_vals)

# Compute loss on grid
Loss_vals = loss_fn(M1, M2)

# Plot surface
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(M1, M2, Loss_vals, cmap='viridis', edgecolor='none')
ax.set_xlabel('m1')
ax.set_ylabel('m2')
ax.set_zlabel('Loss')
ax.set_title('Loss Surface')
plt.show()
