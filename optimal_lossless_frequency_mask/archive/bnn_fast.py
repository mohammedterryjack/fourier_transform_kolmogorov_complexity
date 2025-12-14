"""Produces lossless masks for even complex images - not sure if its the most sparse though"""

import torch
import torch.nn as nn
import torch.fft
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from eca import OneDimensionalElementaryCellularAutomata

# ==== Parameters ====
W, H = 500, 500
rule_number = 1212
lambda_reg = 1e-2
num_iters = 1000
lr = 1e-2
beta = 10.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learn_theta = True

if learn_theta:
    theta = torch.nn.Parameter(torch.tensor(0.5, device=device, dtype=torch.float32))
else:
    theta = 0.5

loss_vals = []
theta_vals = []
sparsity_vals = []
exactness_vals = []

# ==== Binary image ====
configuration = OneDimensionalElementaryCellularAutomata(lattice_width=W)
for _ in range(H - 1):
    configuration.transition(rule_number=rule_number)
s_np = configuration.evolution()
s = torch.tensor(s_np, dtype=torch.float32, device=device)

# ==== Precompute FFT and inverse sigmoid ====
with torch.no_grad():
    F_s = torch.fft.fft2(s)

    # Inverse sigmoid: sigmoid^-1(s) = θ + (1/β) * log(s / (1 - s))
    eps = 1e-4
    s_clamped = s.clamp(min=eps, max=1 - eps)
    sigmoid_inv_s = theta + (1.0 / beta) * torch.log(s_clamped / (1 - s_clamped))

# ==== Learnable binary mask (logits) ====
m_logits = torch.randn(W, H, device=device, requires_grad=True)

# ==== Optimizer ====
if learn_theta:
    optimizer = optim.Adam([m_logits, theta], lr=lr)
else:
    optimizer = optim.Adam([m_logits], lr=lr)
# ==== Training Loop ====
losses = []
for it in range(num_iters):
    optimizer.zero_grad()

    # Soft mask from logits in (0, 1)
    m_soft = (torch.tanh(m_logits) + 1) / 2.0
    

    # Apply mask in frequency domain
    masked_F = m_soft * F_s
    recon = torch.fft.ifft2(masked_F).real

    # Loss: image-space MSE + L1 sparsity
    recon_loss = torch.mean((recon - sigmoid_inv_s) ** 2)

    lambda_reg = torch.exp(theta if learn_theta else torch.tensor(theta, device=device, dtype=torch.float32))
    sparsity_loss = lambda_reg * torch.mean(m_soft)
    loss = recon_loss + sparsity_loss

    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if it % 100 == 0 or it == num_iters - 1:
        print(f"Iter {it:04d} | Loss: {loss.item():.6f} | θ: {theta.item() if learn_theta else theta:.4f} | Recon: {recon_loss.item():.6f} | Sparsity: {sparsity_loss.item():.6f}")

    loss_vals.append(loss.item())
    theta_vals.append(theta.item() if learn_theta else theta)
    sparsity_vals.append(m_soft.mean().item())
    exactness_vals.append(recon_loss.item())

# ==== Plot training loss ====

plt.figure(figsize=(15, 4))

plt.subplot(1, 4, 1)
plt.plot(loss_vals, label='Sparsity (mean of m)')
plt.xlabel('Loss')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.grid(True)

plt.subplot(1, 4, 2)
plt.plot(sparsity_vals, label='Sparsity (mean of m)')
plt.xlabel('Iteration')
plt.ylabel('Mean of m')
plt.title('Sparsity Over Time')
plt.grid(True)

plt.subplot(1, 4, 3)
plt.plot(theta_vals, label='Theta', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Theta')
plt.title('Theta Over Time')
plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(exactness_vals, label='Exactness (Recon Loss)', color='green')
plt.xlabel('Iteration')
plt.ylabel('Reconstruction Loss')
plt.title('Exactness Over Time')
plt.grid(True)

plt.tight_layout()
plt.show()


# ==== Final Reconstruction ====
with torch.no_grad():
    m_final = (torch.tanh(m_logits) > 0).float()
    print(f"mask size = {m_final.sum().item()}")

    masked_S_final = m_final * F_s
    recon_final = torch.fft.ifft2(masked_S_final).real
    recon_bin_final = (recon_final > 0.5).float()

# ==== Visualize ====
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(s.cpu(), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Binary Mask")
plt.imshow(m_final.cpu(), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Reconstruction")
plt.imshow(recon_bin_final.cpu(), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Difference")
plt.imshow(torch.abs(recon_bin_final.cpu() - s.cpu()), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
