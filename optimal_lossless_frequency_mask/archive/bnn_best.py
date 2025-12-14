"""Produces much sparser masks BUT the complex images are more lossy"""

import torch
import torch.nn as nn
import torch.fft
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from eca import OneDimensionalElementaryCellularAutomata
from matplotlib.pyplot import imshow


# ==== Parameters ====
rule_number=30
W, H = 500, 500
lambda_reg = 1e-2
num_iters = 1000
lr = 1e-2
beta = 10.0 # fixed sharpness
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==== Binary image ====
configuration = OneDimensionalElementaryCellularAutomata(lattice_width=W)
for _ in range(H-1):
    configuration.transition(rule_number=rule_number)
s_np = configuration.evolution()
s = torch.tensor(s_np, dtype=torch.float32, device=device)

# ==== Frequency domain ====
S = torch.fft.fft2(s)

# ==== Learnable binary mask logits ====
m_logits = torch.randn(W, H, device=device, requires_grad=True)

# ==== STE Binarizer ====
class STEBinarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# ==== Learnable thresholded sigmoid ====
class ThresholdedSigmoid(nn.Module):
    def __init__(self, beta=10.0):
        super().__init__()
        self.beta = beta
        self.theta = nn.Parameter(torch.tensor(0.0))  # learnable decision threshold

    def forward(self, x):
        return torch.sigmoid(self.beta * (x - self.theta))

    def binarize(self, x):
        return (x > self.theta).float()

# ==== Instantiate threshold module ====
theta_layer = ThresholdedSigmoid(beta=beta).to(device)

# ==== Optimizer ====
optimizer = optim.Adam([m_logits, theta_layer.theta], lr=lr)

# ==== Training Loop ====
losses = []
for it in range(num_iters):
    optimizer.zero_grad()

    m_soft = torch.tanh(m_logits)
    m_binary = STEBinarizer.apply(m_soft)

    masked_S = m_binary * S
    recon = torch.fft.ifft2(masked_S).real

    recon_prob = theta_layer(recon)
    recon_loss = torch.mean((recon_prob - s) ** 2)

    lambda_reg = torch.exp(theta_layer.theta)
    sparsity_loss = lambda_reg * torch.mean(m_binary)
    loss = recon_loss + sparsity_loss

    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if it % 100 == 0 or it == num_iters - 1:
        print(f"Iter {it:04d} | Loss: {loss.item():.6f} | θ: {theta_layer.theta.item():.4f} | Recon: {recon_loss.item():.6f} | Sparsity: {sparsity_loss.item():.6f}")

# ==== Final Reconstruction Using Learned θ ====
with torch.no_grad():
    m_final = (torch.tanh(m_logits) > 0).float()
    print(f"mask size = {m_final.sum().item()}")

    masked_S_final = m_final * S
    recon_final = torch.fft.ifft2(masked_S_final).real
    recon_bin_final = theta_layer.binarize(recon_final)


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
