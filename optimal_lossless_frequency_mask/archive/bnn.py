import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import torch

def step_ifft_masked(s, m):
    """
    s: input tensor (image), shape [H, W]
    m: frequency mask, shape [H, W], binary or real-valued
    Returns:
        Binary step-activated reconstruction
    """
    s_fft = torch.fft.fft2(s)
    masked_fft = m * s_fft
    recon = torch.fft.ifft2(masked_fft).real
    return (recon > 0.5).float()


def binary_mask(logits):
    prob = torch.sigmoid(logits)
    m_hard = (prob > 0.5).float()
    # Use STE trick:
    m = m_hard.detach() - prob.detach() + prob
    return m

def logit(x, eps=1e-6):
    x = torch.clamp(x, eps, 1 - eps)
    return torch.log(x / (1 - x))

class FFTMaskModel(nn.Module):
    def __init__(self, shape):
        super().__init__()
        H, W = shape
        # Initialize real-valued mask with sigmoid parameterization
        self.logits = nn.Parameter(torch.randn(H, W))

    def forward(self, s):
        # s: input in [0, 1]
        s_fft = torch.fft.fft2(s)

        # Relaxed binary mask using sigmoid
        #m = torch.sigmoid(self.logits)
        m = binary_mask(self.logits)

        # Elementwise multiply in frequency domain
        masked_fft = m * s_fft

        # Inverse FFT and return real part
        recon = torch.fft.ifft2(masked_fft).real
        return recon, m

def reconstruction_loss(recon, s, m, lam):
    target = logit(s)
    l2 = F.mse_loss(recon, target)
    l1 = torch.sum(torch.abs(m))
    return l2 + lam * l1

def make_triangle(size=64):
    s = np.zeros((size, size), dtype=np.float32)
    for i in range(size // 2):
        s[size//2 + i, size//2 - i : size//2 + i] = 1.0
    return s


# Example image: random or real image normalized to [0, 1]
H, W = 64, 64
#s_np = np.random.randint(0, 2, (H, W))
#s = torch.tensor(s_np, device='cpu')  # put on GPU if available
s_np = make_triangle(size=64)
s = torch.tensor(s_np)

model = FFTMaskModel((H, W))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
lam = 1e-1
#lam = 1e-2  # L1 sparsity penalty

for epoch in range(10000):
    optimizer.zero_grad()
    recon, m = model(s)
    loss = reconstruction_loss(recon, s, m, lam)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Sparsity: {m.mean().item():.4f}")

binary_output = step_ifft_masked(s, m)
# Show result
plt.subplot(1, 3, 1); plt.imshow(s.cpu(), cmap='gray'); plt.title('Original')
plt.subplot(1, 3, 2); plt.imshow(m.detach().cpu(), cmap='gray'); plt.title('Mask')
plt.subplot(1, 3, 3); plt.imshow(binary_output.cpu(), cmap='gray'); plt.title('Reconstruction')
plt.show()
