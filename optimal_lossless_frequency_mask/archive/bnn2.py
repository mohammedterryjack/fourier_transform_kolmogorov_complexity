import torch
import torch.nn as nn
import torch.fft
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def make_triangle(size=64):
    s = np.zeros((size, size), dtype=np.float32)
    for i in range(size // 2):
        s[size//2 + i, size//2 - i : size//2 + i] = 1.0
    return s            




# ==== Parameters ====
W, H = 64, 64  # image size
lambda_reg = 1e3#1e-2  # L1 regularization weight
num_iters = 1000
lr = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==== Binary image (example) ====
# You can replace this with your own binary image
#s_np = (np.random.rand(W, H) > 0.5).astype(np.float32)
s_np = make_triangle(size=64)
s = torch.tensor(s_np, dtype=torch.float32, device=device)

# ==== Preprocessing ====
S = torch.fft.fft2(s)  # Frequency domain representation
logits_target = torch.logit(s.clamp(1e-6, 1 - 1e-6))  # logit(s) to avoid infinity

# ==== Learnable frequency mask ====
m_real = torch.randn(W, H, device=device, requires_grad=True)

# ==== Optimizer ====
optimizer = optim.Adam([m_real], lr=lr)

# ==== Training Loop ====
losses = []
for it in range(num_iters):
    optimizer.zero_grad()

    m = torch.sigmoid(m_real)  # mask constrained to [0,1]
    masked_S = m * S
    recon = torch.fft.ifft2(masked_S).real
    recon_logits = torch.logit(recon.clamp(1e-6, 1 - 1e-6))  # predicted logits

    # Loss: squared error in logit space + L1 sparsity
    recon_loss = torch.mean((recon_logits - logits_target) ** 2)
    sparsity_loss = lambda_reg * torch.mean(m)
    loss = recon_loss + sparsity_loss

    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if it % 100 == 0 or it == num_iters - 1:
        print(f"Iter {it:04d} | Loss: {loss.item():.6f} | Recon: {recon_loss.item():.6f} | Sparsity: {sparsity_loss.item():.6f}")


# ==== Postprocess ====
m_bin = (m.detach().cpu().numpy() > 0.5).astype(np.float32)
recon_final = torch.fft.ifft2(torch.tensor(m_bin, device=device) * S).real
recon_final_bin = (recon_final > 0.5).float()

# ==== Visualize ====
plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(s.cpu(), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Learned Mask (soft)")
plt.imshow(m.detach().cpu(), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Binary Mask")
plt.imshow(m_bin, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Reconstructed")
plt.imshow(recon_final_bin.cpu(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
