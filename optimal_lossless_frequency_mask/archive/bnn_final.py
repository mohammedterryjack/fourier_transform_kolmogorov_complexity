import torch
import torch.fft
from eca import OneDimensionalElementaryCellularAutomata
import matplotlib.pyplot as plt

# --- Settings ---
W, H = 500, 500
rule_number = 30
lambda_reg = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Inputs ---
configuration = OneDimensionalElementaryCellularAutomata(lattice_width=W)
for _ in range(H-1):
    configuration.transition(rule_number=rule_number)
s_np = configuration.evolution()
s = torch.tensor(s_np, dtype=torch.float32, device=device)

#s = torch.randint(0, 2, (W, H), dtype=torch.float32).to(device)  # binary image {0,1}
s.requires_grad = False

# --- Inverse Sigmoid Parameters ---
theta_inv = 0.5
eps = 1e-6
s_clamped = s.clamp(eps, 1 - eps)

# --- Learnable scalar m ---
m = torch.nn.Parameter(torch.tensor(0.5, device=device))

# --- Annealing Schedule for beta ---
def beta_schedule(epoch, max_beta=50, total_epochs=100):
    return max_beta * (epoch / total_epochs)

# --- Optimizer ---
optimizer = torch.optim.SGD([m], lr=0.1)

# --- Training Loop ---
num_epochs = 1000
for epoch in range(num_epochs):
    beta = beta_schedule(epoch+1)

    # === Step 1: Compute inverse sigmoid with theta ===
    sigmoid_inv = theta_inv + (1.0 / beta) * torch.log(s_clamped / (1 - s_clamped))

    # === Step 2: Compute FFTs (cached after first epoch) ===
    if epoch == 0:
        F_s = torch.fft.fft2(s)
        F_sigmoid_inv = torch.fft.fft2(sigmoid_inv)
        F_s_flat = F_s.reshape(-1)
        F_t_flat = F_sigmoid_inv.reshape(-1)

    # === Step 3: Soft approximation of m using sigmoid (theta = 0) ===
    m_soft = 2 * torch.sigmoid(beta * m) - 1  # sigmoid(βm), no θ shift here

    # === Step 4: Predicted spectrum ===
    pred = m_soft * F_s_flat

    # === Step 5: Loss (L2 + L1) ===
    loss_l2 = torch.sum(torch.abs(pred - F_t_flat) ** 2)
    loss_l1 = lambda_reg * torch.abs(m)
    loss = loss_l2 + loss_l1

    # === Step 6: Optimize ===
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # === Logging ===
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch:3d} | beta = {beta:.2f} | m = {m.item():.6f} | Loss = {loss.item():.6f}")

with torch.no_grad():
    # Broadcast m_soft to match image size
    m_soft_mask = m_soft * torch.ones_like(s)

    print(f"mask sum = {m_soft_mask.sum().item():.2f}")

    # Predict in frequency domain
    pred = m_soft * F_s_flat
    recon_final = torch.fft.ifft2(pred.reshape(W, H)).real
    recon_bin_final = (recon_final > 0.5).float()

# ==== Visualize ====
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(s.cpu(), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Binary Mask (m)")
plt.imshow(m_soft_mask.cpu(), cmap='gray', vmin=0, vmax=1)
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
