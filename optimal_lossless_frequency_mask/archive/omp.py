import numpy as np
from eca import OneDimensionalElementaryCellularAutomata
from matplotlib.pyplot import imshow

def step(x, threshold=0.5):
    return (x > threshold).astype(np.uint8)

def greedy_fft_mask_selection(binary_image, max_iter=None, verbose=False):
    H, W = binary_image.shape
    s = binary_image.astype(np.float32)
    S = np.fft.fft2(s)
    
    # Rank frequencies by magnitude
    freq_indices = np.dstack(np.unravel_index(np.argsort(np.abs(S.ravel()))[::-1], s.shape))[0]
    
    mask = np.zeros_like(S, dtype=np.uint8)
    selected_freqs = []
    
    if max_iter is None:
        max_iter = H * W  # upper bound

    for i in range(min(max_iter, len(freq_indices))):
        u, v = freq_indices[i]
        
        # Add current frequency to mask
        mask_temp = mask.copy()
        mask_temp[u, v] = 1
        
        # Enforce Hermitian symmetry for real-valued signal
        conj_u, conj_v = (-u) % H, (-v) % W
        mask_temp[conj_u, conj_v] = 1
        
        # Apply masked inverse FFT
        S_masked = mask_temp * S
        recon = np.fft.ifft2(S_masked)
        recon_bin = step(np.real(recon))

        if verbose:
            print(f"[{i}] Trying freq ({u}, {v})... match: {np.all(recon_bin == s)}")

        if np.array_equal(recon_bin, s):
            # This frequency makes reconstruction perfect
            mask = mask_temp
            selected_freqs.append((u, v))
            break
        else:
            # Accept this frequency and move on
            mask = mask_temp
            selected_freqs.append((u, v))

    return mask.astype(np.uint8), selected_freqs

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rule = 99

    W, H = 200,200
    configuration = OneDimensionalElementaryCellularAutomata(lattice_width=W)

    for _ in range(H-1):
        configuration.transition(rule_number=rule)
    s = configuration.evolution()

    mask, selected_freqs = greedy_fft_mask_selection(s, verbose=True)

    print(f"Selected {np.sum(mask)} frequencies")

    # Visualize
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1)
    plt.title("Original Binary Image")
    plt.imshow(s, cmap='gray')

    plt.subplot(1,3,2)
    plt.title("Selected Frequencies")
    plt.imshow(mask, cmap='gray')

    plt.subplot(1,3,3)
    S_masked = mask * np.fft.fft2(s)
    s_recon = step(np.real(np.fft.ifft2(S_masked)))
    plt.title("Reconstructed")
    plt.imshow(s_recon, cmap='gray')
    plt.tight_layout()
    plt.show()
