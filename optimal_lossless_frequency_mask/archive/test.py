import numpy as np

def logit(x, eps=1e-6):
    x = np.clip(x, eps, 1 - eps)  # avoid log(0)
    return np.log(x / (1 - x))

def compute_loss(m, s, lam):
    """
    m: np.ndarray, continuous mask in [0, 1]
    s: np.ndarray, input image in [0, 1]
    lam: float, L1 regularization weight
    """
    Fs = np.fft.fft2(s)
    masked_freq = m * Fs
    recon = np.fft.ifft2(masked_freq).real
    s_logit = logit(s)
    
    l2 = np.sum((recon - s_logit) ** 2)
    l1 = np.sum(np.abs(m))
    
    return l2 + lam * l1



if __name__ == "__main__":
    N = 2
    A = fft2_inverse_matrix(N)
    print(A)