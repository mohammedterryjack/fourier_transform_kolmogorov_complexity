import matplotlib.pyplot as plt
import numpy as np
import pydantic 


class TrainingStats(pydantic.BaseModel):
    iteration:int
    exactness:float 
    sparsity:float
    total:float
    quantisation_threshold:float

def plot_loss(losses:list[TrainingStats]) -> None:
    iterations = [stat.iteration for stat in losses]
    exactness = [stat.exactness for stat in losses]
    sparsity = [stat.sparsity for stat in losses]
    total = [stat.total for stat in losses]

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, exactness, label="Exactness (MSE)")
    plt.plot(iterations, sparsity, label="Sparsity (L1)")
    plt.plot(iterations, total, label="Total Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
def plot_filter_and_reconstruction(
    image:np.ndarray,
    filter_mask:np.ndarray,
    reconstruction:np.ndarray,
) -> None:
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Learnt Filter")
    plt.imshow(filter_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Reconstruction of Image")
    plt.imshow(reconstruction, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Differences")
    plt.imshow(np.abs(reconstruction - image), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
