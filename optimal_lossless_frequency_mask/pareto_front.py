from tools.utils import load_evolutions_from_parquet
from tools.utils import learn_filter_by_pareto_optimisation
from tools.utils import ft2d_compression, ft2d_decompression
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from numpy import vstack 

images = load_evolutions_from_parquet('data/toy/chaotic.parquet')
image = list(images.values())[0]
epsilons = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

results = []
reconstructions = []

for eps in epsilons:
    mask, stats = learn_filter_by_pareto_optimisation(
        binary_image=image,
        sigmoid_sharpness=20.0,
        learning_rate=1e-2,
        iterations=2000,
        quantisation_threshold=0.5,
        epsilon_exactness=eps,
	lagrangian_lr=0.01,
	penalty_weight=100.0
    )

    results.append({
        "epsilon": eps,
        "mask_size": mask.sum(),
        "final_exactness": stats[-1].exactness,
    })
    z = ft2d_compression(s=image, m=mask)
    s_hat = ft2d_decompression(z=z, θ=0.5)
    recon = s_hat.detach().cpu().numpy().astype(float)
    diff = image - recon 
    combined = vstack([recon, diff])
    reconstructions.append(combined)

sparsities = [result['mask_size'] for result in results]
exactnesses = [result['final_exactness'] for result in results] 

plt.figure(figsize=(7, 5))
ax = plt.gca()


ax.plot(exactnesses, sparsities, marker="o")
for x, y, eps in zip(exactnesses, sparsities, epsilons):
    ax.annotate(
        f"ε={eps}",
        (x, y),
        textcoords="offset points",
        xytext=(5, 5),
        ha="left",
        fontsize=8
    )

for x, y, img in zip(exactnesses, sparsities, reconstructions):
    img_disp = img.astype(float)
    img_disp -= img_disp.min()
    img_disp /= (img_disp.max() + 1e-8)

    imagebox = OffsetImage(
        img_disp,
        zoom=1.0,        # controls thumbnail size
        cmap="gray"
    )

    ab = AnnotationBbox(
        imagebox,
        (x+0.05, y+0.05),
        xybox=(20, 20),   # offset in points
        xycoords="data",
        boxcoords="offset points",
        frameon=True
    )

    ax.add_artist(ab)


plt.ylabel("Number of active Fourier coefficients (|m|₀)")
plt.xlabel("Reconstruction error")
plt.title("Pareto Front: Sparsity vs Reconstruction Fidelity")
plt.grid(True)
plt.savefig("bla.png")
