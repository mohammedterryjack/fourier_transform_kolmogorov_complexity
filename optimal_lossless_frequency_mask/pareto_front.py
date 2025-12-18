from search import (
    learn_filter_by_gradient_descent,
    ft2d_compression, 
    ft2d_decompression
)
from generate_image import generate_eca_spacetime

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from numpy import ndarray, abs, vstack 

def plot_pareto_front(image:ndarray, values:list[float], key:str, **kwargs) -> None:

    results = []
    reconstructions = []

    for value in values:
       params = dict(kwargs)
       params[key] = value  
       mask, stats = learn_filter_by_gradient_descent(
            binary_image=image,
            **params 
       )

       results.append({
           key: value,
           "final_sparsity": stats[-1].sparsity,
           "final_exactness": stats[-1].exactness,
       })
       z = ft2d_compression(s=image, m=mask)
       s_hat = ft2d_decompression(z=z, θ=params['quantisation_threshold'] )
       recon = s_hat.detach().cpu().numpy().astype(float)
       diff = abs(image - recon) 
       combined = vstack([recon, diff])
       reconstructions.append(combined)

    sparsities = [result['final_sparsity'] for result in results]
    exactnesses = [result['final_exactness'] for result in results] 

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    ax.plot(exactnesses, sparsities, marker="o")
    for x, y, val in zip(exactnesses, sparsities, values, strict=True):
        ax.annotate(
            f"{key}={val}",
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
            (x, y),
            xybox=(-20, 20),   # offset in points
            xycoords="data",
            boxcoords="offset points",
            frameon=True
        )

        ax.add_artist(ab)


    plt.ylabel("Sparsity Loss (Mask Density)")
    plt.xlabel("Exactness Loss (Reconstruction Error)")
    plt.title("Sparsity vs Exactness of Reconstruction")
    plt.grid(True)
    plt.savefig(f"{key}.png")


if __name__ == "__main__":
    evolution = generate_eca_spacetime(
       ic=111,
       rule_number=110,
       width=20,
       height=20
    )

    plot_pareto_front(
       image=evolution,
       values=[int(5e2), int(1e3), int(5e3), int(1e4), int(5e4)],
       key="iterations",
       sigmoid_sharpness=10,
       learning_rate=1e-2,
       sparsity_loss_weight=1,
       #iterations=1000,
       quantisation_threshold=0.5
    )
