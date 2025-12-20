from search import (
    learn_filter_by_gradient_descent,
    ft2d_compression, 
    ft2d_decompression
)
from generate_image import generate_eca_spacetime

import csv
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from numpy import ndarray, abs, vstack, hstack 

def plot_pareto_front(values:list[float], key:str, **kwargs) -> None:

    results = []
    reconstructions = []
    for value in values:
       params = dict(kwargs)
       params[key] = value

       image = generate_eca_spacetime(
          ic=params['ic'],
          rule_number=params['rule'],
          width=params['width'],
          height=params['height']
       )
       mask, stats = learn_filter_by_gradient_descent(
           binary_image=image,
           sigmoid_sharpness=params['sigmoid_sharpness'],
           learning_rate=params['learning_rate'],
           sparsity_loss_weight=params['sparsity_loss_weight'],
           iterations=params['iterations'],
           quantisation_threshold=params['quantisation_threshold']
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
       combined = vstack([hstack([image, mask]), hstack([recon, diff])])
       reconstructions.append(combined)

    sparsities = [result['final_sparsity'] for result in results]
    exactnesses = [result['final_exactness'] for result in results] 

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    ax.plot(exactnesses, sparsities, marker="o")
    for x, y, val in zip(exactnesses, sparsities, values, strict=True):
        ax.annotate(
            f"{val}",
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            fontsize=8
        )

    for x, y, img in zip(exactnesses, sparsities, reconstructions, strict=True):
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

    rule = params['rule']
    plt.ylabel("Sparsity Loss (Mask Density)")
    plt.xlabel("Exactness Loss (Reconstruction Error)")
    plt.title(f"Sparsity vs Exactness ({key})")
    plt.grid(True)
    plt.savefig(f"data/{key}_{rule}.png")
    csv_path = f"data/{key}_{rule}.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)


