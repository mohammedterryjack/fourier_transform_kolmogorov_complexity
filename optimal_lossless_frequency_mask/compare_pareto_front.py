import csv
import matplotlib.pyplot as plt
from numpy import abs

from search import (
    learn_filter_by_gradient_descent,
    ft2d_compression,
    ft2d_decompression
)
from generate_image import generate_eca_spacetime


def run_pareto_for_rule(values, key, *, rule, **params):
    results = []

    for value in values:
        run_params = dict(params)
        run_params[key] = value

        image = generate_eca_spacetime(
            ic=run_params["ic"],
            rule_number=rule,
            width=run_params["width"],
            height=run_params["height"]
        )

        mask, stats = learn_filter_by_gradient_descent(
            binary_image=image,
            sigmoid_sharpness=run_params["sigmoid_sharpness"],
            learning_rate=run_params["learning_rate"],
            sparsity_loss_weight=run_params["sparsity_loss_weight"],
            iterations=run_params["iterations"],
            quantisation_threshold=run_params["quantisation_threshold"]
        )

        results.append({
            key: value,
            "final_sparsity": stats[-1].sparsity,
            "final_exactness": stats[-1].exactness,
        })

    exactnesses = [r["final_exactness"] for r in results]
    sparsities = [r["final_sparsity"] for r in results]

    return exactnesses, sparsities, results


def plot_pareto_front_two_rules(
    *,
    rules: tuple[int, int],
    values: list[float],
    key: str,
    **params
) -> None:

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    styles = [
        dict(marker="o", linestyle="-", label=f"Rule {rules[0]}"),
        dict(marker="s", linestyle="--", label=f"Rule {rules[1]}")
    ]

    for rule, style in zip(rules, styles, strict=True):
        exactnesses, sparsities, results = run_pareto_for_rule(
            values=values,
            key=key,
            rule=rule,
            **params
        )

        # Plot curve
        ax.plot(
            exactnesses,
            sparsities,
            **style,
            zorder=2
        )

        for x, y, val in zip(exactnesses, sparsities, values, strict=True):
            ax.annotate(
                f"{val}",
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                zorder=5
            )

    ax.set_xlabel("Exactness Loss (Reconstruction Error)")
    ax.set_ylabel("Sparsity Loss (Mask Density)")
    ax.set_title(f"Sparsity vs Exactness ({key})")
    ax.grid(True)
    ax.legend()

    plt.savefig(f"data/{key}_rule_comparison.png")
    plt.close()

