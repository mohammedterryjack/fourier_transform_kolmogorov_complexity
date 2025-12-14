from search import learn_filter_by_gradient_descent
from generate_image import generate_eca_spacetime
import matplotlib.pyplot as plt
import numpy as np

def learn_filter_varying_iterations(
    rule_number:int,
    image_size:int,
    sigmoid_sharpness:float,
    learning_rate:float,
    sparsity_loss_weight:float,
    quantisation_threshold:float,
    ax=None
) -> None:
    image = generate_eca_spacetime(
        rule_number=rule_number,
        width=image_size,
        height=image_size
    )
    filter_mask, losses = learn_filter_by_gradient_descent(
        binary_image=image,
        quantisation_threshold=quantisation_threshold,
        sigmoid_sharpness=sigmoid_sharpness,
        learning_rate=learning_rate,
        sparsity_loss_weight=sparsity_loss_weight,
        iterations=3000
    )
    total_losses = [loss.total for loss in losses]
    exactness_losses = [loss.exactness for loss in losses]
    sparsity_losses = [loss.sparsity for loss in losses]
    if ax is None:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
    
    ax.plot(total_losses, label="Total Loss")
    ax.plot(exactness_losses, label="Exactness (MSE)")
    ax.plot(sparsity_losses, label="Sparsity (L1)")
    ax.set_xlabel("Iterations (T)")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    

def learn_filter_varying_quantisation_threshold(
    rule_number:int,
    image_size:int,
    sigmoid_sharpness:float,
    learning_rate:float,
    sparsity_loss_weight:float,
    iterations:int,
    ax=None
) -> None:
    image = generate_eca_spacetime(
        rule_number=rule_number,
        width=image_size,
        height=image_size
    )
    variables = []
    total_losses = []
    exactness_losses = []
    sparsity_losses = []
    for θ in np.arange(-1, 1.5, 0.5):
        filter_mask, losses = learn_filter_by_gradient_descent(
            binary_image=image,
            quantisation_threshold=θ,
            sigmoid_sharpness=sigmoid_sharpness,
            learning_rate=learning_rate,
            sparsity_loss_weight=sparsity_loss_weight,
            iterations=iterations
        )
        variables.append(θ)
        total_losses.append(losses[-1].total)
        exactness_losses.append(losses[-1].exactness)
        sparsity_losses.append(losses[-1].sparsity)
    if ax is None:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
    
    ax.plot(variables, total_losses, label="Total Loss")
    ax.plot(variables, exactness_losses, label="Exactness (MSE)")
    ax.plot(variables, sparsity_losses, label="Sparsity (L1)")
    ax.set_xlabel("Quantisation Threshold (θ)")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    


def learn_filter_varying_sigmoid_sharpness(
    rule_number:int,
    image_size:int,
    quantisation_threshold:float,
    learning_rate:float,
    sparsity_loss_weight:float,
    iterations:int,
    ax=None
) -> None:
    image = generate_eca_spacetime(
        rule_number=rule_number,
        width=image_size,
        height=image_size
    )
    variables = []
    total_losses = []
    exactness_losses = []
    sparsity_losses = []
    for β in (1e1, 1e2, 1e3):
        filter_mask, losses = learn_filter_by_gradient_descent(
            binary_image=image,
            quantisation_threshold=quantisation_threshold,
            sigmoid_sharpness=β,
            learning_rate=learning_rate,
            sparsity_loss_weight=sparsity_loss_weight,
            iterations=iterations
        )
        variables.append(β)
        total_losses.append(losses[-1].total)
        exactness_losses.append(losses[-1].exactness)
        sparsity_losses.append(losses[-1].sparsity)
    if ax is None:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
    
    ax.plot(variables, total_losses, label="Total Loss")
    ax.plot(variables, exactness_losses, label="Exactness (MSE)")
    ax.plot(variables, sparsity_losses, label="Sparsity (L1)")
    ax.set_xlabel("Sigmoid Sharpness (β)")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

def learn_filter_varying_learning_rate(
    rule_number:int,
    image_size:int,
    sigmoid_sharpness:float,
    quantisation_threshold:float,
    sparsity_loss_weight:float,
    iterations:int,
    ax=None
) -> None:
    image = generate_eca_spacetime(
        rule_number=rule_number,
        width=image_size,
        height=image_size
    )
    variables = []
    total_losses = []
    exactness_losses = []
    sparsity_losses = []
    for learning_rate in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2):
        filter_mask, losses = learn_filter_by_gradient_descent(
            binary_image=image,
            quantisation_threshold=quantisation_threshold,
            sigmoid_sharpness=sigmoid_sharpness,
            learning_rate=learning_rate,
            sparsity_loss_weight=sparsity_loss_weight,
            iterations=iterations
        )
        variables.append(learning_rate)
        total_losses.append(losses[-1].total)
        exactness_losses.append(losses[-1].exactness)
        sparsity_losses.append(losses[-1].sparsity)
    if ax is None:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
    
    ax.semilogx(variables, total_losses, label="Total Loss")
    ax.semilogx(variables, exactness_losses, label="Exactness (MSE)")
    ax.semilogx(variables, sparsity_losses, label="Sparsity (L1)")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    



def learn_filter_varying_sparsity_loss_weight(
    rule_number:int,
    image_size:int,
    sigmoid_sharpness:float,
    quantisation_threshold:float,
    learning_rate:float,
    iterations:int,
    ax=None
) -> None:
    image = generate_eca_spacetime(
        rule_number=rule_number,
        width=image_size,
        height=image_size
    )
    variables = []
    total_losses = []
    exactness_losses = []
    sparsity_losses = []
    for λ in (1e-2, 1e-1, 0, 1e1, 1e2, 1e3):
        filter_mask, losses = learn_filter_by_gradient_descent(
            binary_image=image,
            quantisation_threshold=quantisation_threshold,
            sigmoid_sharpness=sigmoid_sharpness,
            learning_rate=learning_rate,
            sparsity_loss_weight=λ,
            iterations=iterations
        )
        variables.append(λ)
        total_losses.append(losses[-1].total)
        exactness_losses.append(losses[-1].exactness)
        sparsity_losses.append(losses[-1].sparsity)
    if ax is None:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
    
    ax.semilogx(variables, total_losses, label="Total Loss")
    ax.semilogx(variables, exactness_losses, label="Exactness (MSE)")
    ax.semilogx(variables, sparsity_losses, label="Sparsity (L1)")
    ax.set_xlabel("Sparsity Loss Weight (λ)")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    

def learn_filter_varying_image_size(
    rule_number:int,
    sigmoid_sharpness:float,
    quantisation_threshold:float,
    learning_rate:float,
    iterations:int,
    sparsity_loss_weight:float,
    ax=None
) -> None:
    variables = []
    total_losses = []
    exactness_losses = []
    sparsity_losses = []
    for image_size in range(100,700, 100):
        image = generate_eca_spacetime(
            rule_number=rule_number,
            width=image_size,
            height=image_size
        )

        filter_mask, losses = learn_filter_by_gradient_descent(
            binary_image=image,
            quantisation_threshold=quantisation_threshold,
            sigmoid_sharpness=sigmoid_sharpness,
            learning_rate=learning_rate,
            sparsity_loss_weight=sparsity_loss_weight,
            iterations=iterations
        )
        variables.append(image_size)
        total_losses.append(losses[-1].total)
        exactness_losses.append(losses[-1].exactness)
        sparsity_losses.append(losses[-1].sparsity)
    if ax is None:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
    
    ax.plot(variables, total_losses, label="Total Loss")
    ax.plot(variables, exactness_losses, label="Exactness (MSE)")
    ax.plot(variables, sparsity_losses, label="Sparsity (L1)")
    ax.set_xlabel("Image Width or Height")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    

def plot_all_studies(
    rule_number:int,
    image_size:int,
    sigmoid_sharpness:int,
    learning_rate:float,
    sparsity_loss_weight:float,
    iterations:int,
    quantisation_threshold:float
) -> None:
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Ablation Studies for Rule {rule_number}', fontsize=16, fontweight='bold')
    
    # image = generate_eca_spacetime(
    #     rule_number=rule_number,
    #     width=image_size,
    #     height=image_size
    # )
    # axes[0, 0].imshow(image, cmap='gray')
    # axes[0, 0].axis('off')

    # 1. Varying iterations
    learn_filter_varying_iterations(
        rule_number=rule_number,
        image_size=image_size,
        sigmoid_sharpness=sigmoid_sharpness,
        learning_rate=learning_rate,
        sparsity_loss_weight=sparsity_loss_weight,
        quantisation_threshold=quantisation_threshold,
        ax=axes[0, 0]
    )
    
    # 2. Varying quantisation threshold
    learn_filter_varying_quantisation_threshold(
        rule_number=rule_number,
        image_size=image_size,
        sigmoid_sharpness=sigmoid_sharpness,
        learning_rate=learning_rate,
        sparsity_loss_weight=sparsity_loss_weight,
        iterations=iterations,
        ax=axes[0, 1]
    )
    
    # 3. Varying learning rate
    learn_filter_varying_learning_rate(
        rule_number=rule_number,
        image_size=image_size,
        sigmoid_sharpness=sigmoid_sharpness,
        quantisation_threshold=quantisation_threshold,
        sparsity_loss_weight=sparsity_loss_weight,
        iterations=iterations,
        ax=axes[0, 2]
    )
    
    # 4. Varying sparsity loss weight
    learn_filter_varying_sparsity_loss_weight(
        rule_number=rule_number,
        image_size=image_size,
        sigmoid_sharpness=sigmoid_sharpness,
        quantisation_threshold=quantisation_threshold,
        learning_rate=learning_rate,
        iterations=iterations,
        ax=axes[1, 0]
    )

    learn_filter_varying_sigmoid_sharpness(
        rule_number=rule_number,
        image_size=image_size,
        sparsity_loss_weight=sparsity_loss_weight,
        quantisation_threshold=quantisation_threshold,
        learning_rate=learning_rate,
        iterations=iterations,
        ax=axes[1, 1]
    )
    
    # 5. Varying image size
    learn_filter_varying_image_size(
        rule_number=rule_number,
        sigmoid_sharpness=sigmoid_sharpness,
        quantisation_threshold=quantisation_threshold,
        learning_rate=learning_rate,
        iterations=iterations,
        sparsity_loss_weight=sparsity_loss_weight,
        ax=axes[1, 2]
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_all_studies(
        rule_number=110,
        image_size=200,
        sigmoid_sharpness=1e2,
        learning_rate=1e-2,
        sparsity_loss_weight=1e1,
        iterations=1500,
        quantisation_threshold=0.5
    )
    