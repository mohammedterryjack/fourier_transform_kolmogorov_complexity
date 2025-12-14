import numpy
import torch

from visualise import TrainingStats


def ft2d_compression(s:numpy.ndarray, m:numpy.ndarray) -> numpy.ndarray:
    s = torch.tensor(s, dtype=torch.int16)
    m = torch.tensor(m, dtype=torch.bool)
    return m*torch.fft.fft2(s) 
    
def ft2d_decompression(z:numpy.ndarray, θ:float) -> numpy.ndarray:
    s_hat = torch.fft.ifft2(z).real
    return (s_hat > θ).float()


def elementwise_sigmoid(
    s:numpy.ndarray, 
    θ:torch.nn.Parameter, 
    β:float,
) -> numpy.ndarray:
    return torch.sigmoid(β * (s - θ))

def safe_elementwise_inverse_sigmoid(
    s:numpy.ndarray, 
    θ:float, 
    β:float,
    ε:float = 1e-4
) -> numpy.ndarray:
    s = s.clamp(min=ε, max=1 - ε)
    return θ + (1.0 / β) * torch.log(s / (1 - s))

def learn_filter_by_gradient_descent(
    binary_image:numpy.ndarray,
    sigmoid_sharpness:float,
    learning_rate:float,
    sparsity_loss_weight:float,
    iterations:int,
    quantisation_threshold:float,
) -> tuple[numpy.ndarray,list[TrainingStats]]:
    losses = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    original_image = torch.tensor(binary_image, dtype=torch.int16, device=device)
    with torch.no_grad():
        representation_of_image = torch.fft.fft2(original_image) 
        acceptable_lossy_image = safe_elementwise_inverse_sigmoid( 
            s=original_image,
            θ=quantisation_threshold,
            β=sigmoid_sharpness
        )
    U,V = representation_of_image.shape
    filter_weights = torch.randn(U, V, device=device, requires_grad=True) 

    optimiser = torch.optim.Adam([filter_weights], lr=learning_rate)

    for i in range(iterations):
        optimiser.zero_grad()

        soft_filter_mask = (torch.tanh(filter_weights) + 1) / 2.0
        compressed_representation = soft_filter_mask * representation_of_image
        lossy_image = torch.fft.ifft2(compressed_representation).real

        l1_sparsity = torch.mean(soft_filter_mask)
        l2_exactness = torch.mean(
            (lossy_image - acceptable_lossy_image) ** 2
        )
        loss = l2_exactness + sparsity_loss_weight * l1_sparsity

        loss.backward()
        optimiser.step()

        losses.append(
            TrainingStats(
                iteration=i,
                total=loss.item(),
                exactness=l2_exactness.item(),
                sparsity=l1_sparsity.item(),
                quantisation_threshold=quantisation_threshold
            )
        )

    with torch.no_grad():
        hard_filter_mask = (torch.tanh(soft_filter_mask) > quantisation_threshold).float()
        print(f"mask size = {hard_filter_mask.sum().item()}")
    return hard_filter_mask.cpu().numpy(), losses


def learn_filter_and_quantisation_threshold_by_gradient_descent(
    binary_image:numpy.ndarray,
    sigmoid_sharpness:float,
    learning_rate:float,
    sparsity_loss_weight:float,
    iterations:int,
) -> tuple[numpy.ndarray,list[TrainingStats]]:
    losses = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    original_image = torch.tensor(binary_image, dtype=torch.int16, device=device)
    with torch.no_grad():
        representation_of_image = torch.fft.fft2(original_image) 
    U,V = representation_of_image.shape
    filter_weights = torch.randn(U, V, device=device, requires_grad=True) 
    quantisation_threshold = torch.nn.Parameter(torch.tensor(0.5, device=device, dtype=torch.float32))

    optimiser = torch.optim.Adam([filter_weights, quantisation_threshold], lr=learning_rate)

    for i in range(iterations):
        optimiser.zero_grad()

        soft_filter_mask = (torch.tanh(filter_weights) + 1) / 2.0
        compressed_representation = soft_filter_mask * representation_of_image
        lossy_image = torch.fft.ifft2(compressed_representation).real
        denoised_image = elementwise_sigmoid(
            s=lossy_image, 
            θ=quantisation_threshold, 
            β=sigmoid_sharpness,
        )

        l1_sparsity = torch.mean(soft_filter_mask)
        l2_exactness = torch.mean(
            (denoised_image - original_image) ** 2
        )
        loss = l2_exactness + sparsity_loss_weight * l1_sparsity

        loss.backward()
        optimiser.step()

        losses.append(
            TrainingStats(
                iteration=i,
                total=loss.item(),
                exactness=l2_exactness.item(),
                sparsity=l1_sparsity.item(),
                quantisation_threshold=quantisation_threshold.item()
            )
        )

    with torch.no_grad():
        hard_filter_mask = (torch.tanh(soft_filter_mask) > quantisation_threshold.item()).float()
        print(f"mask size = {hard_filter_mask.sum().item()}")
    return hard_filter_mask.cpu().numpy(),quantisation_threshold.item(), losses



if __name__ == "__main__":
    from generate_image import generate_eca_spacetime
    from visualise import plot_filter_and_reconstruction, plot_loss

    image = generate_eca_spacetime(
        rule_number=30,
        width=500,
        height=500
    )
    # m,losses = learn_filter_by_gradient_descent(
    #     binary_image=image,
    #     quantisation_threshold=0.5,
    #     sigmoid_sharpness=10,
    #     learning_rate=1e-2,
    #     sparsity_loss_weight=1e-2,
    #     iterations=1000
    # )
    # θ=0.5
    m,θ,losses = learn_filter_and_quantisation_threshold_by_gradient_descent(
        binary_image=image,
        sigmoid_sharpness=10,
        learning_rate=1e-2,
        sparsity_loss_weight=1e-2,
        iterations=1000
    )
    plot_loss(losses)

    z = ft2d_compression(
        s=image,
        m=m
    )
    s_hat = ft2d_decompression(
        z=z,
        θ=θ
    )
    plot_filter_and_reconstruction(
        image=image,
        filter_mask=m,
        reconstruction=s_hat
    )
    
