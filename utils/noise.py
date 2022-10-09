import torch

def create_noise(noise_size:int, batch_size:int) -> torch.Tensor:
    # create standard N(0.0, 1.0) noise
    return torch.randn(batch_size, noise_size)
