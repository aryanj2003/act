import numpy as np
from mamba_ssm import Mamba2, Mamba
import torch

# Mamba2 requires at least RTX 3090 -- Almost impossible for GPU with previous generation

batch, length, dim = 2, 64, 512
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba2(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape
print("Test passed!")