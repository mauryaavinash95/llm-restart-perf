import torch
import deepspeed
from deepspeed.ops.scaled_softmax import FusedScaleMaskSoftmax

# Check CUDA device
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

# Create a small test tensor
batch_size = 2
seq_len = 4
num_heads = 2
head_dim = 3

# Random input tensor
x = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
mask = torch.ones(batch_size, 1, seq_len, seq_len, device='cuda', dtype=torch.bool)

# Initialize fused softmax
fused_softmax = FusedScaleMaskSoftmax(attention_mask_type='causal', scale=1.0)

# Forward pass
try:
    out = fused_softmax(x, mask)
    print("Fused softmax forward pass succeeded, output shape:", out.shape)
except Exception as e:
    print("Error during fused softmax forward:", e)
