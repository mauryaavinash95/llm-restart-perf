import torch
import xformers

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("xFormers version:", xformers.__version__)

try:
    from xformers.ops import memory_efficient_attention
    q = torch.rand(2, 4, 16, 64, device='cuda')
    k = torch.rand(2, 4, 16, 64, device='cuda')
    v = torch.rand(2, 4, 16, 64, device='cuda')
    
    out = memory_efficient_attention(q, k, v)
    print("xFormers memory-efficient attention ran successfully!")
except Exception as e:
    print("Error testing xFormers:", e)
