import torch
from torch import nn
from torch.amp import autocast, GradScaler

model = nn.Linear(10, 1).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

input = torch.randn(5, 10).cuda()
target = torch.randn(5, 1).cuda()

scaler = GradScaler(device="cuda")

with autocast(device_type="cuda"):
    output = model(input)
    loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

print("AMP test passed!")
