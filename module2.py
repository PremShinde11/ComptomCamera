import torch

print("XPU available:", torch.xpu.is_available())

device = torch.device("xpu")

x = torch.randn(2000, 2000).to(device)
y = x @ x

print("Device:", y.device)

