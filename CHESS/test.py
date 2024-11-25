import torch

batch_size = 773
x = torch.randn(batch_size, 19, 8, 8).cuda()

print(f"Estimated memory: {x.element_size() * x.nelement() / 1024 / 1024:.2f} MB")
