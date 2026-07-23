import torch
print("torch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))  # esperado (12, 0)
A = torch.randn(4000, 4000, device="cuda")
print("matmul ok:", (A @ A).shape)