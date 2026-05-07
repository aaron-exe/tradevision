import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")