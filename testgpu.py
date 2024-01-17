import torch

# Check if CUDA (GPU support) is available
print(torch.cuda.is_available())

# Check the GPU device count
print(torch.cuda.device_count())

# Print the name of the current GPU
print(torch.cuda.get_device_name(0))