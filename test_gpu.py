import torch

if torch.backends.mps.is_built():
    print("PyTorch is built with MPS!")
else:
    print("MPS not available.")

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print(f"Success! Device found: {mps_device}")
    
    # Run a quick test tensor on the GPU
    x = torch.ones(1, device=mps_device)
    print(f"Tensor created on: {x.device}")
else:
    print("MPS device not found.")