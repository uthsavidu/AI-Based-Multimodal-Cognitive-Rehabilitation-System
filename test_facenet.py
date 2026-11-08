"""Sanity-check local FaceNet weights and 512-dimensional output."""
import torch
from face_verification import _load_models, DEVICE

detector, model = _load_models()
with torch.no_grad():
    sample = torch.randn(1, 3, 160, 160, device=DEVICE)
    output = model(sample)
assert tuple(output.shape) == (1, 512), f"Unexpected output shape: {tuple(output.shape)}"
print("FaceNet model loaded successfully.")
print("Embedding output shape:", tuple(output.shape))
print("Device:", DEVICE)
