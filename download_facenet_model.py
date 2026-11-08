"""Download the official FaceNet VGGFace2 checkpoint into this project."""
from face_verification import _ensure_local_weights

if __name__ == "__main__":
    path = _ensure_local_weights()
    print(f"FaceNet model ready: {path}")
    print(f"File size: {path.stat().st_size / (1024 * 1024):.1f} MB")
