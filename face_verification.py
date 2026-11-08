
"""
Face verification for AI Cognitive Rehabilitation System.

Detector: MTCNN
Face embedding: InceptionResnetV1 pretrained on VGGFace2
Embedding size: 512 dimensions
"""

from __future__ import annotations

import io
import json
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


MODEL_NAME = "FaceNet-InceptionResnetV1-VGGFace2"
# Multi-pose enrollment is compared against every saved sample; calibrate with local genuine/impostor captures.
FACE_MATCH_THRESHOLD = 0.34
EMBEDDING_DIMENSION = 512
EMBEDDING_DTYPE = np.float32

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = (
    PROJECT_ROOT / "models" / "facenet" / "facenet_vggface2.pt"
)
MODEL_URL = (
    "https://github.com/timesler/facenet-pytorch/releases/download/"
    "v2.2.9/20180402-114759-vggface2.pt"
)

_detector = None
_facenet = None


class FaceEnrollmentError(ValueError):
    """An image could not be used for face enrollment."""


class FaceVerificationError(ValueError):
    """An image or saved face embedding is invalid."""


def _ensure_local_weights() -> Path:
    """Download pretrained weights into the project folder if necessary."""
    LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if (
        LOCAL_MODEL_PATH.is_file()
        and LOCAL_MODEL_PATH.stat().st_size > 10_000_000
    ):
        return LOCAL_MODEL_PATH

    print(f"[FaceNet] Downloading pretrained weights to {LOCAL_MODEL_PATH}")

    try:
        downloaded = torch.hub.load_state_dict_from_url(
            MODEL_URL,
            model_dir=str(LOCAL_MODEL_PATH.parent),
            map_location="cpu",
            progress=True,
            check_hash=False,
            file_name=LOCAL_MODEL_PATH.name,
        )

        if not isinstance(downloaded, dict) or not downloaded:
            raise RuntimeError("The downloaded checkpoint is empty.")

        if not LOCAL_MODEL_PATH.is_file():
            torch.save(downloaded, LOCAL_MODEL_PATH)

    except Exception as exc:
        raise RuntimeError(
            "Could not obtain FaceNet VGGFace2 weights. Check your internet "
            f"connection and model file: {LOCAL_MODEL_PATH}. "
            f"Details: {type(exc).__name__}: {exc}"
        ) from exc

    if (
        not LOCAL_MODEL_PATH.is_file()
        or LOCAL_MODEL_PATH.stat().st_size < 10_000_000
    ):
        raise RuntimeError(
            f"FaceNet checkpoint is missing or incomplete: {LOCAL_MODEL_PATH}"
        )

    return LOCAL_MODEL_PATH


def _load_models():
    """Load MTCNN and FaceNet once per Python process."""
    global _detector, _facenet

    if _detector is None:
        _detector = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=True,
            device=DEVICE,
        )

    if _facenet is None:
        model_path = _ensure_local_weights()

        # The pretrained VGGFace2 checkpoint includes an 8631-class
        # classifier, so its architecture must match while loading weights.
        model = InceptionResnetV1(
            pretrained=None,
            classify=True,
            num_classes=8631,
        )

        try:
            state_dict = torch.load(
                str(model_path),
                map_location=DEVICE,
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(
                str(model_path),
                map_location=DEVICE,
            )

        model.load_state_dict(state_dict, strict=True)

        # Disable classification only AFTER loading the complete checkpoint.
        # Face embeddings are now 512-dimensional.
        model.classify = False
        _facenet = model.eval().to(DEVICE)

        print(f"[FaceNet] Loaded weights: {model_path}")
        print(f"[FaceNet] Device: {DEVICE}")
        print("[FaceNet] Embedding dimension: 512")

    return _detector, _facenet


def _decode_image(image_bytes: bytes) -> Image.Image:
    """Decode uploaded image bytes into RGB."""
    if not image_bytes:
        raise FaceEnrollmentError("No image was received.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            image = source.convert("RGB")
    except Exception as exc:
        raise FaceEnrollmentError(
            f"Could not decode captured image: {type(exc).__name__}: {exc}"
        ) from exc

    if image.width < 80 or image.height < 80:
        raise FaceEnrollmentError(
            "The image is too small. Move closer to the camera."
        )

    return image


def _detect_and_align_single_face(image: Image.Image) -> torch.Tensor:
    """Detect exactly one sufficiently clear face and align it."""
    detector, _ = _load_models()

    boxes, probabilities = detector.detect(image)

    if boxes is None or len(boxes) == 0:
        raise FaceEnrollmentError(
            "No face detected. Face the camera and improve lighting."
        )

    valid_indices = []

    for index, box in enumerate(boxes):
        if box is None:
            continue

        try:
            box = np.asarray(box, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            continue

        if box.size < 4 or not np.isfinite(box[:4]).all():
            continue

        box = box[:4]

        confidence = (
            float(probabilities[index])
            if probabilities is not None
            and probabilities[index] is not None
            else 0.0
        )

        if confidence >= 0.80:
            valid_indices.append(index)

    if not valid_indices:
        raise FaceEnrollmentError(
            "A clear face was not detected. Face the camera and try again."
        )

    if len(valid_indices) > 1:
        raise FaceEnrollmentError(
            "Multiple faces detected. Keep only one person in the frame."
        )

    selected_box = np.asarray(
        [boxes[valid_indices[0]]],
        dtype=np.float32,
    )

    aligned = detector.extract(
        image,
        selected_box,
        save_path=None,
    )

    if aligned is None:
        raise FaceEnrollmentError(
            "The face could not be aligned. Capture another image."
        )

    if aligned.ndim == 3:
        aligned = aligned.unsqueeze(0)

    if aligned.ndim != 4 or aligned.shape[0] != 1:
        raise FaceEnrollmentError(
            "Exactly one clear face is required."
        )

    if tuple(aligned.shape[1:]) != (3, 160, 160):
        raise FaceEnrollmentError(
            "The aligned face has an unexpected image size."
        )

    return aligned.to(DEVICE)


def _embedding_from_image(image_bytes: bytes) -> np.ndarray:
    """Create a normalized 512-dimensional FaceNet embedding."""
    try:
        image = _decode_image(image_bytes)
        aligned_face = _detect_and_align_single_face(image)
        _, model = _load_models()

        with torch.no_grad():
            output = model(aligned_face)

        vector = output[0].detach().cpu().numpy().astype(np.float32)

        if vector.shape != (EMBEDDING_DIMENSION,):
            raise FaceEnrollmentError(
                f"Unexpected FaceNet embedding shape: {vector.shape}"
            )

        if not np.all(np.isfinite(vector)):
            raise FaceEnrollmentError(
                "FaceNet returned invalid numerical values."
            )

        norm = float(np.linalg.norm(vector))

        if norm <= 1e-12:
            raise FaceEnrollmentError(
                "The face embedding is empty."
            )

        return (vector / norm).astype(np.float32)

    except FaceEnrollmentError:
        raise

    except Exception as exc:
        print("\n========== ACTUAL FACENET ERROR ==========")
        traceback.print_exc()
        print("==========================================\n")

        raise FaceEnrollmentError(
            f"FaceNet processing error: {type(exc).__name__}: {exc}"
        ) from exc


def extract_registered_face(image_bytes: bytes) -> bytes:
    """Return a 512-float embedding as bytes for the existing SQLite schema."""
    vector = _embedding_from_image(image_bytes)
    return vector.tobytes()


def _decode_stored_embedding(value) -> np.ndarray:
    """Decode a stored FaceNet embedding from SQLite BLOB or JSON."""
    if value is None:
        raise FaceVerificationError(
            "The registered face embedding is missing."
        )

    if isinstance(value, memoryview):
        value = value.tobytes()

    if isinstance(value, bytearray):
        value = bytes(value)

    if isinstance(value, bytes):
        expected_size = (
            EMBEDDING_DIMENSION * np.dtype(EMBEDDING_DTYPE).itemsize
        )

        if len(value) != expected_size:
            raise FaceVerificationError(
                "This account contains an old face sample instead of a "
                "FaceNet embedding. Please enroll your face again."
            )

        vector = np.frombuffer(
            value,
            dtype=EMBEDDING_DTYPE,
        ).copy()

    elif isinstance(value, str):
        try:
            vector = np.asarray(json.loads(value), dtype=np.float32)
        except Exception as exc:
            raise FaceVerificationError(
                "The saved face embedding is invalid JSON."
            ) from exc

    else:
        raise FaceVerificationError(
            "The saved face embedding uses an unsupported format."
        )

    if vector.shape != (EMBEDDING_DIMENSION,):
        raise FaceVerificationError(
            "The saved face embedding must contain 512 values."
        )

    if not np.all(np.isfinite(vector)):
        raise FaceVerificationError(
            "The saved face embedding contains invalid values."
        )

    norm = float(np.linalg.norm(vector))

    if norm <= 1e-12:
        raise FaceVerificationError(
            "The saved face embedding is empty."
        )

    return (vector / norm).astype(np.float32)


def cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate cosine distance between normalized face embeddings."""
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)

    if first.shape != (EMBEDDING_DIMENSION,):
        raise FaceVerificationError("Invalid candidate embedding.")

    if second.shape != (EMBEDDING_DIMENSION,):
        raise FaceVerificationError("Invalid registered embedding.")

    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))

    if denominator <= 1e-12:
        return 1.0

    similarity = float(np.dot(first, second) / denominator)
    similarity = float(np.clip(similarity, -1.0, 1.0))

    return 1.0 - similarity



def verify_monitoring_faces(
    image_bytes: bytes,
    stored_face_bytes_list: Iterable[bytes],
) -> tuple[list[dict], tuple[int, int]]:
    """Verify each detected face independently against one account's enrollment.

    Returns per-face boxes in source-image pixel coordinates, match decisions,
    and cosine distances. Face detection alone never marks a face as matched.
    """
    try:
        image = _decode_image(image_bytes)
    except FaceEnrollmentError as exc:
        raise FaceVerificationError(str(exc)) from exc

    detector, model = _load_models()
    boxes, probabilities = detector.detect(image)
    if boxes is None or len(boxes) == 0:
        raise FaceVerificationError("No face detected in the monitoring frame.")

    selected_boxes = []
    for index, box in enumerate(boxes):
        if box is None:
            continue
        candidate_box = np.asarray(box, dtype=np.float32).reshape(-1)
        if candidate_box.size < 4 or not np.isfinite(candidate_box[:4]).all():
            continue
        confidence = float(probabilities[index]) if probabilities is not None and probabilities[index] is not None else 0.0
        if confidence >= 0.80:
            selected_boxes.append((candidate_box[:4], confidence))
    if not selected_boxes:
        raise FaceVerificationError("A clear face was not detected in the monitoring frame.")

    if stored_face_bytes_list is None:
        raise FaceVerificationError("No registered face embeddings were supplied.")
    if isinstance(stored_face_bytes_list, (bytes, bytearray, memoryview, str)):
        stored_face_bytes_list = [stored_face_bytes_list]
    registered = [_decode_stored_embedding(value) for value in stored_face_bytes_list]
    if not registered:
        raise FaceVerificationError("No registered FaceNet embeddings were found.")

    crop_boxes = np.asarray([item[0] for item in selected_boxes], dtype=np.float32)
    aligned = detector.extract(image, crop_boxes, save_path=None)
    if aligned is None:
        raise FaceVerificationError("The detected faces could not be aligned.")
    if aligned.ndim == 3:
        aligned = aligned.unsqueeze(0)
    if aligned.ndim != 4 or aligned.shape[0] != len(selected_boxes):
        raise FaceVerificationError("Face alignment returned an unexpected batch shape.")

    with torch.no_grad():
        embeddings = model(aligned.to(DEVICE)).detach().cpu().numpy().astype(np.float32)

    results = []
    for box, confidence, vector in zip(crop_boxes, [x[1] for x in selected_boxes], embeddings):
        if vector.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(vector).all():
            raise FaceVerificationError("FaceNet returned an invalid face embedding.")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise FaceVerificationError("FaceNet returned an empty face embedding.")
        vector = vector / norm
        distance = min(cosine_distance(vector, ref) for ref in registered)
        results.append({
            "box": [round(float(v), 2) for v in box],
            "confidence": round(float(confidence), 4),
            "match": bool(distance <= FACE_MATCH_THRESHOLD),
            "distance": round(float(distance), 4),
        })
    return results, (image.width, image.height)


def verify_login_face(
    image_bytes: bytes,
    stored_face_bytes_list: Iterable[bytes],
) -> tuple[bool, float]:
    """Compare the captured face against a user's registered embeddings."""
    try:
        candidate = _embedding_from_image(image_bytes)
    except FaceEnrollmentError as exc:
        raise FaceVerificationError(str(exc)) from exc

    if stored_face_bytes_list is None:
        raise FaceVerificationError(
            "No registered face embeddings were supplied."
        )

    if isinstance(
        stored_face_bytes_list,
        (bytes, bytearray, memoryview, str),
    ):
        stored_face_bytes_list = [stored_face_bytes_list]

    distances = []

    for stored_value in stored_face_bytes_list:
        registered = _decode_stored_embedding(stored_value)
        distances.append(cosine_distance(candidate, registered))

    if not distances:
        raise FaceVerificationError(
            "No registered FaceNet embeddings were found. "
            "Please enroll your face first."
        )

    best_distance = float(min(distances))
    matched = best_distance <= FACE_MATCH_THRESHOLD

    return matched, round(best_distance, 4)
