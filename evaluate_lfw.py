
"""
FaceNet LFW Evaluation with visual performance charts.

Expected dataset structure:
dataset/lfw/
    Person_1/
        image1.jpg
        image2.jpg
    Person_2/
        image1.jpg
        image2.jpg

Outputs:
    metrics.txt
    classification_report.txt
    pair_results.csv
    metrics_chart.png
    confusion_matrix.png
    roc_curve.png

Note: This uses sampled pairs from the folder structure,
not the official LFW benchmark protocol.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from PIL import Image, ImageOps
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_checkpoint_state(weights_path):
    """Load a checkpoint and normalize common state-dict formats."""
    try:
        checkpoint = torch.load(
            str(weights_path),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(str(weights_path), map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported checkpoint format.")

    if isinstance(checkpoint.get("state_dict"), dict):
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint.get("model_state_dict"), dict):
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint.get("model"), dict):
        state = checkpoint["model"]
    else:
        state = checkpoint

    cleaned = {}

    for key, value in state.items():
        if not isinstance(key, str):
            raise RuntimeError("Invalid checkpoint state dictionary.")

        while key.startswith("module."):
            key = key[len("module."):]

        if key.startswith("model."):
            key = key[len("model."):]

        cleaned[key] = value

    return cleaned


def load_model(weights_path, device):
    print("Loading FaceNet checkpoint:", weights_path)

    state = load_checkpoint_state(weights_path)

    # These are classification-head weights, not embedding weights.
    state.pop("logits.weight", None)
    state.pop("logits.bias", None)

    model = InceptionResnetV1(
        pretrained=None,
        classify=False,
    )

    model_state = model.state_dict()

    missing = sorted(set(model_state) - set(state))
    unexpected = sorted(set(state) - set(model_state))

    mismatches = []

    for key in sorted(set(model_state) & set(state)):
        if tuple(model_state[key].shape) != tuple(state[key].shape):
            mismatches.append(
                f"{key}: checkpoint={tuple(state[key].shape)}, "
                f"expected={tuple(model_state[key].shape)}"
            )

    if missing or unexpected or mismatches:
        message = [
            "The checkpoint does not match facenet-pytorch InceptionResnetV1."
        ]

        if missing:
            message.append(f"Missing keys ({len(missing)}): {missing[:15]}")

        if unexpected:
            message.append(
                f"Unexpected keys ({len(unexpected)}): {unexpected[:15]}"
            )

        if mismatches:
            message.append("Tensor shape mismatches:")
            message.extend(mismatches[:15])

        message.append(
            "Check which FaceNet implementation produced this checkpoint. "
            "Do not force-load incompatible weights."
        )

        raise RuntimeError("\n".join(message))

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    print("FaceNet model loaded successfully.")
    return model


def image_to_tensor(image_path, image_size=160):
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32)

    # Standardization used by facenet-pytorch.
    array = (array - 127.5) / 128.0

    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


@torch.inference_mode()
def get_embedding(model, image_path, device):
    tensor = image_to_tensor(image_path).unsqueeze(0).to(device)

    embedding = model(tensor)
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding[0].cpu().numpy().astype(np.float32)


def cosine_distance(embedding_a, embedding_b):
    similarity = np.clip(
        np.dot(embedding_a, embedding_b),
        -1.0,
        1.0,
    )

    return float(1.0 - similarity)


def collect_identities(dataset_root):
    identities = {}

    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir():
            continue

        images = sorted(
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(images) >= 2:
            identities[folder.name] = images

    return identities


def sample_pairs(identities, pair_count, seed):
    rng = random.Random(seed)
    names = sorted(identities)

    if len(names) < 2:
        raise ValueError(
            "At least two identity folders are required, "
            "with at least two images in each."
        )

    pairs = []

    # Genuine pairs: both images belong to the same identity.
    for _ in range(pair_count):
        name = rng.choice(names)
        image_a, image_b = rng.sample(identities[name], 2)
        pairs.append((image_a, image_b, 1))

    # Impostor pairs: images belong to different identities.
    for _ in range(pair_count):
        name_a, name_b = rng.sample(names, 2)
        image_a = rng.choice(identities[name_a])
        image_b = rng.choice(identities[name_b])
        pairs.append((image_a, image_b, 0))

    rng.shuffle(pairs)
    return pairs


def save_confusion_matrix(y_true, y_pred, output_dir):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Impostor", "Genuine"],
        yticklabels=["Impostor", "Genuine"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title="FaceNet Verification Confusion Matrix",
    )

    cutoff = matrix.max() / 2 if matrix.size else 0

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > cutoff else "black",
            )

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    return matrix


def save_metrics_chart(metrics, output_dir):
    names = list(metrics.keys())
    values = [metrics[name] * 100 for name in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, values)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("FaceNet Verification Performance")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 1, 99),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_dir / "metrics_chart.png", dpi=180)
    plt.close(fig)


def save_roc_curve(y_true, distances, auc, output_dir):
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true,
        -distances,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {auc:.3f}",
    )
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="FaceNet Verification ROC Curve",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FaceNet on folder-organized LFW images."
    )

    parser.add_argument(
        "--dataset",
        default="dataset/lfw",
        help="Path to the LFW dataset root",
    )
    parser.add_argument(
        "--weights",
        default="facenet_vggface2.pt",
        help="Path to the FaceNet checkpoint",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=500,
        help="Number of genuine pairs and impostor pairs",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Cosine-distance threshold for a genuine match",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="facenet_evaluation_results",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    args = parser.parse_args()

    if args.pairs < 1:
        raise SystemExit("--pairs must be at least 1.")

    dataset_root = Path(args.dataset).resolve()
    weights_path = Path(args.weights).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset folder not found: {dataset_root}")

    if not weights_path.is_file():
        raise SystemExit(f"Checkpoint not found: {weights_path}")

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    identities = collect_identities(dataset_root)

    print("Dataset:", dataset_root)
    print("Identity folders with at least two images:", len(identities))
    print("Device:", device)

    if len(identities) < 2:
        raise SystemExit(
            "Expected dataset/lfw/person_name/image.jpg folders "
            "for at least two people."
        )

    model = load_model(weights_path, device)
    pairs = sample_pairs(identities, args.pairs, args.seed)

    embedding_cache = {}
    rows = []
    failures = 0

    for index, (path_a, path_b, truth) in enumerate(pairs, start=1):
        try:
            if path_a not in embedding_cache:
                embedding_cache[path_a] = get_embedding(
                    model, path_a, device
                )

            if path_b not in embedding_cache:
                embedding_cache[path_b] = get_embedding(
                    model, path_b, device
                )

            distance = cosine_distance(
                embedding_cache[path_a],
                embedding_cache[path_b],
            )

            predicted = int(distance <= args.threshold)

            rows.append({
                "image_a": str(path_a.relative_to(dataset_root)),
                "image_b": str(path_b.relative_to(dataset_root)),
                "true_label": truth,
                "predicted_label": predicted,
                "cosine_distance": distance,
                "correct": int(predicted == truth),
            })

        except Exception as error:
            failures += 1
            print(f"[WARN] Pair {index} skipped: {error}")

        if index % 50 == 0 or index == len(pairs):
            print(
                f"Processed {index}/{len(pairs)} pairs; "
                f"valid={len(rows)}; failed={failures}"
            )

    if not rows:
        raise SystemExit("No valid image pairs were evaluated.")

    y_true = np.asarray(
        [row["true_label"] for row in rows],
        dtype=np.int32,
    )
    y_pred = np.asarray(
        [row["predicted_label"] for row in rows],
        dtype=np.int32,
    )
    distances = np.asarray(
        [row["cosine_distance"] for row in rows],
        dtype=np.float64,
    )

    if len(np.unique(y_true)) < 2:
        raise SystemExit(
            "Evaluation must include genuine and impostor pairs."
        )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )
    auc = roc_auc_score(y_true, -distances)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": auc,
    }

    with (output_dir / "pair_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Impostor (different person)",
            "Genuine (same person)",
        ],
        zero_division=0,
    )

    (output_dir / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    matrix = save_confusion_matrix(
        y_true, y_pred, output_dir
    )
    save_metrics_chart(metrics, output_dir)
    save_roc_curve(y_true, distances, auc, output_dir)

    tn, fp, fn, tp = matrix.ravel()

    summary = [
        "FaceNet LFW Folder-Based Evaluation",
        "Protocol: sampled pairs; not the official LFW protocol",
        f"Dataset: {dataset_root}",
        f"Weights: {weights_path}",
        f"Device: {device}",
        f"Identity folders: {len(identities)}",
        f"Valid pairs: {len(rows)}",
        f"Skipped pairs: {failures}",
        f"Cosine-distance threshold: {args.threshold:.4f}",
        "",
    ]

    for name, value in metrics.items():
        summary.append(f"{name}: {value * 100:.2f}%")

    summary.extend([
        "",
        f"True negatives: {tn}",
        f"False positives: {fp}",
        f"False negatives: {fn}",
        f"True positives: {tp}",
        "",
        "Classification report:",
        report,
        "",
        "Select the final threshold using a separate validation set.",
    ])

    (output_dir / "metrics.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("\n=== FACE VERIFICATION RESULTS ===")

    for name, value in metrics.items():
        print(f"{name}: {value * 100:.2f}%")

    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("\nClassification report:")
    print(report)

    print("\nReport images saved in:", output_dir)
    print(" - metrics_chart.png")
    print(" - confusion_matrix.png")
    print(" - roc_curve.png")


if __name__ == "__main__":
    main()
