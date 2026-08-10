import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing.keyframe_dataset import (
    VideoKeyframeDataset,
    build_eval_transform,
    load_video_paths_and_labels,
)
from src.models.cnn_lstm import build_model
from src.training.metrics import compute_metrics
from src.utils.io import ensure_dir


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
SPLIT_FOLDER_NAMES = {"train", "val", "test"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN+LSTM on 100% of a cross-dataset test set."
    )
    parser.add_argument("--data_root", required=True, help="Root of the cross-test dataset")
    parser.add_argument("--method", required=True, help="Keyframe method folder name")
    parser.add_argument("--class_folders", nargs="+", default=["fake", "real"])
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--weights", required=True, help="Training checkpoint (.pth)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="For example: cpu, cuda, or cuda:0")
    parser.add_argument("--output_dir", default="outputs/crosstest")
    parser.add_argument("--max_seq_length", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def _expected_layout(base_dir: str, class_folders: Sequence[str]) -> str:
    lines = ["Expected cross-test structure:", "", f"{base_dir}/"]
    for class_name in class_folders:
        lines.append(f"    {class_name}/<video_id>/*.(png|jpg|jpeg)")
    return "\n".join(lines)


def _contains_image(video_dir: str) -> bool:
    return any(
        name.lower().endswith(IMAGE_EXTENSIONS)
        for name in os.listdir(video_dir)
        if os.path.isfile(os.path.join(video_dir, name))
    )


def _validate_cross_test_layout(
    base_dir: str, class_folders: Sequence[str]
) -> Dict[str, int]:
    if len(class_folders) != 2 or len(set(class_folders)) != 2:
        raise ValueError("Cross-dataset metrics require exactly two unique class folders.")

    class_counts: Dict[str, int] = {}
    for class_name in class_folders:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Class folder not found: {class_dir}\n\n{_expected_layout(base_dir, class_folders)}"
            )

        child_dirs = sorted(
            name
            for name in os.listdir(class_dir)
            if os.path.isdir(os.path.join(class_dir, name))
        )
        split_dirs = sorted(name for name in child_dirs if name.lower() in SPLIT_FOLDER_NAMES)
        if split_dirs:
            raise ValueError(
                "scripts/test.py is for cross-dataset evaluation and does not accept "
                f"train/val/test splits. Found {split_dirs} under {class_dir}.\n\n"
                f"{_expected_layout(base_dir, class_folders)}"
            )
        if not child_dirs:
            raise ValueError(
                f"No video folders found under: {class_dir}\n\n"
                f"{_expected_layout(base_dir, class_folders)}"
            )

        invalid_dirs = [
            os.path.join(class_dir, name)
            for name in child_dirs
            if not _contains_image(os.path.join(class_dir, name))
        ]
        if invalid_dirs:
            preview = "\n".join(f"  - {path}" for path in invalid_dirs[:10])
            suffix = "\n  - ..." if len(invalid_dirs) > 10 else ""
            raise ValueError(
                "Every immediate child directory must be a video containing at least "
                f"one image. Invalid directories:\n{preview}{suffix}"
            )
        class_counts[class_name] = len(child_dirs)

    return class_counts


def _resolve_device(requested_device: str = None) -> torch.device:
    if requested_device:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested_device}")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    video_paths: Sequence[str],
    class_names: Sequence[str],
    device: torch.device,
) -> Tuple[float, float, Dict[str, float], List[Dict[str, object]]]:
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    prediction_rows: List[Dict[str, object]] = []
    path_offset = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            batch_labels = labels.cpu().numpy().tolist()
            batch_predictions = predictions.cpu().numpy().tolist()
            batch_probabilities = probabilities.cpu().numpy()

            y_true.extend(batch_labels)
            y_pred.extend(batch_predictions)
            y_prob.extend(batch_probabilities[:, 1].tolist())

            batch_paths = video_paths[path_offset : path_offset + batch_size]
            for video_path, true_idx, predicted_idx, probs in zip(
                batch_paths, batch_labels, batch_predictions, batch_probabilities
            ):
                row: Dict[str, object] = {
                    "video_id": os.path.basename(video_path),
                    "true_label": class_names[true_idx],
                    "predicted_label": class_names[predicted_idx],
                }
                for class_idx, class_name in enumerate(class_names):
                    row[f"prob_{class_name}"] = float(probs[class_idx])
                prediction_rows.append(row)
            path_offset += batch_size

    if path_offset != len(video_paths):
        raise RuntimeError(
            f"Evaluated {path_offset} videos but received {len(video_paths)} video paths."
        )

    num_videos = len(test_loader.dataset)
    test_loss = running_loss / num_videos
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    metrics = compute_metrics(y_true, y_pred, y_prob)
    return test_loss, accuracy, metrics, prediction_rows


def _json_safe(value: float):
    value = float(value)
    return value if np.isfinite(value) else None


def _save_results(
    args,
    class_names: Sequence[str],
    class_counts: Dict[str, int],
    test_loss: float,
    accuracy: float,
    metrics: Dict[str, float],
    prediction_rows: List[Dict[str, object]],
) -> Tuple[str, str]:
    ensure_dir(args.output_dir)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    predictions_path = os.path.join(args.output_dir, "predictions.csv")

    metric_summary = {"accuracy": _json_safe(accuracy)}
    metric_summary.update({name: _json_safe(value) for name, value in metrics.items()})
    summary = {
        "data_root": args.data_root,
        "method": args.method,
        "model": args.model,
        "checkpoint": args.weights,
        "classes": list(class_names),
        "class_mapping": {name: idx for idx, name in enumerate(class_names)},
        "class_counts": class_counts,
        "num_videos": len(prediction_rows),
        "test_loss": _json_safe(test_loss),
        "metrics": metric_summary,
    }
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, allow_nan=False)

    fieldnames = ["video_id", "true_label", "predicted_label"] + [
        f"prob_{class_name}" for class_name in class_names
    ]
    with open(predictions_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    return metrics_path, predictions_path


def main():
    args = parse_args()
    args.method = args.method.lower()
    args.model = args.model.lower()
    base_dir = os.path.join(args.data_root, args.method)

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    class_counts = _validate_cross_test_layout(base_dir, args.class_folders)
    video_paths, labels, class_names = load_video_paths_and_labels(
        base_dir, args.class_folders, split_name=None
    )
    expected_count = sum(class_counts.values())
    if len(video_paths) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} videos but the dataset loader returned {len(video_paths)}."
        )

    test_dataset = VideoKeyframeDataset(
        video_paths,
        labels,
        args.max_seq_length,
        img_size=args.img_size,
        transform=build_eval_transform(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = _resolve_device(args.device)
    model = build_model(
        model_name=args.model,
        num_classes=len(class_names),
        freeze_cnn=True,
        pretrained=False,
    )
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    class_mapping = {name: index for index, name in enumerate(class_names)}

    print("=" * 40)
    print("Cross-dataset evaluation")
    print("=" * 40)
    print(f"Data root      : {args.data_root}")
    print(f"Method         : {args.method}")
    print(f"Checkpoint     : {args.weights}")
    print(f"Model          : {args.model}")
    print(f"Device         : {device}")
    print(f"Classes        : {class_names}")
    print(f"Class mapping  : {class_mapping}")
    print(f"Total videos   : {len(test_dataset)}")
    for class_name in class_names:
        print(f"{f'{class_name.capitalize()} videos':<15}: {class_counts[class_name]}")
    print("=" * 40)

    test_loss, accuracy, metrics, prediction_rows = _evaluate(
        model, test_loader, video_paths, class_names, device
    )
    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Accuracy       : {accuracy:.4f}")
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():<15}: {value:.4f}")

    metrics_path, predictions_path = _save_results(
        args,
        class_names,
        class_counts,
        test_loss,
        accuracy,
        metrics,
        prediction_rows,
    )
    print("=" * 40)
    print(f"Metrics saved  : {metrics_path}")
    print(f"Predictions    : {predictions_path}")


if __name__ == "__main__":
    main()
