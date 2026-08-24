"""Benchmark per-video keyframe selection and CNN+LSTM classification latency.

The benchmark deliberately runs one complete video at a time. Keyframes are
kept only for that video and passed to the classifier in memory, so the result
does not include writing temporary PNG files or holding a whole dataset in RAM.
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from mtcnn import MTCNN
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_lstm import build_backbone, build_model
from src.utils.io import ensure_dir


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
MODEL_NAMES = ("resnet50", "vgg16", "efficientnet_b0", "mobilenet_v2")
SelectionImage = Tuple[str, np.ndarray]


class BenchmarkKMeansSelector:
    """K-Means selector local to this timing experiment.

    This is intentionally separate from the project extractor so choosing a
    benchmark backbone does not alter the existing extraction pipeline.
    """

    def __init__(self, backbone: str, device: torch.device) -> None:
        self.device = device
        self.model, _ = build_backbone(backbone, freeze=True, pretrained=True)
        self.model = self.model.to(device).eval()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _embed(self, frame: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            inputs = self.preprocess(frame).unsqueeze(0).to(self.device)
            features = self.pool(self.model(inputs))
        return features.flatten(1).squeeze(0).cpu().numpy()

    @staticmethod
    def _read_frames(video_path: Path, frame_step: int, max_frames: int | None) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        frames: List[np.ndarray] = []
        index = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            if index % frame_step == 0:
                frames.append(frame)
            index += 1
            if max_frames is not None and len(frames) >= max_frames:
                break
        cap.release()
        return frames

    def select(
        self,
        video_path: Path,
        min_k: int,
        max_k: int,
        frame_step: int,
        max_frames: int | None,
    ) -> List[SelectionImage]:
        frames = self._read_frames(video_path, frame_step, max_frames)
        if len(frames) < min_k:
            return []
        embeddings = np.asarray([self._embed(frame) for frame in frames])
        best_score = -1.0
        best_centers = None
        for clusters in range(min_k, min(max_k, len(embeddings)) + 1):
            if clusters >= len(embeddings):
                continue
            try:
                kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                if len(np.unique(labels)) < 2:
                    continue
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_centers = kmeans.cluster_centers_
            except ValueError:
                continue
        if best_centers is None:
            return []
        return [
            (f"cluster_{index}.png", frames[int(np.argmin(np.linalg.norm(embeddings - center, axis=1)))])
            for index, center in enumerate(best_centers)
        ]


class BenchmarkOpticalFlowSelector:
    """Optical-flow selector local to this timing experiment."""

    def __init__(self, lambda_weight: float, threshold_ratio: float) -> None:
        self.lambda_weight = lambda_weight
        self.threshold_ratio = threshold_ratio
        self.face_detector = MTCNN()

    @staticmethod
    def _flow(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(current, cv2.COLOR_BGR2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )

    def _face(self, frame: np.ndarray) -> Tuple[int, int, int, int] | None:
        faces = self.face_detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not faces:
            return None
        x, y, width, height = max(faces, key=lambda item: item["confidence"])["box"]
        x = max(0, x)
        y = max(0, y)
        width = min(width, frame.shape[1] - x)
        height = min(height, frame.shape[0] - y)
        margin_width, margin_height = int(width * 0.2), int(height * 0.2)
        x, y = max(0, x - margin_width), max(0, y - margin_height)
        width = min(frame.shape[1] - x, width + 2 * margin_width)
        height = min(frame.shape[0] - y, height + 2 * margin_height)
        return x, y, width, height

    def _energy(self, flow: np.ndarray, face: Tuple[int, int, int, int]) -> float:
        x, y, width, height = face
        flow_face = flow[y : y + height, x : x + width]
        velocity = np.sqrt(flow_face[..., 0] ** 2 + flow_face[..., 1] ** 2)
        angles = np.arctan2(flow_face[..., 1], flow_face[..., 0])
        angles = np.where(angles < 0, angles + 2 * np.pi, angles)
        angle_average = np.mean(angles)
        angle_maximum = angles[np.unravel_index(np.argmax(velocity), velocity.shape)]
        weights = (
            (np.abs(angles - angle_average) / np.pi * self.lambda_weight) ** 2
            + (np.abs(angles - angle_maximum) / np.pi * self.lambda_weight) ** 2
        )
        return float(np.sum(weights * velocity ** 2))

    @staticmethod
    def _flow_image(flow: np.ndarray) -> np.ndarray:
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def select(self, video_path: Path) -> List[SelectionImage]:
        cap = cv2.VideoCapture(str(video_path))
        frames: List[np.ndarray] = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)
        cap.release()
        if len(frames) < 2:
            return []

        energies: List[float] = []
        previous_face = self._face(frames[0])
        for index in range(1, len(frames)):
            current_face = self._face(frames[index])
            if previous_face is None or current_face is None:
                energies.append(0.0)
            else:
                energies.append(self._energy(self._flow(frames[index - 1], frames[index]), current_face))
            previous_face = current_face

        threshold = self.threshold_ratio * (max(energies) if energies else 1.0)
        selected: List[SelectionImage] = []
        for index, energy in enumerate(energies):
            if index <= 0 or index >= len(frames) or energy < threshold:
                continue
            face = self._face(frames[index])
            if face is None:
                continue
            x, y, width, height = face
            cropped = self._flow_image(self._flow(frames[index - 1], frames[index]))[y : y + height, x : x + width]
            selected.append((f"flow_face_{index}.png", cv2.resize(cropped, (224, 224))))
        return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-video keyframe selection and CNN+LSTM classification time."
    )
    parser.add_argument("--input_dir", required=True, type=Path, help="Folder containing videos")
    parser.add_argument("--output_dir", required=True, type=Path, help="Folder for sample and result text files")
    parser.add_argument("--backbone", required=True, choices=MODEL_NAMES)
    parser.add_argument("--method", required=True, choices=("kmeans", "opticalflow"))
    parser.add_argument("--num_videos", required=True, type=int, help="Number of videos to sample")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed used for sampling")
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:0 (default: auto)")
    parser.add_argument("--recursive", action="store_true", help="Find videos recursively")
    parser.add_argument("--max_seq_length", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--min_k", type=int, default=3)
    parser.add_argument("--max_k", type=int, default=15)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--threshold_ratio", type=float, default=0.3)
    parser.add_argument("--lambda_weight", type=float, default=1.0)
    return parser.parse_args()


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but is unavailable: {requested_device}")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_videos(input_dir: Path, recursive: bool) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    videos = sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )
    duplicate_names = {}
    for path in videos:
        key = path.name.casefold()
        if key in duplicate_names:
            raise ValueError(
                "Duplicate video names cannot be recorded unambiguously in the sample file:\n"
                f"  - {duplicate_names[key]}\n  - {path}"
            )
        duplicate_names[key] = path
    return videos


def choose_and_save_videos(
    input_dir: Path,
    output_dir: Path,
    backbone: str,
    method: str,
    num_videos: int,
    seed: int,
    recursive: bool,
) -> Tuple[List[Path], Path]:
    videos = collect_videos(input_dir, recursive)
    if num_videos < 1:
        raise ValueError("--num_videos must be at least 1")
    if num_videos > len(videos):
        raise ValueError(
            f"Requested {num_videos} videos but only found {len(videos)} in {input_dir}"
        )

    selected = random.Random(seed).sample(videos, num_videos)
    sample_file = output_dir / f"{backbone}_{method}.txt"
    sample_file.write_text("\n".join(path.name for path in selected) + "\n", encoding="utf-8")
    return selected, sample_file


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_ms(device: torch.device, operation: Callable[[], object]) -> Tuple[float, object]:
    """Time an operation, synchronizing CUDA so asynchronous work is included."""
    synchronize(device)
    start = time.perf_counter()
    result = operation()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0, result


def make_classifier_input(
    keyframes: Sequence[SelectionImage],
    max_seq_length: int,
    img_size: int,
) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    frames: List[torch.Tensor] = []
    for _, frame in sorted(keyframes, key=lambda item: item[0])[:max_seq_length]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (img_size, img_size))
        frames.append(transform(Image.fromarray(resized)))

    if not frames:
        raise RuntimeError("Keyframe selection produced no usable images")
    if len(frames) < max_seq_length:
        frames = [torch.zeros(3, img_size, img_size)] * (max_seq_length - len(frames)) + frames
    return torch.stack(frames).unsqueeze(0)


def build_keyframe_selector(args: argparse.Namespace, device: torch.device) -> Callable[[Path], List[SelectionImage]]:
    if args.method == "kmeans":
        extractor = BenchmarkKMeansSelector(args.backbone, device)

        def select(video_path: Path) -> List[SelectionImage]:
            return extractor.select(
                video_path,
                args.min_k,
                args.max_k,
                args.frame_step,
                args.max_frames,
            )

        return select

    extractor = BenchmarkOpticalFlowSelector(args.lambda_weight, args.threshold_ratio)

    def select(video_path: Path) -> List[SelectionImage]:
        return extractor.select(video_path)

    return select


def format_result(
    args: argparse.Namespace,
    device: torch.device,
    sample_file: Path,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Tuple[str, str]],
) -> str:
    successful_rows = [row for row in rows if row["status"] == "ok"]

    def average(name: str) -> float:
        return float(np.mean([float(row[name]) for row in successful_rows]))

    lines = [
        "Pipeline timing experiment",
        f"Input directory: {args.input_dir.resolve()}",
        f"Sample file: {sample_file.resolve()}",
        f"Method: {args.method}",
        f"Classification backbone: {args.backbone}",
        (
            f"Keyframe embedding backbone: {args.backbone}"
            if args.method == "kmeans"
            else "Keyframe embedding backbone: not applicable (optical flow uses MTCNN + Farneback)"
        ),
        f"Device: {device}",
        f"Fixed random seed: {args.seed}",
        f"Selected videos: {args.num_videos}",
        f"Successful videos: {len(successful_rows)}",
        f"Failed videos: {len(failures)}",
        "",
    ]
    if successful_rows:
        lines.extend(
            [
                f"Average Key Frame Selection Time (ms): {average('keyframe_ms'):.3f}",
                f"Average Classification Time (ms): {average('classification_ms'):.3f}",
                f"Average Total Time (ms): {average('total_ms'):.3f}",
                "",
                "Per-video timing (ms):",
                "video | keyframe_selection | classification | total | keyframes",
            ]
        )
        for row in successful_rows:
            lines.append(
                f"{row['video']} | {float(row['keyframe_ms']):.3f} | "
                f"{float(row['classification_ms']):.3f} | {float(row['total_ms']):.3f} | "
                f"{row['keyframe_count']}"
            )
    else:
        lines.extend(["No video completed both phases; averages are unavailable."])

    if failures:
        lines.extend(["", "Failures:"])
        lines.extend(f"{video} | {reason}" for video, reason in failures)
    lines.extend(
        [
            "",
            "Timing boundary: model/extractor initialization is excluded.",
            "Keyframes are handed to classification in memory; temporary PNG write/read time is excluded.",
            "Classification time includes image preprocessing, host-to-device transfer, CNN embedding, LSTM, and logits.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if args.min_k < 1 or args.max_k < args.min_k:
        raise ValueError("Require 1 <= --min_k <= --max_k")
    if args.frame_step < 1:
        raise ValueError("--frame_step must be at least 1")

    ensure_dir(str(args.output_dir))
    selected_videos, sample_file = choose_and_save_videos(
        args.input_dir,
        args.output_dir,
        args.backbone,
        args.method,
        args.num_videos,
        args.seed,
        args.recursive,
    )
    print(f"Saved {len(selected_videos)} sampled filenames to: {sample_file}")

    device = resolve_device(args.device)
    selector = build_keyframe_selector(args, device)
    classifier = build_model(
        model_name=args.backbone,
        num_classes=2,
        freeze_cnn=True,
        pretrained=False,
    ).to(device)
    classifier.eval()

    rows: List[Dict[str, object]] = []
    failures: List[Tuple[str, str]] = []
    for video_path in tqdm(selected_videos, desc="Pipeline timing", unit="video"):
        try:
            total_start = time.perf_counter()
            keyframe_ms, keyframes = elapsed_ms(device, lambda: selector(video_path))
            if not keyframes:
                raise RuntimeError("Keyframe selection produced no usable images")

            def classify() -> torch.Tensor:
                inputs = make_classifier_input(keyframes, args.max_seq_length, args.img_size).to(device)
                with torch.no_grad():
                    return classifier(inputs)

            classification_ms, _ = elapsed_ms(device, classify)
            total_ms = (time.perf_counter() - total_start) * 1000.0
            rows.append(
                {
                    "status": "ok",
                    "video": video_path.name,
                    "keyframe_ms": keyframe_ms,
                    "classification_ms": classification_ms,
                    "total_ms": total_ms,
                    "keyframe_count": len(keyframes),
                }
            )
        except Exception as error:
            failures.append((video_path.name, str(error) or error.__class__.__name__))
            rows.append({"status": "failed", "video": video_path.name})
            tqdm.write(f"[ERROR] {video_path.name}: {failures[-1][1]}")

    result_path = args.output_dir / f"result_{args.backbone}_{args.method}.txt"
    result_path.write_text(
        format_result(args, device, sample_file, rows, failures), encoding="utf-8"
    )
    print(f"Timing result saved to: {result_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
