import argparse
import os
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.data_preprocessing.kmeans_keyframes import KMeansKeyframeExtractor
from src.data_preprocessing.optical_flow_keyframes import OpticalFlowKeyframeExtractor
from src.models.cnn_lstm import CnnLstm
from src.utils.io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single video.")
    parser.add_argument("--video_path", required=True, help="Path to input video")
    parser.add_argument("--method", required=True, choices=["kmeans", "opticalflow"])
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--max_seq_length", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--temp_dir", default="outputs/infer")
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep_temp", action="store_true", default=False)
    return parser.parse_args()


def _load_frames(folder: str, max_seq_length: int, img_size: int) -> torch.Tensor:
    frame_files = sorted(
        [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    frames: List[torch.Tensor] = []
    for frame_file in frame_files[:max_seq_length]:
        frame_path = os.path.join(folder, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = Image.fromarray(img)
        frames.append(transform(img))

    if not frames:
        frames = [torch.zeros(3, img_size, img_size) for _ in range(max_seq_length)]
    else:
        if len(frames) < max_seq_length:
            padding = [torch.zeros(3, img_size, img_size)] * (max_seq_length - len(frames))
            frames = padding + frames
        elif len(frames) > max_seq_length:
            indices = np.linspace(0, len(frames) - 1, max_seq_length, dtype=int)
            frames = [frames[i] for i in indices]

    return torch.stack(frames).unsqueeze(0)


def _extract_keyframes(video_path: str, method: str, out_dir: str) -> None:
    if method == "kmeans":
        extractor = KMeansKeyframeExtractor()
        extractor.extract_keyframes(video_path, out_dir)
    elif method == "opticalflow":
        extractor = OpticalFlowKeyframeExtractor()
        extractor.extract_keyframes(video_path, out_dir)
    else:
        raise ValueError(f"Unsupported method: {method}")


def main():
    args = parse_args()
    args.method = args.method.lower()
    args.model = args.model.lower()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    video_id = os.path.splitext(os.path.basename(args.video_path))[0]
    temp_video_dir = os.path.join(args.temp_dir, args.method, video_id)
    ensure_dir(temp_video_dir)

    _extract_keyframes(args.video_path, args.method, temp_video_dir)
    inputs = _load_frames(temp_video_dir, args.max_seq_length, args.img_size).to(device)

    model = CnnLstm(num_classes=2, backbone=args.model, freeze_cnn=True).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    labels = ["fake", "real"]
    pred_idx = int(np.argmax(probs))
    print(f"Prediction: {labels[pred_idx]}")
    print(f"Probabilities: fake={probs[0]:.4f}, real={probs[1]:.4f}")

    if not args.keep_temp:
        for name in os.listdir(temp_video_dir):
            os.remove(os.path.join(temp_video_dir, name))
        os.rmdir(temp_video_dir)


if __name__ == "__main__":
    main()
