import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torchvision import models, transforms

from src.utils.io import ensure_dir


class KMeansKeyframeExtractor:
    def __init__(self, device: str = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _extract_features(self, frame: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            input_tensor = self.preprocess(frame).unsqueeze(0).to(self.device)
            features = self.model(input_tensor)
        return features.squeeze().cpu().numpy().flatten()

    def _read_frames(self, video_path: str, step: int = 1, max_frames: int = None) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()
        return frames

    def extract_keyframes(
        self,
        video_path: str,
        output_dir: str,
        min_k: int = 3,
        max_k: int = 15,
        frame_step: int = 1,
        max_frames: int = None,
    ) -> Tuple[int, List[str]]:
        frames = self._read_frames(video_path, step=frame_step, max_frames=max_frames)
        if len(frames) < min_k:
            return 0, []

        features = np.array([self._extract_features(f) for f in frames])
        best_score = -1
        best_centers = None

        max_k = min(max_k, len(features))
        for k in range(min_k, max_k + 1):
            if k >= len(features):
                continue
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                if len(np.unique(labels)) < 2:
                    continue
                score = silhouette_score(features, labels)
                if score > best_score:
                    best_score = score
                    best_centers = kmeans.cluster_centers_
            except ValueError:
                continue

        if best_centers is None:
            return 0, []

        ensure_dir(output_dir)
        saved_paths = []
        for i, center in enumerate(best_centers):
            distances = np.linalg.norm(features - center, axis=1)
            closest_idx = int(np.argmin(distances))
            output_path = os.path.join(output_dir, f"cluster_{i}.png")
            cv2.imwrite(output_path, frames[closest_idx])
            saved_paths.append(output_path)

        return len(saved_paths), saved_paths
