import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_video_paths_and_labels(
    data_root: str, class_folders: List[str], split_name: str = None
):
    video_paths, labels = [], []
    class_map = {name: i for i, name in enumerate(sorted(class_folders))}

    for class_name in class_folders:
        base_dir = os.path.join(data_root, class_name)
        split_dir = os.path.join(base_dir, split_name) if split_name else base_dir
        if not os.path.isdir(split_dir):
            continue
        for video_folder in sorted(os.listdir(split_dir)):
            video_path = os.path.join(split_dir, video_folder)
            if os.path.isdir(video_path):
                if any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(video_path)):
                    video_paths.append(video_path)
                    labels.append(class_map[class_name])

    return video_paths, np.array(labels), sorted(class_folders)


class VideoKeyframeDataset(Dataset):
    def __init__(
        self,
        video_paths: List[str],
        labels: np.ndarray,
        max_seq_length: int,
        img_size: int = 224,
        transform=None,
    ) -> None:
        self.video_paths = video_paths
        self.labels = labels
        self.max_seq_length = max_seq_length
        self.img_size = img_size
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_frames(self, path: str) -> List[torch.Tensor]:
        frame_files = sorted(
            [f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        frames = []
        for frame_file in frame_files[: self.max_seq_length]:
            frame_path = os.path.join(path, frame_file)
            img = cv2.imread(frame_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = Image.fromarray(img)
            frames.append(self.transform(img))
        return frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path = self.video_paths[idx]
        label = int(self.labels[idx])
        frames = self._load_frames(video_path)

        if not frames:
            frames = [torch.zeros(3, self.img_size, self.img_size) for _ in range(self.max_seq_length)]
        else:
            if len(frames) < self.max_seq_length:
                padding = [torch.zeros(3, self.img_size, self.img_size)] * (
                    self.max_seq_length - len(frames)
                )
                frames = padding + frames
            elif len(frames) > self.max_seq_length:
                indices = np.linspace(0, len(frames) - 1, self.max_seq_length, dtype=int)
                frames = [frames[i] for i in indices]

        return torch.stack(frames), torch.tensor(label, dtype=torch.long)
