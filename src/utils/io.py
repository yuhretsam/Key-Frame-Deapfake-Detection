import os
from typing import Iterable, List


def list_video_files(folder: str, extensions: Iterable[str] = (".mp4", ".avi", ".mov")) -> List[str]:
    files = []
    for name in os.listdir(folder):
        if name.lower().endswith(tuple(ext.lower() for ext in extensions)):
            files.append(os.path.join(folder, name))
    return sorted(files)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
