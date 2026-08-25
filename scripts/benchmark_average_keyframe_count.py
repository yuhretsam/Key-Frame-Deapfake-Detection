"""Measure the average number of extracted keyframes per video folder."""

import argparse
import random
from pathlib import Path
from typing import List, Tuple


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count keyframe images in N video subfolders and report their average."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="One class folder, for example data/kmeans/fake or data/opticalflow/real",
    )
    parser.add_argument(
        "--num_subfolders",
        required=True,
        type=int,
        help="Number of video subfolders to include in the experiment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed seed used to select subfolders (default: 42)",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Optional text file to save the average and per-folder counts",
    )
    return parser.parse_args()


def collect_video_folders(input_dir: Path) -> List[Path]:
    """Return immediate child directories in deterministic order."""
    return sorted(
        (path for path in input_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )


def count_images(video_dir: Path) -> int:
    """Count keyframe images directly inside one video folder."""
    return sum(
        1
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_report(input_dir: Path, seed: int, counts: List[Tuple[Path, int]]) -> str:
    average = sum(count for _, count in counts) / len(counts)
    lines = [
        "Average Number of Frames Per Method",
        f"Input directory: {input_dir.resolve()}",
        f"Fixed random seed: {seed}",
        f"Video subfolders sampled: {len(counts)}",
        f"Average keyframes per video: {average:.3f}",
        "",
        "video_folder | keyframe_count",
    ]
    lines.extend(f"{folder.name} | {count}" for folder, count in counts)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.num_subfolders < 1:
        raise ValueError("--num_subfolders must be at least 1")

    video_folders = collect_video_folders(input_dir)
    if args.num_subfolders > len(video_folders):
        raise ValueError(
            f"Requested {args.num_subfolders} subfolders but found only "
            f"{len(video_folders)} in {input_dir}"
        )

    selected_folders = random.Random(args.seed).sample(video_folders, args.num_subfolders)
    counts = [(folder, count_images(folder)) for folder in selected_folders]
    report = build_report(input_dir, args.seed, counts)
    print(report, end="")

    if args.output_file is not None:
        output_file = args.output_file.expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report, encoding="utf-8")
        print(f"Saved report: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
