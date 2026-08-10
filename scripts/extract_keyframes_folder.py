import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from src.utils.io import list_video_files


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
COMPLETE_MARKER = ".complete"


class ExtractionSetupError(RuntimeError):
    """Raised when the batch cannot be started safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract keyframes from videos directly inside one folder."
    )
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=("kmeans", "opticalflow"))

    parser.add_argument("--min_k", type=int, default=3)
    parser.add_argument("--max_k", type=int, default=15)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)

    parser.add_argument("--threshold_ratio", type=float, default=0.3)
    parser.add_argument("--lambda_weight", type=float, default=1.0)

    existing_mode = parser.add_mutually_exclusive_group()
    existing_mode.add_argument(
        "--skip_existing",
        dest="overwrite",
        action="store_false",
        help="Skip outputs containing keyframes and a .complete marker (default)",
    )
    existing_mode.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Re-extract videos even when a completed output already exists",
    )
    parser.set_defaults(overwrite=False)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise ExtractionSetupError(f"Input directory not found: {args.input_dir}")
    if args.min_k < 1:
        raise ExtractionSetupError("--min_k must be at least 1")
    if args.max_k < args.min_k:
        raise ExtractionSetupError("--max_k must be greater than or equal to --min_k")
    if args.frame_step < 1:
        raise ExtractionSetupError("--frame_step must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ExtractionSetupError("--max_frames must be at least 1 when provided")


def collect_videos(input_dir: Path) -> List[Path]:
    """Collect supported video files directly from input_dir, without recursion."""
    videos = [
        Path(path)
        for path in list_video_files(str(input_dir))
        if Path(path).is_file()
    ]
    seen_stems = {}
    for video_path in videos:
        normalized_stem = video_path.stem.casefold()
        if normalized_stem in seen_stems:
            raise ExtractionSetupError(
                "Multiple input videos would use the same output folder:\n"
                f"  - {seen_stems[normalized_stem]}\n"
                f"  - {video_path}"
            )
        seen_stems[normalized_stem] = video_path
    return videos


def build_extractor(args: argparse.Namespace):
    """Instantiate the existing extractor implementation selected by the user."""
    if args.method == "kmeans":
        from src.data_preprocessing.kmeans_keyframes import KMeansKeyframeExtractor

        return KMeansKeyframeExtractor()

    from src.data_preprocessing.optical_flow_keyframes import OpticalFlowKeyframeExtractor

    return OpticalFlowKeyframeExtractor(
        lambda_weight=args.lambda_weight,
        threshold_ratio=args.threshold_ratio,
    )


def contains_keyframes(video_output_dir: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in video_output_dir.iterdir()
    )


def is_complete(video_output_dir: Path) -> bool:
    return (
        video_output_dir.is_dir()
        and (video_output_dir / COMPLETE_MARKER).is_file()
        and contains_keyframes(video_output_dir)
    )


def reset_output_dir(video_output_dir: Path, output_root: Path) -> None:
    """Remove one incomplete/overwritten video output without touching other outputs."""
    if video_output_dir.parent.resolve() != output_root.resolve():
        raise RuntimeError(f"Unsafe output path outside output root: {video_output_dir}")
    if video_output_dir.is_symlink():
        raise RuntimeError(f"Refusing to reset symlink output: {video_output_dir}")
    if video_output_dir.exists():
        if not video_output_dir.is_dir():
            raise RuntimeError(f"Output path exists and is not a directory: {video_output_dir}")
        shutil.rmtree(video_output_dir)
    video_output_dir.mkdir(parents=False)


def extract_video(
    extractor,
    video_path: Path,
    video_output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Call the existing extractor API without reimplementing either algorithm."""
    if args.method == "kmeans":
        extractor.extract_keyframes(
            str(video_path),
            str(video_output_dir),
            min_k=args.min_k,
            max_k=args.max_k,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
        )
    else:
        extractor.extract_keyframes(str(video_path), str(video_output_dir))

    if not contains_keyframes(video_output_dir):
        raise RuntimeError("Extractor completed but produced no keyframe image")
    (video_output_dir / COMPLETE_MARKER).touch()


def print_summary(
    args: argparse.Namespace,
    total: int,
    processed: int,
    skipped: int,
    failures: List[Tuple[Path, str]],
) -> None:
    print("\n" + "=" * 40)
    print("Extraction summary")
    print("=" * 40)
    print(f"Input:     {args.input_dir}")
    print(f"Output:    {args.output_dir}")
    print(f"Method:    {args.method}")
    print()
    print(f"Total:       {total}")
    print(f"Processed:   {processed}")
    print(f"Skipped:     {skipped}")
    print(f"Failed:      {len(failures)}")
    print("=" * 40)

    if failures:
        print("\nFailed videos:")
        for video_path, reason in failures:
            print(f"  - {video_path.name}: {reason}")


def run(args: argparse.Namespace) -> int:
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(args.input_dir)
    if not videos:
        print(f"[WARNING] No video files found in:\n{args.input_dir}")
        print_summary(args, total=0, processed=0, skipped=0, failures=[])
        return 0

    try:
        extractor = build_extractor(args)
    except Exception as error:
        raise ExtractionSetupError(
            f"Could not initialize the {args.method} extractor: {error}"
        ) from error
    processed = 0
    skipped = 0
    failures: List[Tuple[Path, str]] = []
    progress_name = "KMeans" if args.method == "kmeans" else "Optical Flow"

    for video_path in tqdm(videos, desc=progress_name, unit="video"):
        video_output_dir = args.output_dir / video_path.stem
        if not args.overwrite and is_complete(video_output_dir):
            tqdm.write(f"[SKIP] {video_path.name} - already processed")
            skipped += 1
            continue

        try:
            reset_output_dir(video_output_dir, args.output_dir)
            extract_video(extractor, video_path, video_output_dir, args)
            processed += 1
        except Exception as error:
            reason = str(error) or error.__class__.__name__
            tqdm.write(f"[ERROR] {video_path.name}\nReason: {reason}")
            failures.append((video_path, reason))

    print_summary(args, len(videos), processed, skipped, failures)
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except (ExtractionSetupError, OSError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
