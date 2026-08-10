import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List


CLASSES = ("fake", "real")
SPLITS = ("train", "val", "test")
MANIFEST_NAME = "rebuild_split_manifest.json"

ReferenceIds = Dict[str, Dict[str, List[str]]]
SourceVideos = Dict[str, Dict[str, Path]]
ReconstructionPlan = Dict[str, Dict[str, List[dict]]]


class ReconstructionError(RuntimeError):
    """Raised when reconstruction cannot proceed without risking dataset corruption."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the BioDeepAV train/val/test split from processed keyframe "
            "directory names. Dry-run is the default."
        )
    )
    parser.add_argument(
        "--video_root",
        required=True,
        type=Path,
        help="BioDeepAV root containing fake/videos and real/videos",
    )
    parser.add_argument(
        "--reference_root",
        required=True,
        type=Path,
        help="Processed reference root containing <class>/<split>/<video_id>",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan, validate, and write the manifest without changing video files (default)",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Apply the validated manifest by moving and deleting videos",
    )
    return parser.parse_args()


def collect_reference_ids(reference_root: Path) -> ReferenceIds:
    """Collect direct child directory names as video IDs for every class/split."""
    if not reference_root.is_dir():
        raise ReconstructionError(f"Reference root not found: {reference_root}")

    reference_ids: ReferenceIds = {}
    for class_name in CLASSES:
        reference_ids[class_name] = {}
        for split_name in SPLITS:
            split_dir = reference_root / class_name / split_name
            if not split_dir.is_dir():
                raise ReconstructionError(f"Reference split directory not found: {split_dir}")
            reference_ids[class_name][split_name] = sorted(
                child.name for child in split_dir.iterdir() if child.is_dir()
            )
    return reference_ids


def collect_source_videos(video_root: Path) -> SourceVideos:
    """Collect MP4 files from each <class>/videos directory by exact stem."""
    if not video_root.is_dir():
        raise ReconstructionError(f"Video root not found: {video_root}")

    source_videos: SourceVideos = {}
    for class_name in CLASSES:
        source_dir = video_root / class_name / "videos"
        if not source_dir.is_dir():
            raise ReconstructionError(f"Source video directory not found: {source_dir}")

        entries = sorted(source_dir.iterdir(), key=lambda path: path.name)
        unexpected = [
            path for path in entries if not path.is_file() or path.suffix.lower() != ".mp4"
        ]
        if unexpected:
            preview = "\n".join(f"  - {path}" for path in unexpected[:10])
            suffix = "\n  - ..." if len(unexpected) > 10 else ""
            raise ReconstructionError(
                f"Only MP4 files are allowed in {source_dir}. Unexpected entries:\n"
                f"{preview}{suffix}"
            )

        videos_by_id: Dict[str, Path] = {}
        ids_by_casefold: Dict[str, str] = {}
        for video_path in entries:
            video_id = video_path.stem
            normalized_id = video_id.casefold()
            if video_id in videos_by_id or normalized_id in ids_by_casefold:
                previous_id = ids_by_casefold.get(normalized_id, video_id)
                previous_path = videos_by_id.get(previous_id)
                raise ReconstructionError(
                    "Duplicate source video stems detected:\n"
                    f"  - {previous_path}\n"
                    f"  - {video_path}"
                )
            videos_by_id[video_id] = video_path
            ids_by_casefold[normalized_id] = video_id
        source_videos[class_name] = videos_by_id

    return source_videos


def validate_reference_splits(reference_ids: ReferenceIds) -> None:
    """Reject split conflicts, case collisions, and cross-class reference IDs."""
    normalized_by_class: Dict[str, Dict[str, tuple]] = {}

    for class_name in CLASSES:
        seen: Dict[str, tuple] = {}
        for split_name in SPLITS:
            for video_id in reference_ids[class_name][split_name]:
                normalized_id = video_id.casefold()
                if normalized_id in seen:
                    previous_id, previous_split = seen[normalized_id]
                    raise ReconstructionError(
                        "Reference video ID conflict detected:\n"
                        f"  class={class_name}\n"
                        f"  id={previous_id!r}, split={previous_split}\n"
                        f"  id={video_id!r}, split={split_name}"
                    )
                seen[normalized_id] = (video_id, split_name)
        normalized_by_class[class_name] = seen

    overlap = set(normalized_by_class["fake"]).intersection(normalized_by_class["real"])
    if overlap:
        normalized_id = sorted(overlap)[0]
        fake_id, fake_split = normalized_by_class["fake"][normalized_id]
        real_id, real_split = normalized_by_class["real"][normalized_id]
        raise ReconstructionError(
            "Reference video ID appears in both classes:\n"
            f"  fake/{fake_split}/{fake_id}\n"
            f"  real/{real_split}/{real_id}"
        )


def validate_class_consistency(
    reference_ids: ReferenceIds, source_videos: SourceVideos
) -> None:
    """Reject source IDs duplicated across classes or found only in the wrong class."""
    source_by_casefold = {
        class_name: {video_id.casefold(): video_id for video_id in source_videos[class_name]}
        for class_name in CLASSES
    }
    source_overlap = set(source_by_casefold["fake"]).intersection(source_by_casefold["real"])
    if source_overlap:
        normalized_id = sorted(source_overlap)[0]
        fake_id = source_by_casefold["fake"][normalized_id]
        real_id = source_by_casefold["real"][normalized_id]
        raise ReconstructionError(
            "Source video ID appears in both classes:\n"
            f"  {source_videos['fake'][fake_id]}\n"
            f"  {source_videos['real'][real_id]}"
        )

    for class_name, other_class in (("fake", "real"), ("real", "fake")):
        for split_name in SPLITS:
            for video_id in reference_ids[class_name][split_name]:
                if video_id in source_videos[class_name]:
                    continue
                wrong_class_id = source_by_casefold[other_class].get(video_id.casefold())
                if wrong_class_id is not None:
                    raise ReconstructionError(
                        "Referenced video exists only in the wrong class:\n"
                        f"  reference={class_name}/{split_name}/{video_id}\n"
                        f"  source={source_videos[other_class][wrong_class_id]}"
                    )


def validate_destination_dirs(video_root: Path) -> None:
    """Require pre-existing split directories to be empty before reconstruction."""
    occupied: List[Path] = []
    for class_name in CLASSES:
        for split_name in SPLITS:
            split_dir = video_root / class_name / split_name
            if split_dir.exists():
                if not split_dir.is_dir():
                    raise ReconstructionError(
                        f"Expected a directory but found another entry: {split_dir}"
                    )
                occupied.extend(path for path in split_dir.iterdir())
    if occupied:
        preview = "\n".join(f"  - {path}" for path in occupied[:10])
        suffix = "\n  - ..." if len(occupied) > 10 else ""
        raise ReconstructionError(
            "Destination split directories must be empty before reconstruction:\n"
            f"{preview}{suffix}"
        )


def build_reconstruction_plan(
    video_root: Path,
    reference_ids: ReferenceIds,
    source_videos: SourceVideos,
) -> ReconstructionPlan:
    """Build a deterministic, JSON-serializable move/delete plan without mutations."""
    plan: ReconstructionPlan = {}

    for class_name in CLASSES:
        class_plan: Dict[str, List[dict]] = {split_name: [] for split_name in SPLITS}
        missing: List[dict] = []
        referenced_ids = set()

        for split_name in SPLITS:
            for video_id in reference_ids[class_name][split_name]:
                referenced_ids.add(video_id)
                source_path = source_videos[class_name].get(video_id)
                destination_path = video_root / class_name / split_name / f"{video_id}.mp4"
                if source_path is None:
                    missing.append(
                        {
                            "video_id": video_id,
                            "split": split_name,
                            "expected_source_path": str(
                                video_root / class_name / "videos" / f"{video_id}.mp4"
                            ),
                        }
                    )
                    continue
                class_plan[split_name].append(
                    {
                        "video_id": video_id,
                        "source_path": str(source_path),
                        "destination_path": str(destination_path),
                    }
                )

        class_plan["missing"] = missing
        class_plan["unmatched_to_delete"] = [
            {"video_id": video_id, "source_path": str(source_videos[class_name][video_id])}
            for video_id in sorted(set(source_videos[class_name]) - referenced_ids)
        ]
        plan[class_name] = class_plan

    return plan


def save_manifest(plan: ReconstructionPlan, video_root: Path) -> Path:
    """Atomically write the complete plan before any video mutation occurs."""
    manifest_path = video_root / MANIFEST_NAME
    temporary_path = video_root / f"{MANIFEST_NAME}.tmp"
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(plan, file, indent=2, ensure_ascii=False)
            file.write("\n")
        temporary_path.replace(manifest_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return manifest_path


def print_summary(
    reference_root: Path,
    video_root: Path,
    reference_ids: ReferenceIds,
    source_videos: SourceVideos,
    plan: ReconstructionPlan,
    manifest_path: Path,
    apply: bool,
) -> None:
    print("=" * 40)
    print("BioDeepAV split reconstruction")
    print("=" * 40)
    print(f"Reference root : {reference_root}")
    print(f"Video root     : {video_root}")
    print(f"Manifest       : {manifest_path}")

    for class_name in CLASSES:
        print(f"\n{class_name.upper()}")
        for split_name in SPLITS:
            print(
                f"  {split_name:<5} reference IDs : "
                f"{len(reference_ids[class_name][split_name])}"
            )

    print("\nSource videos:")
    for class_name in CLASSES:
        print(f"  {class_name}: {len(source_videos[class_name])}")

    print("\nMatched:")
    for class_name in CLASSES:
        for split_name in SPLITS:
            print(f"  {class_name}/{split_name:<5}: {len(plan[class_name][split_name])}")

    print("\nMissing referenced videos:")
    for class_name in CLASSES:
        print(f"  {class_name}: {len(plan[class_name]['missing'])}")
    for class_name in CLASSES:
        for item in plan[class_name]["missing"]:
            print("\n[MISSING]")
            print(f"class={class_name}")
            print(f"split={item['split']}")
            print(f"id={item['video_id']}")

    print("\nUnmatched source videos to DELETE:")
    for class_name in CLASSES:
        print(f"  {class_name}: {len(plan[class_name]['unmatched_to_delete'])}")
        for item in plan[class_name]["unmatched_to_delete"]:
            print(
                f"    [DELETE] id={item['video_id']} path={item['source_path']}"
            )

    print()
    if apply:
        print("APPLY requested: the validated plan will now be executed.")
    else:
        print("DRY RUN: no video files or dataset directories were modified.")
        print("Only the reconstruction manifest was created or updated.")


def _assert_source_path_is_safe(source_path: Path, expected_source_dir: Path) -> None:
    if source_path.parent.resolve() != expected_source_dir.resolve():
        raise ReconstructionError(f"Unsafe source path outside videos directory: {source_path}")
    if source_path.suffix.lower() != ".mp4":
        raise ReconstructionError(f"Unsafe non-MP4 source path: {source_path}")


def _preflight_apply(plan: ReconstructionPlan, video_root: Path) -> None:
    """Recheck every planned path immediately before the first mutation."""
    planned_sources = set()
    for class_name in CLASSES:
        expected_source_dir = video_root / class_name / "videos"
        for split_name in SPLITS:
            for item in plan[class_name][split_name]:
                source_path = Path(item["source_path"])
                destination_path = Path(item["destination_path"])
                _assert_source_path_is_safe(source_path, expected_source_dir)
                if not source_path.is_file():
                    raise ReconstructionError(f"Planned source disappeared: {source_path}")
                if destination_path.exists():
                    raise ReconstructionError(
                        f"Refusing to overwrite destination: {destination_path}"
                    )
                if source_path in planned_sources:
                    raise ReconstructionError(f"Source appears twice in plan: {source_path}")
                planned_sources.add(source_path)

        for item in plan[class_name]["unmatched_to_delete"]:
            source_path = Path(item["source_path"])
            _assert_source_path_is_safe(source_path, expected_source_dir)
            if not source_path.is_file():
                raise ReconstructionError(f"Planned deletion source disappeared: {source_path}")
            if source_path in planned_sources:
                raise ReconstructionError(f"Source appears twice in plan: {source_path}")
            planned_sources.add(source_path)


def apply_plan(plan: ReconstructionPlan, video_root: Path) -> None:
    """Move matched videos, delete unmatched source MP4s, then remove empty videos dirs."""
    _preflight_apply(plan, video_root)

    for class_name in CLASSES:
        for split_name in SPLITS:
            (video_root / class_name / split_name).mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        for split_name in SPLITS:
            for item in plan[class_name][split_name]:
                shutil.move(item["source_path"], item["destination_path"])

    for class_name in CLASSES:
        source_dir = video_root / class_name / "videos"
        for item in plan[class_name]["unmatched_to_delete"]:
            Path(item["source_path"]).unlink()
        if any(source_dir.iterdir()):
            raise ReconstructionError(f"Source directory is not empty after apply: {source_dir}")
        source_dir.rmdir()


def verify_final_dataset(plan: ReconstructionPlan, video_root: Path) -> None:
    """Verify and print direct MP4 counts in every reconstructed split."""
    print("\nFinal dataset:")
    for class_name in CLASSES:
        print(f"\n{class_name}:")
        total = 0
        for split_name in SPLITS:
            split_dir = video_root / class_name / split_name
            actual_count = sum(
                1
                for path in split_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".mp4"
            )
            expected_count = len(plan[class_name][split_name])
            if actual_count != expected_count:
                raise ReconstructionError(
                    f"Final count mismatch for {class_name}/{split_name}: "
                    f"expected {expected_count}, found {actual_count}"
                )
            total += actual_count
            print(f"  {split_name:<5}: {actual_count}")
        print(f"  total: {total}")

        source_dir = video_root / class_name / "videos"
        if source_dir.exists():
            raise ReconstructionError(f"Source directory still exists: {source_dir}")


def run(args: argparse.Namespace) -> None:
    video_root = args.video_root.expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()

    reference_ids = collect_reference_ids(reference_root)
    source_videos = collect_source_videos(video_root)
    validate_reference_splits(reference_ids)
    validate_class_consistency(reference_ids, source_videos)
    validate_destination_dirs(video_root)
    plan = build_reconstruction_plan(video_root, reference_ids, source_videos)

    manifest_path = save_manifest(plan, video_root)
    print_summary(
        reference_root,
        video_root,
        reference_ids,
        source_videos,
        plan,
        manifest_path,
        apply=args.apply,
    )

    if args.apply:
        apply_plan(plan, video_root)
        verify_final_dataset(plan, video_root)
        print("\nSplit reconstruction completed successfully.")


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except (ReconstructionError, OSError, json.JSONDecodeError) as error:
        print("=" * 40, file=sys.stderr)
        print("ABORTED: no further operations will be performed.", file=sys.stderr)
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
