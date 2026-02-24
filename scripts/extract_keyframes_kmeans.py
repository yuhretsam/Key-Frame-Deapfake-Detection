import argparse
import os

from tqdm import tqdm

from src.data_preprocessing.kmeans_keyframes import KMeansKeyframeExtractor
from src.utils.io import ensure_dir, list_video_files


def parse_args():
    parser = argparse.ArgumentParser(description="Extract keyframes using K-Means.")
    parser.add_argument("--input_root", required=True, help="Root folder with fake/real splits")
    parser.add_argument("--output_root", default="data", help="Output folder for keyframes")
    parser.add_argument("--method", default="kmeans", help="Method name for output folder")
    parser.add_argument("--min_k", type=int, default=3)
    parser.add_argument("--max_k", type=int, default=15)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    args.method = args.method.lower()
    extractor = KMeansKeyframeExtractor()

    for class_name in ["fake", "real"]:
        input_class_dir = os.path.join(args.input_root, class_name)
        if not os.path.isdir(input_class_dir):
            continue

        split_dirs = [
            split for split in args.splits if os.path.isdir(os.path.join(input_class_dir, split))
        ]

        if split_dirs:
            for split in split_dirs:
                input_dir = os.path.join(input_class_dir, split)
                output_dir = os.path.join(args.output_root, args.method, class_name, split)
                ensure_dir(output_dir)
                video_files = list_video_files(input_dir)
                for video_path in tqdm(video_files, desc=f"{class_name}:{split}"):
                    video_id = os.path.splitext(os.path.basename(video_path))[0]
                    video_output_dir = os.path.join(output_dir, video_id)
                    extractor.extract_keyframes(
                        video_path,
                        video_output_dir,
                        min_k=args.min_k,
                        max_k=args.max_k,
                        frame_step=args.frame_step,
                        max_frames=args.max_frames,
                    )
        else:
            output_dir = os.path.join(args.output_root, args.method, class_name)
            ensure_dir(output_dir)
            video_files = list_video_files(input_class_dir)
            for video_path in tqdm(video_files, desc=f"{class_name}"):
                video_id = os.path.splitext(os.path.basename(video_path))[0]
                video_output_dir = os.path.join(output_dir, video_id)
                extractor.extract_keyframes(
                    video_path,
                    video_output_dir,
                    min_k=args.min_k,
                    max_k=args.max_k,
                    frame_step=args.frame_step,
                    max_frames=args.max_frames,
                )


if __name__ == "__main__":
    main()
