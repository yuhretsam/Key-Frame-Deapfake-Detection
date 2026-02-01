import argparse
import os

from tqdm import tqdm

from src.data.optical_flow_keyframes import OpticalFlowKeyframeExtractor
from src.utils.io import ensure_dir, list_video_files


def parse_args():
    parser = argparse.ArgumentParser(description="Extract keyframes using Optical Flow.")
    parser.add_argument("--input_root", required=True, help="Root folder with fake/real splits")
    parser.add_argument("--output_root", default="data", help="Output folder for keyframes")
    parser.add_argument("--method", default="opticalflow", help="Method name for output folder")
    parser.add_argument("--threshold_ratio", type=float, default=0.3)
    parser.add_argument("--lambda_weight", type=float, default=1.0)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    args.method = args.method.lower()
    extractor = OpticalFlowKeyframeExtractor(
        lambda_weight=args.lambda_weight, threshold_ratio=args.threshold_ratio
    )

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
                    video_dir = os.path.join(output_dir, video_id)
                    ensure_dir(video_dir)
                    extractor.extract_keyframes(video_path, video_dir)
        else:
            output_dir = os.path.join(args.output_root, args.method, class_name)
            ensure_dir(output_dir)
            video_files = list_video_files(input_class_dir)
            for video_path in tqdm(video_files, desc=f"{class_name}"):
                video_id = os.path.splitext(os.path.basename(video_path))[0]
                video_dir = os.path.join(output_dir, video_id)
                ensure_dir(video_dir)
                extractor.extract_keyframes(video_path, video_dir)


if __name__ == "__main__":
    main()
