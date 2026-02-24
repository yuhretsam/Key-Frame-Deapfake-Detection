import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_preprocessing.keyframe_dataset import VideoKeyframeDataset, load_video_paths_and_labels
from src.models.cnn_lstm import CnnLstm
from src.training.train import train_model, _run_epoch
from src.utils.io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN+LSTM on keyframe sequences.")
    parser.add_argument("--data_root", default="data", help="Root folder of keyframe dataset")
    parser.add_argument("--method", default=None, help="Keyframe method folder name")
    parser.add_argument("--class_folders", nargs="+", default=["fake", "real"])
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_cnn", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve_data_root(data_root: str, method: str = None) -> str:
    return os.path.join(data_root, method) if method else data_root


def _has_split_folders(base_dir: str, class_folders: Tuple[str, ...]) -> bool:
    for class_name in class_folders:
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(os.path.join(class_dir, "train")):
            return True
    return False


def _random_split(paths, labels, val_ratio: float, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths))
    rng.shuffle(indices)
    test_size = int(len(indices) * test_ratio)
    val_size = int(len(indices) * val_ratio)
    test_idx = indices[:test_size]
    val_idx = indices[test_size : test_size + val_size]
    train_idx = indices[test_size + val_size :]

    def subset(idx):
        return [paths[i] for i in idx], labels[idx]

    return subset(train_idx), subset(val_idx), subset(test_idx)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.method:
        args.method = args.method.lower()
    args.model = args.model.lower()

    base_dir = _resolve_data_root(args.data_root, args.method)
    class_folders = args.class_folders

    if _has_split_folders(base_dir, class_folders):
        train_paths, train_labels, _ = load_video_paths_and_labels(base_dir, class_folders, "train")
        val_paths, val_labels, _ = load_video_paths_and_labels(base_dir, class_folders, "val")
        test_paths, test_labels, _ = load_video_paths_and_labels(base_dir, class_folders, "test")
    else:
        all_paths, all_labels, _ = load_video_paths_and_labels(base_dir, class_folders, None)
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = _random_split(
            all_paths, all_labels, args.val_ratio, args.test_ratio, args.seed
        )

    train_dataset = VideoKeyframeDataset(
        train_paths, train_labels, args.max_seq_length, img_size=args.img_size
    )
    val_dataset = VideoKeyframeDataset(
        val_paths, val_labels, args.max_seq_length, img_size=args.img_size
    )
    test_dataset = VideoKeyframeDataset(
        test_paths, test_labels, args.max_seq_length, img_size=args.img_size
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = CnnLstm(
        num_classes=len(class_folders), backbone=args.model, freeze_cnn=args.freeze_cnn
    ).to(device)

    model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    ensure_dir(args.output_dir)
    out_path = os.path.join(args.output_dir, f"best_{args.model}.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Saved model: {out_path}")

    if len(test_dataset) > 0:
        criterion = torch.nn.CrossEntropyLoss()
        test_loss, test_acc, test_metrics = _run_epoch(
            model, test_loader, criterion, optimizer=None, device=device, train=False
        )
        print(
            f"[Test]  Loss: {test_loss:.4f} | Acc: {test_acc:.4f} "
            f"| P: {test_metrics.get('precision', 0):.4f} "
            f"| R: {test_metrics.get('recall', 0):.4f} "
            f"| F1: {test_metrics.get('f1', 0):.4f} "
            f"| AUC: {test_metrics.get('auc', 0):.4f} "
            f"| EER: {test_metrics.get('eer', 0):.4f}"
        )


if __name__ == "__main__":
    main()
