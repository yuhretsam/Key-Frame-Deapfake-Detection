from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.training.metrics import compute_metrics


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

    epoch_loss = running_loss / max(len(dataloader.dataset), 1)
    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
    metrics = compute_metrics(y_true, y_pred, y_prob) if y_true else {}
    return epoch_loss, acc, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-4,
    patience: int = 10,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.2, verbose=True)

    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        train_loss, train_acc, train_metrics = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_metrics = _run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"[Train] Loss: {train_loss:.4f} | Acc: {train_acc:.4f} "
            f"| P: {train_metrics.get('precision', 0):.4f} "
            f"| R: {train_metrics.get('recall', 0):.4f} "
            f"| F1: {train_metrics.get('f1', 0):.4f} "
            f"| AUC: {train_metrics.get('auc', 0):.4f} "
            f"| EER: {train_metrics.get('eer', 0):.4f}"
        )
        print(
            f"[Val]   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} "
            f"| P: {val_metrics.get('precision', 0):.4f} "
            f"| R: {val_metrics.get('recall', 0):.4f} "
            f"| F1: {val_metrics.get('f1', 0):.4f} "
            f"| AUC: {val_metrics.get('auc', 0):.4f} "
            f"| EER: {val_metrics.get('eer', 0):.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        scheduler.step(val_acc)

    if best_state:
        model.load_state_dict(best_state)
    return model
