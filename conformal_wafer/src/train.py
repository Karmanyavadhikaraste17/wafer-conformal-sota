import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from .config import cfg
from .models import get_model
from .dataset import get_dataloaders

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=cfg.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if cfg.USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_depths = []

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        depths = batch['depth']

        with autocast('cuda', enabled=cfg.USE_AMP):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

        _, preds = torch.max(probs, 1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_depths.append(depths.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_depths = np.concatenate(all_depths, axis=0)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    macro_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    # Per-depth metrics
    depth_metrics = {}
    for d in sorted(np.unique(all_depths)):
        mask = (all_depths == d)
        if np.sum(mask) > 0:
            d_acc = accuracy_score(all_labels[mask], all_preds[mask])
            d_f1 = f1_score(all_labels[mask], all_preds[mask], average='macro', zero_division=0)
            depth_metrics[int(d)] = {'accuracy': d_acc, 'macro_f1': d_f1, 'count': int(np.sum(mask))}

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'depth_metrics': depth_metrics,
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs,
        'depths': all_depths
    }

def train_model(
    model_name: str,
    epochs: int = cfg.EPOCHS,
    lr: float = cfg.LR,
    weight_decay: float = cfg.WEIGHT_DECAY,
    label_smoothing: float = 0.0,
    patience: int = cfg.PATIENCE,
    save_path: str = None,
    loaders: dict = None,
    device: str = cfg.DEVICE
):
    """
    Trains a model with early stopping on validation macro-F1.
    Saves best checkpoint and returns training history and best model.
    """
    if loaders is None:
        loaders = get_dataloaders()

    model = get_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg.MIN_LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = GradScaler('cuda', enabled=cfg.USE_AMP)

    if save_path is None:
        os.makedirs(cfg.CHECKPOINTS_DIR, exist_ok=True)
        save_path = os.path.join(cfg.CHECKPOINTS_DIR, f"{model_name}_best.pt")

    best_macro_f1 = -1.0
    best_weights = None
    patience_counter = 0
    history = []

    print(f"\n=======================================================")
    print(f"Starting Training: {model_name.upper()} on {device}")
    print(f"Max Epochs: {epochs} | LR: {lr} | Label Smoothing: {label_smoothing}")
    print(f"=======================================================")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, loaders['train'], optimizer, criterion, scaler, device)
        scheduler.step()

        val_res = evaluate_model(model, loaders['val'], device)
        val_acc = val_res['accuracy']
        val_f1 = val_res['macro_f1']
        elapsed = time.time() - t0

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_macro_f1': val_f1,
            'time_s': elapsed
        })

        improved = ""
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                'model_state_dict': best_weights,
                'model_name': model_name,
                'epoch': epoch,
                'val_macro_f1': best_macro_f1,
                'val_acc': val_acc
            }, save_path)
            improved = " [SAVED BEST]"
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}%{improved}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s ({total_time/60:.2f} mins). Best Val Macro-F1: {best_macro_f1*100:.2f}%")

    # Load best weights
    model.load_state_dict(best_weights)
    return model, history, best_macro_f1
