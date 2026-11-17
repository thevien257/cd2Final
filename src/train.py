"""
train.py - Vòng lặp huấn luyện MLP
====================================
TV3: Huấn luyện mạng MLP bằng Cross-Entropy trên nhãn tối ưu.
Hỗ trợ warm-start (nhãn từ Exhaustive Search) và pseudo-label.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Optional
import time

from model import RoutingMLP, create_model
from dataset import create_dataset, normalize_features


def prepare_dataloaders(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42
) -> tuple:
    """
    Chuyển dữ liệu NumPy sang DataLoader PyTorch.
    
    Returns
    -------
    train_loader, val_loader, feat_mean, feat_std
    """
    # Chia train/val
    rng = np.random.default_rng(seed)
    n = len(features)
    indices = rng.permutation(n)
    val_size = int(n * val_split)
    
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    train_feat = features[train_idx]
    train_labels = labels[train_idx]
    val_feat = features[val_idx]
    val_labels = labels[val_idx]
    
    # Chuẩn hóa
    train_feat, mean, std = normalize_features(train_feat)
    val_feat, _, _ = normalize_features(val_feat, mean, std)
    
    # Chuyển sang Tensor
    train_X = torch.FloatTensor(train_feat)
    train_Y = torch.LongTensor(train_labels)
    val_X = torch.FloatTensor(val_feat)
    val_Y = torch.LongTensor(val_labels)
    
    train_ds = TensorDataset(train_X, train_Y)
    val_ds = TensorDataset(val_X, val_Y)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, mean, std


def train_epoch(
    model: RoutingMLP,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Chạy một epoch huấn luyện, trả về loss trung bình."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    criterion = nn.CrossEntropyLoss()
    
    for batch_X, batch_Y in train_loader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)  # shape (batch, M)
        
        optimizer.zero_grad()
        
        # Forward: probs shape (batch, M, K)
        logits = model.get_logits(batch_X)  # (batch, M, K)
        
        # Loss: Cross-Entropy cho mỗi luồng, trung bình qua M
        # logits: (batch, M, K) -> reshape (batch*M, K)
        # labels: (batch, M) -> reshape (batch*M,)
        B, M, K = logits.shape
        logits_flat = logits.view(B * M, K)
        labels_flat = batch_Y.view(B * M)
        
        loss = criterion(logits_flat, labels_flat)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def validate(
    model: RoutingMLP,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Đánh giá trên tập validation."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            logits = model.get_logits(batch_X)
            B, M, K = logits.shape
            
            logits_flat = logits.view(B * M, K)
            labels_flat = batch_Y.view(B * M)
            
            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()
            
            preds = torch.argmax(logits_flat, dim=1)
            correct += (preds == labels_flat).sum().item()
            total += labels_flat.numel()
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    
    return {'val_loss': avg_loss, 'val_accuracy': accuracy}


def train_model(
    config: Dict[str, Any],
    dataset: Dict[str, Any],
    save_dir: str = 'results',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Huấn luyện MLP đầy đủ.
    
    Parameters
    ----------
    config : dict
    dataset : dict
        Từ dataset.create_dataset().
    save_dir : str
        Thư mục lưu checkpoint.
    
    Returns
    -------
    result : dict
        'model' : RoutingMLP - Model đã train
        'train_losses' : list - Loss theo epoch
        'val_losses' : list
        'val_accuracies' : list
        'best_val_loss' : float
        'feat_mean', 'feat_std' : np.ndarray
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Device: {device}")
    
    ml_cfg = config['ml']
    feat_dim = dataset['features'].shape[1]
    
    # Chuẩn bị dữ liệu
    train_loader, val_loader, feat_mean, feat_std = prepare_dataloaders(
        dataset['features'], dataset['labels'],
        batch_size=ml_cfg['batch_size'],
        seed=config['seed']
    )
    
    # Tạo model
    model = create_model(config, feat_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=ml_cfg['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=20, factor=0.5, min_lr=1e-6
    )
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {total_params:,} tham số")
        print(f"Feature dim: {feat_dim}")
        print(f"Bắt đầu huấn luyện {ml_cfg['num_epochs']} epochs...\n")
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(ml_cfg['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_result = validate(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_result['val_loss'])
        val_accuracies.append(val_result['val_accuracy'])
        
        scheduler.step(val_result['val_loss'])
        
        # Early stopping
        if val_result['val_loss'] < best_val_loss:
            best_val_loss = val_result['val_loss']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_result['val_loss']:.4f}, "
                  f"val_acc={val_result['val_accuracy']:.3f}, "
                  f"lr={lr:.2e}")
        
        if patience_counter >= 40:
            if verbose:
                print(f"  Early stopping tại epoch {epoch+1}")
            break
    
    # Khôi phục model tốt nhất
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Lưu checkpoint
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'best_mlp.pt')
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'feat_dim': feat_dim,
        'feat_mean': feat_mean,
        'feat_std': feat_std,
    }, checkpoint_path)
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\nHoàn tất trong {elapsed:.1f}s")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Best val acc:  {max(val_accuracies):.3f}")
        print(f"Checkpoint: {checkpoint_path}")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss,
        'feat_mean': feat_mean,
        'feat_std': feat_std,
        'device': device,
    }


def load_model(checkpoint_path: str, device: torch.device = None) -> Dict[str, Any]:
    """Nạp model từ checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = create_model(ckpt['config'], ckpt['feat_dim']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    return {
        'model': model,
        'config': ckpt['config'],
        'feat_mean': ckpt['feat_mean'],
        'feat_std': ckpt['feat_std'],
        'device': device,
    }


if __name__ == '__main__':
    import yaml
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Giảm để test nhanh
    config['experiment']['num_train_instances'] = 50
    config['ml']['num_epochs'] = 50
    
    print("Sinh dữ liệu huấn luyện...")
    dataset = create_dataset(config, num_instances=50, seed=config['seed'])
    
    print("\nBắt đầu huấn luyện...")
    result = train_model(config, dataset, save_dir='results')
    
    print(f"\nTrain loss cuối: {result['train_losses'][-1]:.4f}")
    print(f"Val loss cuối:   {result['val_losses'][-1]:.4f}")
