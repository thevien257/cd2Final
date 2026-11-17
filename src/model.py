"""
model.py - Kiến trúc mạng MLP Softmax M × K
=============================================
TV3: Mạng nơ-ron đa tầng dự đoán xác suất chọn tuyến cho mỗi luồng.
Đầu ra: ma trận xác suất π ∈ R^(M×K), Softmax theo chiều K.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class RoutingMLP(nn.Module):
    """
    MLP dự đoán xác suất chọn tuyến cho bài toán định tuyến đa luồng.
    
    Kiến trúc:
        Input (feat_dim) -> [Linear -> BatchNorm -> ReLU -> Dropout] x num_layers
                         -> Linear (M * K) -> Reshape (M, K) -> Softmax (dim K)
    
    Parameters
    ----------
    feat_dim : int
        Chiều vector đặc trưng đầu vào.
    M : int
        Số luồng dữ liệu.
    K : int
        Số tuyến ứng viên mỗi luồng.
    hidden_dims : list of int
        Số nơ-ron mỗi lớp ẩn.
    dropout : float
        Tỉ lệ dropout.
    """
    
    def __init__(
        self,
        feat_dim: int,
        M: int,
        K: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.M = M
        self.K = K
        self.feat_dim = feat_dim
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Xây dựng các lớp ẩn
        layers = []
        in_dim = feat_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        self.hidden = nn.Sequential(*layers)
        
        # Lớp đầu ra: M * K logits
        self.output_layer = nn.Linear(in_dim, M * K)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, feat_dim)
        
        Returns
        -------
        probs : torch.Tensor, shape (batch, M, K)
            Xác suất Softmax cho mỗi luồng.
        """
        h = self.hidden(x)
        logits = self.output_layer(h)  # (batch, M*K)
        logits = logits.view(-1, self.M, self.K)  # (batch, M, K)
        probs = F.softmax(logits, dim=2)  # Softmax theo chiều K
        return probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dự đoán routing tối ưu (argmax).
        
        Returns
        -------
        routing : torch.Tensor, shape (batch, M), dtype long
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
            routing = torch.argmax(probs, dim=2)
        return routing
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về logits trước Softmax, shape (batch, M, K)."""
        h = self.hidden(x)
        logits = self.output_layer(h)
        return logits.view(-1, self.M, self.K)


def create_model(config: dict, feat_dim: int) -> RoutingMLP:
    """Tạo model từ config."""
    M = config['flows']['num_flows']
    K = config['flows']['num_paths']
    hidden_dims = config['ml']['hidden_dims']
    dropout = config['ml'].get('dropout', 0.1)
    
    model = RoutingMLP(
        feat_dim=feat_dim,
        M=M,
        K=K,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return model


if __name__ == '__main__':
    # Test kiến trúc
    feat_dim = 84  # Ví dụ: M=4, K=3, 6 features/path + 6 global
    M, K = 4, 3
    batch_size = 8
    
    model = RoutingMLP(feat_dim, M, K, hidden_dims=[128, 64], dropout=0.1)
    print(model)
    print(f"\nTổng tham số: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    x = torch.randn(batch_size, feat_dim)
    probs = model(x)
    print(f"\nĐầu vào:  {x.shape}")
    print(f"Đầu ra:   {probs.shape}")
    print(f"Tổng prob theo K: {probs.sum(dim=2)}")  # Phải ≈ 1.0
    
    # Test predict
    routing = model.predict(x)
    print(f"Routing:  {routing.shape}")
    print(f"Routing mẫu: {routing[0]}")
