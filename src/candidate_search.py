"""
candidate_search.py - Sinh L cấu hình ứng viên từ MLP và chọn nghiệm tốt nhất
================================================================================
TV3: Nhận xác suất từ MLP, sinh L cấu hình (argmax + sampling),
giải Water-Filling cho mỗi cấu hình, chọn cấu hình có J nhỏ nhất.
"""

import numpy as np
import torch
import time
from typing import Dict, Any, List, Optional

from model import RoutingMLP
from water_filling import solve_water_filling
from objective import compute_objective, compute_alpha
from dataset import extract_features, normalize_features


def generate_candidates(
    probs: np.ndarray,
    L: int,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Sinh L cấu hình routing từ ma trận xác suất.
    
    Parameters
    ----------
    probs : np.ndarray, shape (M, K)
        Xác suất Softmax cho mỗi luồng.
    L : int
        Số cấu hình ứng viên.
    rng : np.random.Generator, optional
    
    Returns
    -------
    candidates : np.ndarray, shape (L, M)
        Mỗi hàng là một cấu hình routing.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    M, K = probs.shape
    candidates = []
    seen = set()
    
    # Cấu hình 1: Argmax (tất định)
    argmax_routing = np.argmax(probs, axis=1)
    candidates.append(argmax_routing)
    seen.add(tuple(argmax_routing))
    
    # Cấu hình 2..L: Lấy mẫu theo phân phối Softmax
    max_attempts = L * 5  # Tránh lặp vô hạn
    attempts = 0
    
    while len(candidates) < L and attempts < max_attempts:
        routing = np.array([
            rng.choice(K, p=probs[m] / probs[m].sum())
            for m in range(M)
        ])
        key = tuple(routing)
        if key not in seen:
            candidates.append(routing)
            seen.add(key)
        attempts += 1
    
    # Nếu vẫn chưa đủ L, thêm random
    while len(candidates) < L:
        routing = rng.integers(0, K, size=M)
        candidates.append(routing)
    
    return np.array(candidates)


def evaluate_candidates(
    candidates: np.ndarray,
    path_gains: np.ndarray,
    all_paths: List[List[List[int]]],
    w: np.ndarray,
    beta: float,
    P_max: float
) -> Dict[str, Any]:
    """
    Đánh giá từng cấu hình ứng viên bằng Water-Filling.
    
    Parameters
    ----------
    candidates : np.ndarray, shape (L, M)
    path_gains : np.ndarray, shape (M, K)
    all_paths : list
    w : np.ndarray, shape (M,)
    beta : float
    P_max : float
    
    Returns
    -------
    result : dict
        'best_routing' : np.ndarray - Cấu hình tốt nhất
        'best_p' : np.ndarray - Phân bổ công suất tối ưu
        'best_J' : float - Giá trị J tốt nhất
        'all_J' : list - J cho mọi candidate
        'best_idx' : int - Chỉ số candidate tốt nhất
    """
    L = len(candidates)
    all_J = []
    all_p = []
    best_idx = 0
    best_J = float('inf')
    
    for i, routing in enumerate(candidates):
        alpha = compute_alpha(routing, path_gains)
        
        try:
            wf = solve_water_filling(alpha, w, P_max)
            obj = compute_objective(routing, wf['p'], w, path_gains, all_paths, beta)
            J = obj['J']
            all_J.append(J)
            all_p.append(wf['p'])
            
            if J < best_J:
                best_J = J
                best_idx = i
        except Exception:
            all_J.append(float('inf'))
            all_p.append(np.zeros(len(w)))
    
    return {
        'best_routing': candidates[best_idx].copy(),
        'best_p': all_p[best_idx].copy(),
        'best_J': best_J,
        'all_J': all_J,
        'best_idx': best_idx,
    }


def dnn_guided_search(
    model: RoutingMLP,
    features: np.ndarray,
    path_gains: np.ndarray,
    all_paths: List[List[List[int]]],
    w: np.ndarray,
    beta: float,
    P_max: float,
    L: int,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    device: torch.device = None,
    rng: np.random.Generator = None
) -> Dict[str, Any]:
    """
    Pipeline đầy đủ: MLP -> sinh L ứng viên -> đánh giá -> chọn tốt nhất.
    
    Parameters
    ----------
    model : RoutingMLP
    features : np.ndarray, shape (feat_dim,) - Đặc trưng RAW (chưa chuẩn hóa)
    path_gains, all_paths, w, beta, P_max : dữ liệu bài toán
    L : int - Số ứng viên
    feat_mean, feat_std : np.ndarray - Thống kê chuẩn hóa từ training
    device : torch.device
    rng : np.random.Generator
    
    Returns
    -------
    result : dict - Từ evaluate_candidates + thêm inference_time, eval_time
    """
    if device is None:
        device = torch.device('cpu')
    if rng is None:
        rng = np.random.default_rng()
    
    start = time.time()
    
    # Chuẩn hóa features
    feat_norm = (features - feat_mean) / feat_std
    feat_tensor = torch.FloatTensor(feat_norm).unsqueeze(0).to(device)
    
    # Suy luận MLP
    model.eval()
    with torch.no_grad():
        probs = model(feat_tensor)  # (1, M, K)
    probs_np = probs.cpu().numpy()[0]  # (M, K)
    inference_time = time.time() - start
    
    # Sinh ứng viên
    candidates = generate_candidates(probs_np, L, rng)
    
    # Đánh giá
    eval_start = time.time()
    result = evaluate_candidates(candidates, path_gains, all_paths, w, beta, P_max)
    eval_time = time.time() - eval_start
    
    result['inference_time'] = inference_time
    result['eval_time'] = eval_time
    result['total_time'] = time.time() - start
    result['probs'] = probs_np
    result['num_candidates'] = L
    
    return result


if __name__ == '__main__':
    import yaml
    from graph_builder import build_graph, generate_flows, get_flow_weights
    from path_generation import generate_all_candidates
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    rng = np.random.default_rng(config['seed'])
    M = config['flows']['num_flows']
    K = config['flows']['num_paths']
    
    # Sinh dữ liệu test
    G = build_graph(config, rng)
    flows = generate_flows(G, M, rng=rng)
    w = get_flow_weights(flows)
    cand = generate_all_candidates(G, flows, K, noise_std=0.3, rng=rng)
    
    # Test sinh ứng viên từ xác suất giả
    fake_probs = np.random.dirichlet(np.ones(K), size=M)
    print(f"Xác suất giả:\n{fake_probs}")
    
    candidates = generate_candidates(fake_probs, L=10, rng=rng)
    print(f"\n{len(candidates)} cấu hình ứng viên:")
    for i, r in enumerate(candidates):
        print(f"  Ứng viên {i}: {r}")
    
    # Đánh giá
    P_max = config['optimization']['P_max']
    beta = config['optimization']['beta']
    result = evaluate_candidates(candidates, cand['path_gains'], cand['all_paths'],
                                  w, beta, P_max)
    print(f"\nNghiệm tốt nhất: routing={result['best_routing']}, J={result['best_J']:.6f}")
