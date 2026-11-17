"""
dataset.py - Trích xuất đặc trưng và tạo bộ dữ liệu huấn luyện
================================================================
TV3: Chuẩn hóa vector đặc trưng mạng cho MLP.
Sinh nhiều kịch bản mạng, lấy nhãn tối ưu từ Exhaustive Search.
"""

import numpy as np
import time
from typing import Dict, Any, Tuple, List, Optional

from graph_builder import build_graph, generate_flows, get_flow_weights
from path_generation import generate_all_candidates
from baseline import exhaustive_search


def extract_features(
    path_gains: np.ndarray,
    path_costs: np.ndarray,
    path_hops: np.ndarray,
    w: np.ndarray,
    P_max: float
) -> np.ndarray:
    """
    Trích xuất vector đặc trưng z cho một trạng thái mạng.
    
    Đặc trưng cho mỗi luồng m và tuyến k:
        - g_mk (độ lợi kênh)
        - cost_mk (chi phí tuyến)
        - hops_mk (số bước nhảy)
        - w_m (trọng số ưu tiên, lặp lại K lần)
        - w_m * g_mk (tích trọng số × độ lợi)
        - g_mk / max(g_m) (độ lợi chuẩn hóa theo luồng)
    
    Parameters
    ----------
    path_gains : np.ndarray, shape (M, K)
    path_costs : np.ndarray, shape (M, K)
    path_hops  : np.ndarray, shape (M, K)
    w : np.ndarray, shape (M,)
    P_max : float
    
    Returns
    -------
    features : np.ndarray, shape (M * K * num_feat_per_path,)
        Vector đặc trưng phẳng.
    """
    M, K = path_gains.shape
    
    feat_list = []
    for m in range(M):
        g_max = np.max(path_gains[m]) + 1e-10
        c_max = np.max(path_costs[m]) + 1e-10
        h_max = np.max(path_hops[m]) + 1e-10
        
        for k in range(K):
            feat_list.extend([
                path_gains[m, k],                  # g_mk
                path_gains[m, k] / g_max,          # g_mk chuẩn hóa
                path_costs[m, k] / c_max,          # cost chuẩn hóa
                path_hops[m, k] / h_max,           # hops chuẩn hóa
                w[m],                               # trọng số ưu tiên
                w[m] * path_gains[m, k],            # w_m * g_mk
            ])
    
    # Thêm đặc trưng toàn cục
    feat_list.extend([
        P_max,
        float(M),
        float(K),
        float(np.mean(path_gains)),
        float(np.std(path_gains)),
        float(np.mean(w)),
    ])
    
    return np.array(feat_list, dtype=np.float32)


def create_dataset(
    config: Dict[str, Any],
    num_instances: int,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Sinh bộ dữ liệu huấn luyện: nhiều kịch bản mạng + nhãn tối ưu.
    
    Parameters
    ----------
    config : dict
    num_instances : int
    seed : int
    verbose : bool
    
    Returns
    -------
    dataset : dict
        'features' : np.ndarray, shape (num_instances, feat_dim)
        'labels'   : np.ndarray, shape (num_instances, M) - routing tối ưu
        'objectives' : np.ndarray, shape (num_instances,) - J* tối ưu
        'metadata' : list of dict - thông tin từng instance
    """
    rng = np.random.default_rng(seed)
    
    M = config['flows']['num_flows']
    K = config['flows']['num_paths']
    P_max = config['optimization']['P_max']
    beta = config['optimization']['beta']
    noise_std = config['experiment'].get('gain_noise_std', 0.3)
    w_min = config['experiment'].get('w_min', 0.5)
    w_max = config['experiment'].get('w_max', 2.0)
    
    features_list = []
    labels_list = []
    objectives_list = []
    metadata_list = []
    
    start_time = time.time()
    
    for i in range(num_instances):
        # Sinh đồ thị mới
        G = build_graph(config, rng)
        flows = generate_flows(G, M, w_min, w_max, rng)
        w = get_flow_weights(flows)
        
        # Sinh tuyến ứng viên
        cand = generate_all_candidates(G, flows, K, noise_std, rng)
        
        # Trích xuất đặc trưng
        feat = extract_features(
            cand['path_gains'], cand['path_costs'],
            cand['path_hops'], w, P_max
        )
        
        # Tìm nhãn tối ưu bằng Exhaustive Search
        ex = exhaustive_search(cand['path_gains'], cand['all_paths'], w, beta, P_max)
        
        features_list.append(feat)
        labels_list.append(ex['routing'])
        objectives_list.append(ex['J'])
        
        metadata_list.append({
            'path_gains': cand['path_gains'].copy(),
            'path_costs': cand['path_costs'].copy(),
            'path_hops': cand['path_hops'].copy(),
            'all_paths': cand['all_paths'],
            'w': w.copy(),
            'optimal_p': ex['p'].copy() if ex['p'] is not None else None,
        })
        
        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (num_instances - i - 1)
            print(f"  Sinh instance {i+1}/{num_instances}, "
                  f"J* = {ex['J']:.4f}, "
                  f"elapsed = {elapsed:.1f}s, ETA = {eta:.1f}s")
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    objectives = np.array(objectives_list)
    
    if verbose:
        total_time = time.time() - start_time
        print(f"\nHoàn tất: {num_instances} instances trong {total_time:.1f}s")
        print(f"Feature dim: {features.shape[1]}")
        print(f"J* trung bình: {np.mean(objectives):.4f} ± {np.std(objectives):.4f}")
    
    return {
        'features': features,
        'labels': labels,
        'objectives': objectives,
        'metadata': metadata_list,
    }


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chuẩn hóa z-score cho ma trận đặc trưng.
    
    Returns
    -------
    normalized, mean, std
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
        std[std < 1e-8] = 1.0  # Tránh chia cho 0
    
    normalized = (features - mean) / std
    return normalized, mean, std


if __name__ == '__main__':
    import yaml
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Sinh bộ dữ liệu nhỏ để test
    dataset = create_dataset(config, num_instances=20, seed=42, verbose=True)
    
    print(f"\nFeatures shape: {dataset['features'].shape}")
    print(f"Labels shape:   {dataset['labels'].shape}")
    print(f"Labels mẫu:\n{dataset['labels'][:5]}")
    
    # Chuẩn hóa
    feat_norm, mean, std = normalize_features(dataset['features'])
    print(f"\nSau chuẩn hóa - mean ≈ 0: {np.abs(np.mean(feat_norm, axis=0)).max():.6f}")
    print(f"Sau chuẩn hóa - std  ≈ 1: {np.abs(np.std(feat_norm, axis=0) - 1).max():.6f}")
