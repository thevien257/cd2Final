"""
path_generation.py - Sinh tập tuyến ứng viên bằng thuật toán Yen K-Shortest Paths
==================================================================================
TV1: Với mỗi luồng (nguồn, đích), tìm K đường đi ngắn nhất không lặp nút.
Tính độ lợi hiệu dụng g_mk cho mỗi tuyến.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional


def find_k_shortest_paths(
    G: nx.Graph,
    source: int,
    target: int,
    K: int,
    weight: str = 'weight'
) -> List[List[int]]:
    """
    Tìm K đường đi đơn ngắn nhất từ source đến target.
    
    Sử dụng nx.shortest_simple_paths (Yen's algorithm).
    
    Parameters
    ----------
    G : nx.Graph
    source, target : int
    K : int
        Số tuyến tối đa cần tìm.
    weight : str
        Tên thuộc tính trọng số cạnh.
    
    Returns
    -------
    paths : list of list of int
        Danh sách các đường đi (mỗi đường là danh sách nút).
        Có thể ít hơn K nếu đồ thị không đủ tuyến.
    """
    paths = []
    try:
        for i, path in enumerate(nx.shortest_simple_paths(G, source, target, weight=weight)):
            paths.append(path)
            if i + 1 >= K:
                break
    except nx.NetworkXNoPath:
        pass
    
    return paths


def compute_path_cost(G: nx.Graph, path: List[int], weight: str = 'weight') -> float:
    """Tổng trọng số cạnh trên một đường đi (chi phí định tuyến)."""
    cost = 0.0
    for i in range(len(path) - 1):
        cost += G[path[i]][path[i + 1]][weight]
    return cost


def compute_path_gain(
    G: nx.Graph,
    path: List[int],
    noise_std: float = 0.0,
    rng: np.random.Generator = None
) -> float:
    """
    Tính độ lợi hiệu dụng g_mk cho một tuyến.
    
    g_mk = 1 / (tổng trọng số cạnh) * exp(noise)
    
    Trọng số cạnh lớn = đường đi dài/tốn kém = độ lợi kênh thấp.
    Nhiễu log-normal mô phỏng biến động kênh truyền.
    
    Parameters
    ----------
    G : nx.Graph
    path : list of int
    noise_std : float
        Độ lệch chuẩn nhiễu log-normal (0 = không nhiễu).
    rng : np.random.Generator, optional
    
    Returns
    -------
    gain : float
        Độ lợi hiệu dụng dương.
    """
    cost = compute_path_cost(G, path)
    if cost <= 0:
        cost = 1e-6
    
    gain = 1.0 / cost
    
    if noise_std > 0 and rng is not None:
        noise = rng.normal(0, noise_std)
        gain *= np.exp(noise)
    
    return max(gain, 1e-10)  # Đảm bảo dương


def get_path_edges(path: List[int]) -> List[tuple]:
    """Trả về danh sách các cạnh (u, v) trên đường đi, chuẩn hóa u < v."""
    edges = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edges.append((min(u, v), max(u, v)))
    return edges


def generate_all_candidates(
    G: nx.Graph,
    flows: List[Dict[str, Any]],
    K: int,
    noise_std: float = 0.0,
    rng: np.random.Generator = None
) -> Dict[str, Any]:
    """
    Sinh tập tuyến ứng viên đầy đủ cho tất cả các luồng.
    
    Parameters
    ----------
    G : nx.Graph
    flows : list of dict
        Danh sách luồng từ graph_builder.generate_flows().
    K : int
        Số tuyến ứng viên tối đa mỗi luồng.
    noise_std : float
        Nhiễu log-normal cho độ lợi kênh.
    rng : np.random.Generator, optional
    
    Returns
    -------
    result : dict
        'all_paths' : list of list of list[int]
            all_paths[m][k] = danh sách nút tuyến k của luồng m
        'path_gains' : np.ndarray shape (M, K)
            path_gains[m][k] = g_mk, độ lợi kênh tuyến k luồng m
            (padding = 0.0 nếu luồng m có ít hơn K tuyến)
        'path_costs' : np.ndarray shape (M, K)
            path_costs[m][k] = tổng trọng số cạnh
        'path_hops'  : np.ndarray shape (M, K)
            path_hops[m][k] = số bước nhảy (số cạnh)
        'path_mask'  : np.ndarray shape (M, K) bool
            True nếu tuyến k của luồng m tồn tại
    """
    if rng is None:
        rng = np.random.default_rng()
    
    M = len(flows)
    all_paths = []
    path_gains = np.zeros((M, K))
    path_costs = np.zeros((M, K))
    path_hops = np.zeros((M, K), dtype=int)
    path_mask = np.zeros((M, K), dtype=bool)
    
    for m, flow in enumerate(flows):
        paths = find_k_shortest_paths(G, flow['source'], flow['dest'], K)
        
        # Padding nếu ít hơn K tuyến
        while len(paths) < K:
            # Lặp lại tuyến cuối cùng (với nhiễu khác) nếu cần
            if len(paths) > 0:
                paths.append(paths[-1])
            else:
                raise ValueError(
                    f"Không tìm được đường đi từ {flow['source']} đến {flow['dest']}"
                )
        
        flow_paths = []
        for k in range(K):
            path = paths[k]
            flow_paths.append(path)
            path_gains[m, k] = compute_path_gain(G, path, noise_std, rng)
            path_costs[m, k] = compute_path_cost(G, path)
            path_hops[m, k] = len(path) - 1
            path_mask[m, k] = True
        
        all_paths.append(flow_paths)
    
    return {
        'all_paths': all_paths,
        'path_gains': path_gains,
        'path_costs': path_costs,
        'path_hops': path_hops,
        'path_mask': path_mask,
    }


if __name__ == '__main__':
    import yaml
    from graph_builder import build_graph, generate_flows
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    rng = np.random.default_rng(config['seed'])
    G = build_graph(config, rng)
    
    M = config['flows']['num_flows']
    K = config['flows']['num_paths']
    flows = generate_flows(G, M, rng=rng)
    
    cand = generate_all_candidates(G, flows, K, noise_std=0.3, rng=rng)
    
    print(f"Số luồng: {M}, Số tuyến/luồng: {K}")
    print(f"Ma trận độ lợi kênh g_mk:\n{cand['path_gains']}")
    print(f"Số bước nhảy:\n{cand['path_hops']}")
    
    for m in range(M):
        print(f"\nLuồng {m}: {flows[m]['source']} -> {flows[m]['dest']}")
        for k in range(K):
            path = cand['all_paths'][m][k]
            print(f"  Tuyến {k}: {path}, gain={cand['path_gains'][m,k]:.6f}, "
                  f"cost={cand['path_costs'][m,k]:.2f}, hops={cand['path_hops'][m,k]}")
