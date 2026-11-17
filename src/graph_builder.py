"""
graph_builder.py - Module sinh đồ thị mạng truyền thông
========================================================
TV1: Tạo topology mạng ngẫu nhiên có trọng số dương, đảm bảo đồ thị liên thông.
Sinh M cặp nguồn-đích ngẫu nhiên (không trùng) làm luồng dữ liệu.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any


def build_graph(config: Dict[str, Any], rng: np.random.Generator = None) -> nx.Graph:
    """
    Sinh đồ thị Erdős–Rényi liên thông có trọng số cạnh dương.
    
    Parameters
    ----------
    config : dict
        Cấu hình mạng (num_nodes, edge_prob, weight_min, weight_max).
    rng : np.random.Generator, optional
        Bộ sinh số ngẫu nhiên (để tái lập).
    
    Returns
    -------
    G : nx.Graph
        Đồ thị liên thông có trọng số 'weight' trên mỗi cạnh.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    net_cfg = config['network']
    N = net_cfg['num_nodes']
    p = net_cfg['edge_prob']
    w_min = net_cfg['weight_min']
    w_max = net_cfg['weight_max']
    
    # Tạo đồ thị Erdős–Rényi, thử lại nếu không liên thông
    max_attempts = 50
    for attempt in range(max_attempts):
        # Dùng seed khác nhau mỗi lần thử
        seed_val = int(rng.integers(0, 2**31))
        G = nx.erdos_renyi_graph(N, p, seed=seed_val)
        
        if nx.is_connected(G):
            break
    else:
        # Fallback: tạo cây khung rồi thêm cạnh ngẫu nhiên
        G = nx.random_tree(N, seed=int(rng.integers(0, 2**31)))
        # Thêm thêm cạnh
        nodes = list(G.nodes())
        num_extra = int(N * (N - 1) * p / 2) - G.number_of_edges()
        for _ in range(max(0, num_extra)):
            u, v = rng.choice(nodes, size=2, replace=False)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
    
    # Gán trọng số dương cho mỗi cạnh
    for u, v in G.edges():
        G[u][v]['weight'] = float(rng.uniform(w_min, w_max))
    
    return G


def generate_flows(
    G: nx.Graph,
    M: int,
    w_min: float = 0.5,
    w_max: float = 2.0,
    rng: np.random.Generator = None
) -> List[Dict[str, Any]]:
    """
    Sinh M luồng dữ liệu với cặp nguồn-đích ngẫu nhiên và trọng số ưu tiên.
    
    Parameters
    ----------
    G : nx.Graph
        Đồ thị mạng liên thông.
    M : int
        Số luồng dữ liệu.
    w_min, w_max : float
        Khoảng sinh trọng số ưu tiên w_m.
    rng : np.random.Generator, optional
        Bộ sinh số ngẫu nhiên.
    
    Returns
    -------
    flows : list of dict
        Mỗi phần tử gồm {'source': int, 'dest': int, 'weight': float}.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    nodes = list(G.nodes())
    flows = []
    used_pairs = set()
    
    for _ in range(M):
        # Chọn cặp nguồn-đích khác nhau và chưa dùng
        max_try = 100
        for _ in range(max_try):
            s, d = rng.choice(nodes, size=2, replace=False)
            s, d = int(s), int(d)
            if (s, d) not in used_pairs and nx.has_path(G, s, d):
                used_pairs.add((s, d))
                break
        
        w_m = float(rng.uniform(w_min, w_max))
        flows.append({
            'source': s,
            'dest': d,
            'weight': w_m
        })
    
    return flows


def get_flow_weights(flows: List[Dict[str, Any]]) -> np.ndarray:
    """Trích xuất vector trọng số w từ danh sách flows."""
    return np.array([f['weight'] for f in flows])


if __name__ == '__main__':
    import yaml
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    rng = np.random.default_rng(config['seed'])
    
    G = build_graph(config, rng)
    print(f"Đồ thị: {G.number_of_nodes()} nút, {G.number_of_edges()} cạnh")
    print(f"Liên thông: {nx.is_connected(G)}")
    
    M = config['flows']['num_flows']
    flows = generate_flows(G, M, rng=rng)
    for i, f in enumerate(flows):
        print(f"  Luồng {i}: {f['source']} -> {f['dest']}, w = {f['weight']:.3f}")
