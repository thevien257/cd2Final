"""
baseline.py - Các phương pháp baseline so sánh
===============================================
TV1: Random, Shortest Path, Greedy, Exhaustive Search.
Tất cả baseline đều dùng chung bộ giải Water-Filling để đảm bảo so sánh công bằng.
"""

import numpy as np
import itertools
import time
from typing import Dict, Any, List

from water_filling import solve_water_filling
from objective import compute_alpha, compute_objective


def random_routing(M: int, K: int, rng: np.random.Generator = None) -> np.ndarray:
    """Mỗi luồng chọn ngẫu nhiên 1 trong K tuyến."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, K, size=M)


def shortest_path_routing(M: int, K: int) -> np.ndarray:
    """Mỗi luồng chọn tuyến ngắn nhất (tuyến 0, vì Yen trả về theo thứ tự tăng dần)."""
    return np.zeros(M, dtype=int)


def greedy_routing(
    path_gains: np.ndarray,
    all_paths: List[List[List[int]]],
    w: np.ndarray,
    beta: float,
    P_max: float
) -> np.ndarray:
    """
    Thuật toán Greedy: lần lượt chọn tuyến tốt nhất cho từng luồng,
    có xét tương tác tải cạnh với các luồng đã chọn.
    
    Với mỗi luồng m (theo thứ tự), thử tất cả K tuyến,
    giải Water-Filling cho toàn bộ luồng đã chọn + luồng hiện tại,
    chọn tuyến cho J nhỏ nhất.
    """
    M, K = path_gains.shape
    routing = np.zeros(M, dtype=int)
    
    for m in range(M):
        best_k = 0
        best_J = float('inf')
        
        for k in range(K):
            routing[m] = k
            # Tính alpha cho các luồng đã chọn (0..m)
            alpha_partial = np.array([
                path_gains[i, routing[i]] for i in range(m + 1)
            ])
            w_partial = w[:m + 1]
            
            # Giải Water-Filling cho m+1 luồng đầu
            try:
                wf = solve_water_filling(alpha_partial, w_partial, P_max)
                p_partial = wf['p']
                
                # Tính J cho phần đã chọn
                obj = compute_objective(
                    routing[:m + 1], p_partial, w_partial,
                    path_gains[:m + 1], all_paths[:m + 1], beta
                )
                
                if obj['J'] < best_J:
                    best_J = obj['J']
                    best_k = k
            except Exception:
                continue
        
        routing[m] = best_k
    
    return routing


def exhaustive_search(
    path_gains: np.ndarray,
    all_paths: List[List[List[int]]],
    w: np.ndarray,
    beta: float,
    P_max: float,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Duyệt toàn bộ K^M cấu hình tuyến và chọn cấu hình tốt nhất.
    Chỉ dùng cho mạng nhỏ (M*K nhỏ) làm chuẩn đối chiếu.
    
    Returns
    -------
    result : dict
        'routing' : np.ndarray - Cấu hình tốt nhất
        'p' : np.ndarray - Phân bổ công suất tốt nhất
        'J' : float - Giá trị hàm mục tiêu tốt nhất
        'num_configs' : int - Tổng số cấu hình đã duyệt
        'time' : float - Thời gian chạy (giây)
    """
    M, K = path_gains.shape
    total_configs = K ** M
    
    if verbose:
        print(f"Exhaustive Search: duyệt {total_configs} cấu hình (M={M}, K={K})")
    
    best_routing = None
    best_p = None
    best_J = float('inf')
    
    start_time = time.time()
    
    for idx, combo in enumerate(itertools.product(range(K), repeat=M)):
        routing = np.array(combo)
        alpha = np.array([path_gains[m, routing[m]] for m in range(M)])
        
        try:
            wf = solve_water_filling(alpha, w, P_max)
            obj = compute_objective(routing, wf['p'], w, path_gains, all_paths, beta)
            
            if obj['J'] < best_J:
                best_J = obj['J']
                best_routing = routing.copy()
                best_p = wf['p'].copy()
        except Exception:
            continue
        
        if verbose and (idx + 1) % 100 == 0:
            print(f"  Đã duyệt {idx+1}/{total_configs}, J tốt nhất = {best_J:.6f}")
    
    elapsed = time.time() - start_time
    
    return {
        'routing': best_routing,
        'p': best_p,
        'J': best_J,
        'num_configs': total_configs,
        'time': elapsed,
    }


def run_baseline(
    method: str,
    path_gains: np.ndarray,
    all_paths: List[List[List[int]]],
    w: np.ndarray,
    beta: float,
    P_max: float,
    rng: np.random.Generator = None
) -> Dict[str, Any]:
    """
    Chạy một phương pháp baseline và trả về kết quả đầy đủ.
    
    Parameters
    ----------
    method : str
        'random', 'shortest', 'greedy', 'exhaustive'
    
    Returns
    -------
    result : dict
        'method', 'routing', 'p', 'J', 'utility', 'time', ...
    """
    M, K = path_gains.shape
    start = time.time()
    
    if method == 'random':
        routing = random_routing(M, K, rng)
    elif method == 'shortest':
        routing = shortest_path_routing(M, K)
    elif method == 'greedy':
        routing = greedy_routing(path_gains, all_paths, w, beta, P_max)
    elif method == 'exhaustive':
        ex_result = exhaustive_search(path_gains, all_paths, w, beta, P_max)
        elapsed = time.time() - start
        alpha = np.array([path_gains[m, ex_result['routing'][m]] for m in range(M)])
        return {
            'method': method,
            'routing': ex_result['routing'],
            'p': ex_result['p'],
            'J': ex_result['J'],
            'alpha': alpha,
            'utility': -ex_result['J'] + compute_objective(
                ex_result['routing'], ex_result['p'], w, path_gains, all_paths, beta
            )['edge_penalty'],
            'time': elapsed,
            'num_configs': ex_result['num_configs'],
        }
    else:
        raise ValueError(f"Phương pháp không hợp lệ: {method}")
    
    # Giải Water-Filling cho routing đã chọn
    alpha = np.array([path_gains[m, routing[m]] for m in range(M)])
    wf = solve_water_filling(alpha, w, P_max)
    obj = compute_objective(routing, wf['p'], w, path_gains, all_paths, beta)
    elapsed = time.time() - start
    
    return {
        'method': method,
        'routing': routing,
        'p': wf['p'],
        'J': obj['J'],
        'alpha': alpha,
        'utility': obj['utility'],
        'edge_penalty': obj['edge_penalty'],
        'time': elapsed,
    }


if __name__ == '__main__':
    import yaml
    from graph_builder import build_graph, generate_flows, get_flow_weights
    from path_generation import generate_all_candidates
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    rng = np.random.default_rng(config['seed'])
    G = build_graph(config, rng)
    M = config['flows']['num_flows']
    K = config['flows']['num_paths']
    beta = config['optimization']['beta']
    P_max = config['optimization']['P_max']
    
    flows = generate_flows(G, M, rng=rng)
    w = get_flow_weights(flows)
    cand = generate_all_candidates(G, flows, K, noise_std=0.3, rng=rng)
    
    print("=" * 60)
    print("SO SÁNH CÁC BASELINE")
    print("=" * 60)
    
    for method in ['random', 'shortest', 'greedy', 'exhaustive']:
        result = run_baseline(method, cand['path_gains'], cand['all_paths'],
                              w, beta, P_max, rng)
        print(f"\n{method.upper():>12}: J = {result['J']:.6f}, "
              f"time = {result['time']:.4f}s, "
              f"routing = {result['routing']}")
