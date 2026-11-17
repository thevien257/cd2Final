"""
objective.py - Hàm mục tiêu J(x, p) và các thành phần
======================================================
TV2: Tính giá trị hàm mục tiêu, độ lợi hiệu dụng alpha, phạt cạnh.

Hàm mục tiêu:
    J(x,p) = -Σ w_m * ln(1 + α_m * p_m) + β * Σ_e (load_e)^2

Trong đó:
    α_m(x) = Σ_k g_mk * x_mk   (độ lợi hiệu dụng khi chọn tuyến)
    load_e = Σ_m Σ_{k: e ∈ P_mk} x_mk  (số luồng dùng chung cạnh e)
"""

import numpy as np
from typing import List, Dict, Any, Optional


def compute_alpha(
    routing: np.ndarray,
    path_gains: np.ndarray
) -> np.ndarray:
    """
    Tính vector độ lợi hiệu dụng α_m cho mỗi luồng.
    
    Parameters
    ----------
    routing : np.ndarray, shape (M,), dtype int
        routing[m] = k, chỉ số tuyến được chọn cho luồng m.
    path_gains : np.ndarray, shape (M, K)
        path_gains[m][k] = g_mk.
    
    Returns
    -------
    alpha : np.ndarray, shape (M,)
        alpha[m] = g_{m, routing[m]}.
    """
    M = len(routing)
    alpha = np.array([path_gains[m, routing[m]] for m in range(M)])
    return alpha


def compute_utility(
    alpha: np.ndarray,
    p: np.ndarray,
    w: np.ndarray
) -> float:
    """
    Tính tổng độ hữu dụng truyền dẫn (càng lớn càng tốt).
    
    utility = Σ w_m * ln(1 + α_m * p_m)
    """
    return float(np.sum(w * np.log(1.0 + alpha * p)))


def compute_edge_penalty(
    routing: np.ndarray,
    all_paths: List[List[List[int]]],
    beta: float
) -> float:
    """
    Tính thành phần phạt dùng chung cạnh.
    
    penalty = β * Σ_e (load_e)^2
    
    Parameters
    ----------
    routing : np.ndarray, shape (M,)
    all_paths : list
        all_paths[m][k] = danh sách nút tuyến k luồng m.
    beta : float
        Hệ số phạt.
    
    Returns
    -------
    penalty : float
    """
    if beta <= 0:
        return 0.0
    
    # Đếm số luồng sử dụng mỗi cạnh
    edge_load = {}
    M = len(routing)
    
    for m in range(M):
        k = routing[m]
        path = all_paths[m][k]
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = (min(u, v), max(u, v))
            edge_load[edge] = edge_load.get(edge, 0) + 1
    
    penalty = beta * sum(load ** 2 for load in edge_load.values())
    return penalty


def compute_objective(
    routing: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    path_gains: np.ndarray,
    all_paths: List[List[List[int]]],
    beta: float
) -> Dict[str, float]:
    """
    Tính hàm mục tiêu đầy đủ J(x, p).
    
    J = -utility + edge_penalty
    
    Parameters
    ----------
    routing : np.ndarray, shape (M,)
    p : np.ndarray, shape (M,)
    w : np.ndarray, shape (M,)
    path_gains : np.ndarray, shape (M, K)
    all_paths : list
    beta : float
    
    Returns
    -------
    result : dict
        'J' : float - Giá trị hàm mục tiêu (càng nhỏ càng tốt)
        'utility' : float - Tổng độ hữu dụng
        'edge_penalty' : float - Phạt cạnh
        'alpha' : np.ndarray - Vector độ lợi hiệu dụng
    """
    alpha = compute_alpha(routing, path_gains)
    utility = compute_utility(alpha, p, w)
    edge_pen = compute_edge_penalty(routing, all_paths, beta)
    
    J = -utility + edge_pen
    
    return {
        'J': J,
        'utility': utility,
        'edge_penalty': edge_pen,
        'alpha': alpha,
    }


def evaluate_solution(
    routing: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    path_gains: np.ndarray,
    all_paths: List[List[List[int]]],
    beta: float,
    P_max: float
) -> Dict[str, Any]:
    """
    Đánh giá toàn diện một nghiệm (routing, p).
    
    Returns
    -------
    result : dict
        Bao gồm J, utility, edge_penalty, alpha,
        power_residual (vi phạm ràng buộc công suất),
        is_feasible (nghiệm hợp lệ hay không).
    """
    obj = compute_objective(routing, p, w, path_gains, all_paths, beta)
    
    total_power = float(np.sum(p))
    power_residual = max(0.0, total_power - P_max)
    is_feasible = (power_residual < 1e-6) and np.all(p >= -1e-10)
    
    obj['total_power'] = total_power
    obj['power_residual'] = power_residual
    obj['is_feasible'] = is_feasible
    
    return obj


if __name__ == '__main__':
    # Test nhanh
    M, K = 3, 2
    path_gains = np.array([[0.1, 0.05], [0.08, 0.12], [0.15, 0.07]])
    routing = np.array([0, 1, 0])  # Luồng 0 chọn tuyến 0, luồng 1 chọn tuyến 1, ...
    p = np.array([4.0, 3.0, 3.0])
    w = np.array([1.0, 1.5, 0.8])
    
    alpha = compute_alpha(routing, path_gains)
    print(f"Alpha: {alpha}")
    print(f"Utility: {compute_utility(alpha, p, w):.6f}")
    print(f"Objective (không phạt cạnh): {-compute_utility(alpha, p, w):.6f}")
