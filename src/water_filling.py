"""
water_filling.py - Thuật toán KKT Water-Filling qua Bisection
=============================================================
TV2: Lõi giải phân bổ công suất tối ưu khi đã cố định cấu hình tuyến.

Bài toán con (lồi chặt khi α_m > 0, w_m > 0):
    min  -Σ w_m * ln(1 + α_m * p_m)
    s.t. Σ p_m ≤ P_max,   p_m ≥ 0

Nghiệm KKT Water-Filling:
    p_m(λ) = max(0, w_m/λ - 1/α_m)

Tìm λ* bằng Bisection sao cho Σ p_m(λ*) = P_max.
"""

import numpy as np
from typing import Dict, Any, Tuple


def water_filling_closed_form(
    alpha: np.ndarray,
    w: np.ndarray,
    lam: float
) -> np.ndarray:
    """
    Tính p_m(λ) = max(0, w_m/λ - 1/α_m) cho tất cả m.
    
    Parameters
    ----------
    alpha : np.ndarray, shape (M,)
        Độ lợi hiệu dụng (dương).
    w : np.ndarray, shape (M,)
        Trọng số ưu tiên (dương).
    lam : float
        Nhân tử Lagrange λ > 0.
    
    Returns
    -------
    p : np.ndarray, shape (M,)
        Vector phân bổ công suất.
    """
    p = np.maximum(0.0, w / lam - 1.0 / alpha)
    return p


def solve_water_filling(
    alpha: np.ndarray,
    w: np.ndarray,
    P_max: float,
    tolerance: float = 1e-8,
    max_iter: int = 100
) -> Dict[str, Any]:
    """
    Giải bài toán phân bổ công suất Water-Filling bằng Bisection.
    
    Parameters
    ----------
    alpha : np.ndarray, shape (M,)
        Độ lợi hiệu dụng (phải dương).
    w : np.ndarray, shape (M,)
        Trọng số ưu tiên (phải dương).
    P_max : float
        Tổng ngân sách công suất (phải dương).
    tolerance : float
        Ngưỡng dừng |Σp_m - P_max|.
    max_iter : int
        Số vòng lặp tối đa.
    
    Returns
    -------
    result : dict
        'p' : np.ndarray - Phân bổ công suất tối ưu
        'lambda_star' : float - Nhân tử Lagrange tối ưu
        'residual' : float - |Σp_m - P_max|
        'iterations' : int - Số vòng lặp đã dùng
        'converged' : bool - Đã hội tụ hay chưa
        'objective' : float - Giá trị hàm mục tiêu -Σ w_m ln(1 + α_m p_m)
    """
    M = len(alpha)
    
    # Kiểm tra đầu vào
    assert np.all(alpha > 0), "Tất cả α_m phải dương"
    assert np.all(w > 0), "Tất cả w_m phải dương"
    assert P_max > 0, "P_max phải dương"
    
    # Thiết lập khoảng tìm kiếm cho λ
    # Khi λ rất nhỏ -> p_m rất lớn -> Σp_m >> P_max
    # Khi λ rất lớn -> p_m = 0 -> Σp_m = 0
    lambda_low = 1e-12
    lambda_high = float(np.max(w * alpha)) * 10.0  # Đủ lớn để mọi p_m = 0
    
    # Kiểm tra: nếu λ_high chưa đủ lớn, tăng lên
    p_test = water_filling_closed_form(alpha, w, lambda_high)
    while np.sum(p_test) > P_max:
        lambda_high *= 10.0
        p_test = water_filling_closed_form(alpha, w, lambda_high)
    
    # Bisection loop
    converged = False
    iterations = 0
    lambda_mid = (lambda_low + lambda_high) / 2.0
    
    for it in range(max_iter):
        iterations = it + 1
        lambda_mid = (lambda_low + lambda_high) / 2.0
        
        p = water_filling_closed_form(alpha, w, lambda_mid)
        P_sum = np.sum(p)
        residual = abs(P_sum - P_max)
        
        if residual <= tolerance:
            converged = True
            break
        
        if P_sum > P_max:
            # Cần tăng λ để giảm p
            lambda_low = lambda_mid
        else:
            # Cần giảm λ để tăng p
            lambda_high = lambda_mid
    
    # Tính p cuối cùng
    p = water_filling_closed_form(alpha, w, lambda_mid)
    
    # Đảm bảo không âm (sai số số học)
    p = np.maximum(p, 0.0)
    
    # Tính hàm mục tiêu
    objective = -float(np.sum(w * np.log(1.0 + alpha * p)))
    
    return {
        'p': p,
        'lambda_star': lambda_mid,
        'residual': abs(np.sum(p) - P_max),
        'iterations': iterations,
        'converged': converged,
        'objective': objective,
    }


def solve_equal_allocation(
    alpha: np.ndarray,
    w: np.ndarray,
    P_max: float
) -> Dict[str, Any]:
    """
    Baseline: Phân bổ công suất đều cho tất cả luồng.
    p_m = P_max / M.
    """
    M = len(alpha)
    p = np.full(M, P_max / M)
    objective = -float(np.sum(w * np.log(1.0 + alpha * p)))
    return {
        'p': p,
        'lambda_star': None,
        'residual': 0.0,
        'iterations': 0,
        'converged': True,
        'objective': objective,
    }


if __name__ == '__main__':
    # Test cơ bản
    print("=== Test Water-Filling ===")
    
    alpha = np.array([0.1, 0.2, 0.05, 0.15])
    w = np.array([1.0, 1.5, 0.8, 1.2])
    P_max = 10.0
    
    result = solve_water_filling(alpha, w, P_max)
    
    print(f"p* = {result['p']}")
    print(f"λ* = {result['lambda_star']:.10f}")
    print(f"Σp = {np.sum(result['p']):.10f} (P_max = {P_max})")
    print(f"Residual = {result['residual']:.2e}")
    print(f"Converged: {result['converged']} in {result['iterations']} iterations")
    print(f"Objective = {result['objective']:.6f}")
    
    # Kiểm tra: p_m >= 0
    assert np.all(result['p'] >= -1e-10), "Vi phạm p_m >= 0"
    print("\n✓ Tất cả p_m >= 0")
    
    # Kiểm tra: tổng công suất
    assert abs(np.sum(result['p']) - P_max) < 1e-6, "Vi phạm tổng công suất"
    print("✓ Tổng công suất đúng")
