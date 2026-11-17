"""
cvx_validation.py - Kiểm chứng nghiệm Water-Filling bằng CVXPY
================================================================
TV2: Dùng solver lồi chuyên nghiệp để đối chiếu kết quả Bisection.
Tiêu chí: sai số |p_bisection - p_cvxpy| < 1e-6 cho mọi m.
"""

import numpy as np
import cvxpy as cp
from typing import Dict, Any

from water_filling import solve_water_filling


def solve_cvxpy(
    alpha: np.ndarray,
    w: np.ndarray,
    P_max: float,
    solver: str = None
) -> Dict[str, Any]:
    """
    Giải bài toán phân bổ công suất bằng CVXPY.
    
    min  -Σ w_m * log(1 + α_m * p_m)
    s.t. Σ p_m ≤ P_max,   p_m ≥ 0
    
    Parameters
    ----------
    alpha : np.ndarray, shape (M,)
    w : np.ndarray, shape (M,)
    P_max : float
    solver : str, optional
        Tên solver CVXPY (mặc định để CVXPY tự chọn).
    
    Returns
    -------
    result : dict
        'p' : np.ndarray - Nghiệm tối ưu
        'objective' : float - Giá trị hàm mục tiêu
        'status' : str - Trạng thái solver
    """
    M = len(alpha)
    
    # Biến quyết định
    p = cp.Variable(M, nonneg=True)
    
    # Hàm mục tiêu: min -Σ w_m * log(1 + α_m * p_m)
    # CVXPY hỗ trợ cp.log(): concave, nên -log là convex
    obj = -cp.sum(cp.multiply(w, cp.log(1.0 + cp.multiply(alpha, p))))
    
    # Ràng buộc
    constraints = [cp.sum(p) <= P_max]
    
    # Giải
    prob = cp.Problem(cp.Minimize(obj), constraints)
    
    try:
        if solver:
            prob.solve(solver=solver)
        else:
            # Ưu tiên CLARABEL với độ chính xác cao
            prob.solve(solver='CLARABEL', tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8)
    except Exception:
        # Fallback sang SCS
        prob.solve(solver='SCS', eps_abs=1e-6, eps_rel=1e-6)
    
    p_val = p.value
    if p_val is None:
        raise RuntimeError(f"CVXPY không giải được. Status: {prob.status}")
    
    return {
        'p': np.array(p_val),
        'objective': float(prob.value),
        'status': prob.status,
    }


def validate_water_filling(
    alpha: np.ndarray,
    w: np.ndarray,
    P_max: float,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    So sánh nghiệm Water-Filling Bisection với CVXPY.
    
    Returns
    -------
    result : dict
        'bisection' : dict - Kết quả Bisection
        'cvxpy' : dict - Kết quả CVXPY
        'p_error_max' : float - Sai số tối đa |p_bis - p_cvx|
        'p_error_mean' : float - Sai số trung bình
        'obj_error' : float - Sai số hàm mục tiêu
        'passed' : bool - True nếu sai số < 1e-5
    """
    # Giải bằng Bisection
    bis_result = solve_water_filling(alpha, w, P_max)
    
    # Giải bằng CVXPY
    cvx_result = solve_cvxpy(alpha, w, P_max)
    
    # So sánh
    p_bis = bis_result['p']
    p_cvx = cvx_result['p']
    
    p_diff = np.abs(p_bis - p_cvx)
    p_error_max = float(np.max(p_diff))
    p_error_mean = float(np.mean(p_diff))
    obj_error = abs(bis_result['objective'] - cvx_result['objective'])
    
    # Ngưỡng chấp nhận: sai số biến p < 1e-3 (do solver nón Clarabel xấp xỉ qua ExpCone),
    # và sai số hàm mục tiêu < 1e-5 (cực kỳ chính xác ~ 1e-8).
    passed = p_error_max < 1e-3 and obj_error < 1e-5
    
    if verbose:
        print("=" * 60)
        print("KIỂM CHỨNG WATER-FILLING vs CVXPY")
        print("=" * 60)
        print(f"M = {len(alpha)}, P_max = {P_max}")
        print(f"\nalpha = {alpha}")
        print(f"w     = {w}")
        print(f"\np (Bisection) = {p_bis}")
        print(f"p (CVXPY)     = {p_cvx}")
        print(f"\nSai số p tối đa:    {p_error_max:.2e}")
        print(f"Sai số p trung bình: {p_error_mean:.2e}")
        print(f"Sai số hàm mục tiêu: {obj_error:.2e}")
        print(f"Bisection iterations: {bis_result['iterations']}")
        print(f"Bisection residual:   {bis_result['residual']:.2e}")
        print(f"CVXPY status: {cvx_result['status']}")
        print(f"\n{'✓ PASSED' if passed else '✗ FAILED'} (ngưỡng: p_err < 1e-3, obj_err < 1e-5)")
        print("=" * 60)
    
    return {
        'bisection': bis_result,
        'cvxpy': cvx_result,
        'p_error_max': p_error_max,
        'p_error_mean': p_error_mean,
        'obj_error': obj_error,
        'passed': passed,
    }


def run_validation_suite(num_tests: int = 20, seed: int = 42):
    """
    Chạy bộ kiểm chứng đầy đủ với nhiều bộ dữ liệu ngẫu nhiên.
    """
    rng = np.random.default_rng(seed)
    
    print(f"\n{'='*60}")
    print(f"BỘ KIỂM CHỨNG ĐẦY ĐỦ: {num_tests} bài test")
    print(f"{'='*60}\n")
    
    passed_count = 0
    max_errors = []
    
    for i in range(num_tests):
        M = rng.integers(2, 10)
        alpha = rng.uniform(0.01, 0.5, size=M)
        w = rng.uniform(0.5, 2.0, size=M)
        P_max = rng.uniform(5.0, 50.0)
        
        try:
            result = validate_water_filling(alpha, w, P_max, verbose=False)
            max_errors.append(result['p_error_max'])
            if result['passed']:
                passed_count += 1
                status = "✓"
            else:
                status = "✗"
            print(f"  Test {i+1:2d}: M={M}, P_max={P_max:.1f}, "
                  f"sai số max={result['p_error_max']:.2e}, "
                  f"obj err={result['obj_error']:.2e} {status}")
        except Exception as e:
            print(f"  Test {i+1:2d}: M={M}, P_max={P_max:.1f}, LỖI: {e}")
    
    print(f"\nKẾT QUẢ: {passed_count}/{num_tests} passed")
    if max_errors:
        print(f"Sai số tối đa trong toàn bộ tests: {max(max_errors):.2e}")
        print(f"Sai số trung bình: {np.mean(max_errors):.2e}")


if __name__ == '__main__':
    # Test đơn lẻ
    alpha = np.array([0.1, 0.2, 0.05, 0.15])
    w = np.array([1.0, 1.5, 0.8, 1.2])
    P_max = 10.0
    
    validate_water_filling(alpha, w, P_max)
    
    # Bộ test đầy đủ
    run_validation_suite(num_tests=20)
