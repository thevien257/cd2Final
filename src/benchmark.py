"""
benchmark.py - Đánh giá tổng hợp và so sánh các phương pháp
=============================================================
Chạy benchmark trên tập test: DNN-Guided vs các baseline.
Xuất bảng CSV và vẽ biểu đồ so sánh.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from typing import Dict, Any, List

from graph_builder import build_graph, generate_flows, get_flow_weights
from path_generation import generate_all_candidates
from objective import compute_objective, compute_alpha
from water_filling import solve_water_filling
from baseline import run_baseline
from dataset import extract_features, normalize_features
from candidate_search import dnn_guided_search


def run_single_instance(
    config: Dict[str, Any],
    model,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    device,
    rng: np.random.Generator,
    methods: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Chạy tất cả phương pháp trên một instance.
    
    Returns
    -------
    results : list of dict
        Mỗi dict gồm method, J, utility, time, routing, ...
    """
    if methods is None:
        methods = ['random', 'shortest', 'greedy', 'exhaustive', 'dnn_guided']
    
    M = config['flows']['num_flows']
    K = config['flows']['num_paths']
    P_max = config['optimization']['P_max']
    beta = config['optimization']['beta']
    L = config['ml']['num_candidates']
    noise_std = config['experiment'].get('gain_noise_std', 0.3)
    
    # Sinh instance
    G = build_graph(config, rng)
    flows = generate_flows(G, M, rng=rng)
    w = get_flow_weights(flows)
    cand = generate_all_candidates(G, flows, K, noise_std, rng)
    
    results = []
    
    for method in methods:
        if method == 'dnn_guided' and model is not None:
            # DNN-Guided
            features = extract_features(
                cand['path_gains'], cand['path_costs'],
                cand['path_hops'], w, P_max
            )
            dnn_result = dnn_guided_search(
                model, features, cand['path_gains'], cand['all_paths'],
                w, beta, P_max, L, feat_mean, feat_std, device, rng
            )
            
            alpha = compute_alpha(dnn_result['best_routing'], cand['path_gains'])
            obj = compute_objective(
                dnn_result['best_routing'], dnn_result['best_p'],
                w, cand['path_gains'], cand['all_paths'], beta
            )
            
            results.append({
                'method': 'DNN-Guided',
                'J': dnn_result['best_J'],
                'utility': obj['utility'],
                'edge_penalty': obj['edge_penalty'],
                'time': dnn_result['total_time'],
                'inference_time': dnn_result['inference_time'],
                'eval_time': dnn_result['eval_time'],
                'routing': dnn_result['best_routing'].tolist(),
                'num_candidates': L,
            })
        elif method != 'dnn_guided':
            # Baseline
            bl_result = run_baseline(
                method, cand['path_gains'], cand['all_paths'],
                w, beta, P_max, rng
            )
            results.append({
                'method': method.capitalize(),
                'J': bl_result['J'],
                'utility': bl_result.get('utility', None),
                'edge_penalty': bl_result.get('edge_penalty', None),
                'time': bl_result['time'],
                'inference_time': 0.0,
                'eval_time': bl_result['time'],
                'routing': bl_result['routing'].tolist(),
                'num_candidates': bl_result.get('num_configs', 1),
            })
    
    return results


def run_benchmark(
    config: Dict[str, Any],
    model=None,
    feat_mean: np.ndarray = None,
    feat_std: np.ndarray = None,
    device=None,
    num_test: int = None,
    seed: int = None,
    methods: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Chạy benchmark đầy đủ trên nhiều instance.
    
    Returns
    -------
    df : pd.DataFrame
        Bảng kết quả chi tiết.
    """
    if num_test is None:
        num_test = config['experiment']['num_test_instances']
    if seed is None:
        seed = config['seed'] + 9999  # Seed khác training
    if methods is None:
        methods = ['random', 'shortest', 'greedy', 'exhaustive']
        if model is not None:
            methods.append('dnn_guided')
    
    rng = np.random.default_rng(seed)
    
    all_results = []
    start_time = time.time()
    
    for i in range(num_test):
        instance_results = run_single_instance(
            config, model, feat_mean, feat_std, device, rng, methods
        )
        
        for r in instance_results:
            r['instance'] = i
            all_results.append(r)
        
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Instance {i+1}/{num_test}, elapsed = {elapsed:.1f}s")
    
    df = pd.DataFrame(all_results)
    
    if verbose:
        print(f"\n{'='*70}")
        print("BẢNG TỔNG HỢP KẾT QUẢ")
        print(f"{'='*70}")
        
        summary = df.groupby('method').agg({
            'J': ['mean', 'std'],
            'time': ['mean', 'std'],
        }).round(6)
        print(summary)
        
        # Tính Optimality Gap nếu có Exhaustive
        if 'Exhaustive' in df['method'].values:
            ex_J = df[df['method'] == 'Exhaustive'].groupby('instance')['J'].first()
            
            print(f"\n{'='*70}")
            print("OPTIMALITY GAP (%) so với Exhaustive Search")
            print(f"{'='*70}")
            
            for method in df['method'].unique():
                if method == 'Exhaustive':
                    continue
                method_J = df[df['method'] == method].groupby('instance')['J'].first()
                
                # Align indices
                common = ex_J.index.intersection(method_J.index)
                gaps = ((method_J[common].values - ex_J[common].values) 
                        / np.maximum(np.abs(ex_J[common].values), 1e-10) * 100)
                
                print(f"  {method:>12}: gap = {np.mean(gaps):.2f}% ± {np.std(gaps):.2f}%")
    
    return df


def plot_results(
    df: pd.DataFrame,
    save_dir: str = 'results/figures',
    show: bool = False
):
    """Vẽ biểu đồ so sánh và lưu file."""
    os.makedirs(save_dir, exist_ok=True)
    
    methods = df['method'].unique()
    
    # ---- Hình 1: So sánh J trung bình ----
    fig, ax = plt.subplots(figsize=(8, 5))
    summary = df.groupby('method')['J'].agg(['mean', 'std'])
    bars = ax.bar(summary.index, summary['mean'], yerr=summary['std'],
                  capsize=5, color='steelblue', alpha=0.8)
    ax.set_ylabel('Objective J (lower is better)')
    ax.set_title('So sánh J trung bình giữa các phương pháp')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'objective_comparison.png'), dpi=150)
    if show:
        plt.show()
    plt.close()
    
    # ---- Hình 2: Runtime ----
    fig, ax = plt.subplots(figsize=(8, 5))
    time_summary = df.groupby('method')['time'].agg(['mean', 'std'])
    bars = ax.bar(time_summary.index, time_summary['mean'], yerr=time_summary['std'],
                  capsize=5, color='coral', alpha=0.8)
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Thời gian chạy trung bình')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'runtime_comparison.png'), dpi=150)
    if show:
        plt.show()
    plt.close()
    
    # ---- Hình 3: Optimality Gap ----
    if 'Exhaustive' in methods:
        ex_J = df[df['method'] == 'Exhaustive'].groupby('instance')['J'].first()
        
        gap_data = {}
        for method in methods:
            if method == 'Exhaustive':
                continue
            method_J = df[df['method'] == method].groupby('instance')['J'].first()
            common = ex_J.index.intersection(method_J.index)
            gaps = ((method_J[common].values - ex_J[common].values) 
                    / np.maximum(np.abs(ex_J[common].values), 1e-10) * 100)
            gap_data[method] = gaps
        
        if gap_data:
            fig, ax = plt.subplots(figsize=(8, 5))
            try:
                ax.boxplot(list(gap_data.values()), tick_labels=list(gap_data.keys()))
            except TypeError:
                ax.boxplot(list(gap_data.values()), labels=list(gap_data.keys()))
            ax.set_ylabel('Optimality Gap (%)')
            ax.set_title('Phân bố Optimality Gap so với Exhaustive Search')
            ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'optimality_gap.png'), dpi=150)
            if show:
                plt.show()
            plt.close()
    
    print(f"Đã lưu biểu đồ vào {save_dir}/")


def save_results(df: pd.DataFrame, save_path: str = 'results/benchmark_results.csv'):
    """Lưu kết quả ra CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Đã lưu kết quả: {save_path}")


if __name__ == '__main__':
    import yaml
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Chạy benchmark chỉ với baseline (không có model)
    config['experiment']['num_test_instances'] = 20
    
    print("Chạy benchmark baseline (không có DNN)...")
    df = run_benchmark(
        config, model=None, num_test=20,
        methods=['random', 'shortest', 'greedy', 'exhaustive']
    )
    
    save_results(df)
    plot_results(df)
