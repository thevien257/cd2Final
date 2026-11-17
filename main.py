"""
main.py - Pipeline End-to-End: Tối Ưu Định Tuyến Đa Luồng & Phân Bổ Công Suất
================================================================================
Chạy toàn bộ quy trình từ sinh dữ liệu -> huấn luyện MLP -> benchmark.

Cách chạy:
    cd C:\\hehe\\Chuyende2\\Ck
    python main.py                    # Chạy đầy đủ
    python main.py --skip-train       # Bỏ qua huấn luyện, dùng checkpoint có sẵn
    python main.py --quick            # Chạy nhanh (ít dữ liệu) để test
"""

import os
import sys
import argparse
import yaml
import numpy as np
import time

# Thêm src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from graph_builder import build_graph, generate_flows, get_flow_weights
from path_generation import generate_all_candidates
from objective import compute_objective, evaluate_solution
from water_filling import solve_water_filling
from cvx_validation import validate_water_filling, run_validation_suite
from baseline import run_baseline
from dataset import create_dataset, extract_features, normalize_features
from model import create_model
from train import train_model, load_model
from candidate_search import dnn_guided_search
from benchmark import run_benchmark, plot_results, save_results


def print_header(text: str):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def step1_validate_water_filling():
    """Bước 1: Kiểm chứng Water-Filling vs CVXPY."""
    print_header("BƯỚC 1: KIỂM CHỨNG WATER-FILLING vs CVXPY")
    run_validation_suite(num_tests=15, seed=42)


def step2_demo_single_instance(config: dict):
    """Bước 2: Demo trên một kịch bản mạng đơn lẻ."""
    print_header("BƯỚC 2: DEMO TRÊN MỘT KỊCH BẢN MẠNG")
    
    rng = np.random.default_rng(config['seed'])
    M = config['flows']['num_flows']
    K = config['flows']['num_paths']
    P_max = config['optimization']['P_max']
    beta = config['optimization']['beta']
    
    # Sinh mạng
    G = build_graph(config, rng)
    flows = generate_flows(G, M, rng=rng)
    w = get_flow_weights(flows)
    cand = generate_all_candidates(G, flows, K, noise_std=0.3, rng=rng)
    
    print(f"Mạng: {G.number_of_nodes()} nút, {G.number_of_edges()} cạnh")
    print(f"Luồng: M={M}, Tuyến/luồng: K={K}")
    print(f"P_max={P_max}, beta={beta}")
    
    print(f"\nĐộ lợi kênh g_mk:")
    print(cand['path_gains'])
    
    # Chạy các baseline
    print(f"\n{'─'*50}")
    print("So sánh baseline:")
    print(f"{'─'*50}")
    
    for method in ['random', 'shortest', 'greedy', 'exhaustive']:
        result = run_baseline(method, cand['path_gains'], cand['all_paths'],
                              w, beta, P_max, rng)
        print(f"  {method:>12}: J = {result['J']:>10.6f}, "
              f"time = {result['time']:.4f}s, "
              f"routing = {result['routing']}")


def step3_generate_data(config: dict, quick: bool = False) -> dict:
    """Bước 3: Sinh bộ dữ liệu huấn luyện."""
    print_header("BƯỚC 3: SINH DỮ LIỆU HUẤN LUYỆN")
    
    n_train = 30 if quick else config['experiment']['num_train_instances']
    
    print(f"Sinh {n_train} kịch bản mạng (M={config['flows']['num_flows']}, "
          f"K={config['flows']['num_paths']})...")
    print("(Exhaustive Search trên mỗi kịch bản để lấy nhãn tối ưu)\n")
    
    dataset = create_dataset(config, num_instances=n_train, seed=config['seed'])
    
    # Lưu dataset
    np.savez(
        'data/training_dataset.npz',
        features=dataset['features'],
        labels=dataset['labels'],
        objectives=dataset['objectives'],
    )
    print(f"\nĐã lưu dataset: data/training_dataset.npz")
    
    return dataset


def step4_train(config: dict, dataset: dict) -> dict:
    """Bước 4: Huấn luyện MLP."""
    print_header("BƯỚC 4: HUẤN LUYỆN MLP")
    
    result = train_model(config, dataset, save_dir='results')
    return result


def step5_benchmark(config: dict, model, feat_mean, feat_std, device, quick=False):
    """Bước 5: Benchmark đầy đủ."""
    print_header("BƯỚC 5: BENCHMARK ĐẦY ĐỦ")
    
    n_test = 10 if quick else config['experiment']['num_test_instances']
    
    methods = ['random', 'shortest', 'greedy', 'exhaustive', 'dnn_guided']
    
    df = run_benchmark(
        config, model=model, feat_mean=feat_mean, feat_std=feat_std,
        device=device, num_test=n_test, methods=methods
    )
    
    save_results(df, 'results/benchmark_results.csv')
    plot_results(df, 'results/figures')
    
    return df


def step6_plot_training(train_result: dict):
    """Bước 6: Vẽ biểu đồ huấn luyện."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs('results/figures', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax = axes[0]
    ax.plot(train_result['train_losses'], label='Train Loss', alpha=0.8)
    ax.plot(train_result['val_losses'], label='Val Loss', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Accuracy
    ax = axes[1]
    ax.plot(train_result['val_accuracies'], label='Val Accuracy', color='green', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/training_curves.png', dpi=150)
    plt.close()
    print("Đã lưu: results/figures/training_curves.png")


def main():
    parser = argparse.ArgumentParser(description='Network Routing Optimization')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file')
    parser.add_argument('--skip-train', action='store_true', help='Dùng checkpoint sẵn')
    parser.add_argument('--quick', action='store_true', help='Chạy nhanh (ít dữ liệu)')
    parser.add_argument('--skip-validation', action='store_true', help='Bỏ qua kiểm chứng')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.quick:
        config['ml']['num_epochs'] = 30
        config['experiment']['num_train_instances'] = 30
        config['experiment']['num_test_instances'] = 10
    
    total_start = time.time()
    
    print_header("TỐI ƯU ĐỊNH TUYẾN ĐA LUỒNG & PHÂN BỔ CÔNG SUẤT")
    print("Deep Learning + KKT Water-Filling Hybrid Algorithm")
    print(f"Config: {args.config}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    
    # ---- Bước 1: Kiểm chứng toán ----
    if not args.skip_validation:
        step1_validate_water_filling()
    
    # ---- Bước 2: Demo ----
    step2_demo_single_instance(config)
    
    # ---- Bước 3 & 4: Train ----
    if args.skip_train:
        print_header("BƯỚC 3-4: NẠP MODEL TỪ CHECKPOINT")
        ckpt = load_model('results/best_mlp.pt')
        model = ckpt['model']
        feat_mean = ckpt['feat_mean']
        feat_std = ckpt['feat_std']
        device = ckpt['device']
        train_result = None
    else:
        dataset = step3_generate_data(config, args.quick)
        train_result = step4_train(config, dataset)
        model = train_result['model']
        feat_mean = train_result['feat_mean']
        feat_std = train_result['feat_std']
        device = train_result['device']
        
        step6_plot_training(train_result)
    
    # ---- Bước 5: Benchmark ----
    df = step5_benchmark(config, model, feat_mean, feat_std, device, args.quick)
    
    # ---- Tổng kết ----
    total_time = time.time() - total_start
    print_header("HOÀN TẤT")
    print(f"Tổng thời gian: {total_time:.1f}s ({total_time/60:.1f} phút)")
    print(f"\nSản phẩm đã tạo:")
    print(f"  ✓ data/training_dataset.npz")
    print(f"  ✓ results/best_mlp.pt")
    print(f"  ✓ results/benchmark_results.csv")
    print(f"  ✓ results/figures/objective_comparison.png")
    print(f"  ✓ results/figures/runtime_comparison.png")
    print(f"  ✓ results/figures/optimality_gap.png")
    print(f"  ✓ results/figures/training_curves.png")


if __name__ == '__main__':
    main()
