"""
stress_test.py - Kiểm thử chịu tải & Phân tích Khả năng Mở rộng
==================================================================
Chạy thí nghiệm tăng dần độ khó:
  Mức 1: M=4, K=3  (81 cấu hình)
  Mức 2: M=6, K=3  (729 cấu hình)
  Mức 3: M=8, K=3  (6561 cấu hình)

Đo: Optimality Gap, Runtime, Chất lượng nghiệm J khi bài toán khó dần.

Cách chạy:
    cd C:\hehe\Chuyende2\Ck
    .\.venv\Scripts\python.exe stress_test.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import yaml
from graph_builder import build_graph, generate_flows, get_flow_weights
from path_generation import generate_all_candidates
from objective import compute_objective, compute_alpha
from water_filling import solve_water_filling
from baseline import run_baseline
from dataset import extract_features, normalize_features, create_dataset
from model import create_model
from train import train_model
from candidate_search import dnn_guided_search


# ============================================================
# CẤU HÌNH CÁC MỨC ĐỘ KHÓ
# ============================================================
DIFFICULTY_LEVELS = [
    {'name': 'Dễ',          'M': 4, 'K': 3, 'combos': 81,    'num_train': 200, 'num_test': 50},
    {'name': 'Trung bình',  'M': 6, 'K': 3, 'combos': 729,   'num_train': 300, 'num_test': 50},
    {'name': 'Khó',         'M': 8, 'K': 3, 'combos': 6561,  'num_train': 400, 'num_test': 30},
]

METHODS = ['random', 'shortest', 'greedy', 'exhaustive']


def run_stress_level(base_config, level, seed=42):
    """
    Chạy một mức độ khó: sinh dữ liệu → train MLP → benchmark.
    """
    config = dict(base_config)
    config['flows'] = {'num_flows': level['M'], 'num_paths': level['K']}
    config['ml'] = dict(base_config['ml'])
    config['ml']['num_epochs'] = 100
    config['experiment'] = dict(base_config['experiment'])

    M, K = level['M'], level['K']
    P_max = config['optimization']['P_max']
    beta = config['optimization']['beta']
    L = config['ml']['num_candidates']
    noise_std = config['experiment'].get('gain_noise_std', 0.3)

    print(f"\n{'='*70}")
    print(f"  MỨC ĐỘ: {level['name']} | M={M}, K={K} | "
          f"Không gian tìm kiếm: {K}^{M} = {level['combos']:,} cấu hình")
    print(f"  L={L} ứng viên AI → Tỉ lệ thử: {L/level['combos']*100:.2f}%")
    print(f"{'='*70}")

    # ---- Bước 1: Sinh dữ liệu huấn luyện ----
    print(f"\n[1/3] Sinh {level['num_train']} mẫu huấn luyện (Exhaustive Search)...")
    t0 = time.time()
    dataset = create_dataset(config, num_instances=level['num_train'], seed=seed, verbose=False)
    print(f"      Hoàn tất trong {time.time()-t0:.1f}s | Feature dim: {dataset['features'].shape[1]}")

    # ---- Bước 2: Huấn luyện MLP ----
    print(f"\n[2/3] Huấn luyện MLP ({config['ml']['num_epochs']} epochs)...")
    t0 = time.time()
    train_result = train_model(config, dataset, save_dir=f'results/stress_M{M}', verbose=False)
    model = train_result['model']
    feat_mean = train_result['feat_mean']
    feat_std = train_result['feat_std']
    device = train_result['device']
    best_acc = max(train_result['val_accuracies'])
    print(f"      Hoàn tất trong {time.time()-t0:.1f}s | "
          f"Best val_acc: {best_acc:.3f} | Best val_loss: {train_result['best_val_loss']:.4f}")

    # ---- Bước 3: Benchmark trên test set ----
    print(f"\n[3/3] Benchmark trên {level['num_test']} kịch bản kiểm thử...")
    rng = np.random.default_rng(seed + 7777)

    all_results = []
    t0 = time.time()

    for i in range(level['num_test']):
        G = build_graph(config, rng)
        flows = generate_flows(G, M, rng=rng)
        w = get_flow_weights(flows)
        cand = generate_all_candidates(G, flows, K, noise_std, rng)

        # Baseline methods
        for method in METHODS:
            bl = run_baseline(method, cand['path_gains'], cand['all_paths'], w, beta, P_max, rng)
            all_results.append({
                'level': level['name'], 'M': M, 'K': K,
                'combos': level['combos'],
                'method': method.capitalize(),
                'J': bl['J'], 'time': bl['time'],
                'instance': i,
            })

        # DNN-Guided
        features = extract_features(
            cand['path_gains'], cand['path_costs'], cand['path_hops'], w, P_max)
        dnn = dnn_guided_search(
            model, features, cand['path_gains'], cand['all_paths'],
            w, beta, P_max, L, feat_mean, feat_std, device, rng)
        all_results.append({
            'level': level['name'], 'M': M, 'K': K,
            'combos': level['combos'],
            'method': 'DNN-Guided',
            'J': dnn['best_J'], 'time': dnn['total_time'],
            'instance': i,
        })

        if (i + 1) % 10 == 0:
            print(f"      Đã test {i+1}/{level['num_test']}...")

    elapsed = time.time() - t0
    print(f"      Benchmark hoàn tất trong {elapsed:.1f}s")

    return pd.DataFrame(all_results)


def analyze_and_plot(df):
    """Phân tích kết quả và vẽ biểu đồ so sánh đa cấp độ."""
    os.makedirs('results/figures', exist_ok=True)

    levels = df['level'].unique()
    methods_order = ['Random', 'Shortest', 'Greedy', 'DNN-Guided', 'Exhaustive']
    methods_plot = [m for m in methods_order if m in df['method'].unique()]

    # ============================================================
    # BẢNG TỔNG HỢP
    # ============================================================
    print(f"\n{'='*80}")
    print(f"{'BẢNG TỔNG HỢP STRESS TEST':^80}")
    print(f"{'='*80}")

    summary_rows = []

    for lvl in levels:
        df_lvl = df[df['level'] == lvl]
        M = df_lvl['M'].iloc[0]
        K = df_lvl['K'].iloc[0]
        combos = df_lvl['combos'].iloc[0]

        ex_J = df_lvl[df_lvl['method'] == 'Exhaustive'].groupby('instance')['J'].first()

        print(f"\n--- {lvl} (M={M}, K={K}, Không gian={combos:,}) ---")
        print(f"{'Phương pháp':>14} | {'J trung bình':>12} | {'Gap (%)':>10} | "
              f"{'Gap Median':>10} | {'Gap Max':>10} | {'Time (ms)':>10}")
        print(f"{'-'*14}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        for method in methods_plot:
            m_df = df_lvl[df_lvl['method'] == method]
            m_J = m_df.groupby('instance')['J'].first()
            m_time = m_df['time'].mean() * 1000

            if method == 'Exhaustive':
                gap_mean = gap_med = gap_max = 0.0
            else:
                common = ex_J.index.intersection(m_J.index)
                gaps = (m_J[common].values - ex_J[common].values) / np.maximum(
                    np.abs(ex_J[common].values), 1e-10) * 100
                gap_mean = np.mean(gaps)
                gap_med = np.median(gaps)
                gap_max = np.max(gaps)

            print(f"{method:>14} | {m_J.mean():>12.4f} | {gap_mean:>9.2f}% | "
                  f"{gap_med:>9.2f}% | {gap_max:>9.2f}% | {m_time:>9.2f}")

            summary_rows.append({
                'Level': lvl, 'M': M, 'K': K, 'Combos': combos,
                'Method': method,
                'J_mean': m_J.mean(), 'J_std': m_J.std(),
                'Gap_mean': gap_mean, 'Gap_median': gap_med, 'Gap_max': gap_max,
                'Time_ms': m_time,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('results/stress_test_summary.csv', index=False, encoding='utf-8-sig')
    print(f"\nĐã lưu: results/stress_test_summary.csv")

    # ============================================================
    # HÌNH 1: Optimality Gap theo mức độ khó (Chỉ DNN-Guided & Greedy)
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    focus_methods = ['DNN-Guided', 'Greedy']
    colors = {'DNN-Guided': '#2196F3', 'Greedy': '#FF9800'}
    x_positions = np.arange(len(levels))
    width = 0.35

    for i, method in enumerate(focus_methods):
        means = []
        stds = []
        for lvl in levels:
            row = summary_df[(summary_df['Level'] == lvl) & (summary_df['Method'] == method)]
            if len(row) > 0:
                means.append(row['Gap_mean'].values[0])
                # Tính std gap từ raw data
                df_lvl = df[(df['level'] == lvl) & (df['method'] == method)]
                ex_J = df[(df['level'] == lvl) & (df['method'] == 'Exhaustive')].groupby('instance')['J'].first()
                m_J = df_lvl.groupby('instance')['J'].first()
                common = ex_J.index.intersection(m_J.index)
                gaps = (m_J[common].values - ex_J[common].values) / np.maximum(
                    np.abs(ex_J[common].values), 1e-10) * 100
                stds.append(np.std(gaps))
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(x_positions + i * width - width/2, means, width,
                      yerr=stds, capsize=5, label=method, color=colors[method], alpha=0.85)
        # Ghi số lên đầu cột
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{m:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    combo_labels = [f"{lvl}\nM={summary_df[summary_df['Level']==lvl]['M'].iloc[0]}, "
                    f"K={summary_df[summary_df['Level']==lvl]['K'].iloc[0]}\n"
                    f"({summary_df[summary_df['Level']==lvl]['Combos'].iloc[0]:,} configs)"
                    for lvl in levels]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(combo_labels)
    ax.set_ylabel('Optimality Gap (%)', fontsize=12)
    ax.set_title('Độ lệch so với nghiệm tối ưu khi bài toán khó dần', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('results/figures/stress_gap_by_difficulty.png', dpi=150)
    plt.close()

    # ============================================================
    # HÌNH 2: Runtime - Đường cong thời gian (Exhaustive bùng nổ vs AI phẳng)
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    runtime_methods = ['Exhaustive', 'DNN-Guided', 'Greedy', 'Shortest']
    markers = {'Exhaustive': 's', 'DNN-Guided': 'o', 'Greedy': '^', 'Shortest': 'D'}
    colors2 = {'Exhaustive': '#F44336', 'DNN-Guided': '#2196F3',
               'Greedy': '#FF9800', 'Shortest': '#4CAF50'}

    combos_list = [summary_df[summary_df['Level'] == lvl]['Combos'].iloc[0] for lvl in levels]

    for method in runtime_methods:
        times = []
        for lvl in levels:
            row = summary_df[(summary_df['Level'] == lvl) & (summary_df['Method'] == method)]
            times.append(row['Time_ms'].values[0] if len(row) > 0 else 0)

        ax.plot(combos_list, times, marker=markers[method], linewidth=2.5,
                markersize=10, label=method, color=colors2[method])
        # Ghi số bên cạnh
        for x, y in zip(combos_list, times):
            ax.annotate(f'{y:.1f}ms', (x, y), textcoords='offset points',
                        xytext=(10, 5), fontsize=9)

    ax.set_xscale('log')
    ax.set_xlabel('Không gian tìm kiếm (K^M cấu hình)', fontsize=12)
    ax.set_ylabel('Thời gian trung bình (ms)', fontsize=12)
    ax.set_title('Thời gian chạy khi không gian bài toán tăng theo hàm mũ', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(combos_list)
    ax.set_xticklabels([f'{c:,}' for c in combos_list])
    plt.tight_layout()
    plt.savefig('results/figures/stress_runtime_scaling.png', dpi=150)
    plt.close()

    # ============================================================
    # HÌNH 3: Boxplot Gap phân bố theo từng mức độ (chỉ DNN-Guided)
    # ============================================================
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 5), sharey=True)
    if len(levels) == 1:
        axes = [axes]

    for idx, lvl in enumerate(levels):
        ax = axes[idx]
        df_lvl = df[df['level'] == lvl]
        ex_J = df_lvl[df_lvl['method'] == 'Exhaustive'].groupby('instance')['J'].first()

        gap_data = {}
        for method in ['DNN-Guided', 'Greedy']:
            m_J = df_lvl[df_lvl['method'] == method].groupby('instance')['J'].first()
            common = ex_J.index.intersection(m_J.index)
            gaps = (m_J[common].values - ex_J[common].values) / np.maximum(
                np.abs(ex_J[common].values), 1e-10) * 100
            gap_data[method] = gaps

        if gap_data:
            try:
                bp = ax.boxplot(list(gap_data.values()), tick_labels=list(gap_data.keys()),
                                patch_artist=True)
            except TypeError:
                bp = ax.boxplot(list(gap_data.values()), patch_artist=True)
                ax.set_xticklabels(list(gap_data.keys()))

            colors_box = ['#2196F3', '#FF9800']
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        M_val = df_lvl['M'].iloc[0]
        combos_val = df_lvl['combos'].iloc[0]
        ax.set_title(f'{lvl}\nM={M_val} ({combos_val:,} configs)', fontweight='bold')
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Optimality Gap (%)')
    fig.suptitle('Phân bố Optimality Gap khi tăng độ phức tạp bài toán',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/figures/stress_gap_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ============================================================
    # HÌNH 4: Tỉ lệ AI tìm đúng nghiệm tối ưu (Hit Rate)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    hit_rates_dnn = []
    hit_rates_greedy = []

    for lvl in levels:
        df_lvl = df[df['level'] == lvl]
        ex_J = df_lvl[df_lvl['method'] == 'Exhaustive'].groupby('instance')['J'].first()

        for method, storage in [('DNN-Guided', hit_rates_dnn), ('Greedy', hit_rates_greedy)]:
            m_J = df_lvl[df_lvl['method'] == method].groupby('instance')['J'].first()
            common = ex_J.index.intersection(m_J.index)
            gaps = np.abs(m_J[common].values - ex_J[common].values) / np.maximum(
                np.abs(ex_J[common].values), 1e-10) * 100
            hit_rate = np.mean(gaps < 1.0) * 100  # Gap < 1% = gần tối ưu
            storage.append(hit_rate)

    x = np.arange(len(levels))
    ax.bar(x - 0.18, hit_rates_dnn, 0.35, label='DNN-Guided', color='#2196F3', alpha=0.85)
    ax.bar(x + 0.18, hit_rates_greedy, 0.35, label='Greedy', color='#FF9800', alpha=0.85)

    for i, (d, g) in enumerate(zip(hit_rates_dnn, hit_rates_greedy)):
        ax.text(i - 0.18, d + 1, f'{d:.0f}%', ha='center', fontweight='bold')
        ax.text(i + 0.18, g + 1, f'{g:.0f}%', ha='center', fontweight='bold')

    combo_labels = [f"{lvl}\n({summary_df[summary_df['Level']==lvl]['Combos'].iloc[0]:,} configs)"
                    for lvl in levels]
    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels)
    ax.set_ylabel('Tỉ lệ tìm đúng nghiệm gần tối ưu (Gap < 1%)', fontsize=11)
    ax.set_title('Hit Rate: Xác suất AI tìm được nghiệm chuẩn', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/stress_hit_rate.png', dpi=150)
    plt.close()

    print(f"\nĐã lưu 4 biểu đồ vào results/figures/:")
    print(f"  ✓ stress_gap_by_difficulty.png   (Optimality Gap theo độ khó)")
    print(f"  ✓ stress_runtime_scaling.png     (Runtime bùng nổ vs AI phẳng)")
    print(f"  ✓ stress_gap_boxplot.png         (Phân bố Gap chi tiết)")
    print(f"  ✓ stress_hit_rate.png            (Tỉ lệ AI tìm đúng nghiệm)")


def main():
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    print("=" * 70)
    print("  STRESS TEST: KIỂM THỬ CHỊU TẢI & KHẢ NĂNG MỞ RỘNG")
    print("  Tăng dần M (số luồng) để đánh giá AI dưới áp lực thực tế")
    print("=" * 70)

    all_dfs = []
    total_start = time.time()

    for level in DIFFICULTY_LEVELS:
        df = run_stress_level(base_config, level, seed=base_config['seed'])
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Lưu raw data
    combined.to_csv('results/stress_test_raw.csv', index=False, encoding='utf-8-sig')

    # Phân tích & Vẽ biểu đồ
    analyze_and_plot(combined)

    total = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  STRESS TEST HOÀN TẤT trong {total:.0f}s ({total/60:.1f} phút)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
