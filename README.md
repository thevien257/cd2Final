# Tối Ưu Định Tuyến Đa Luồng & Phân Bổ Công Suất
## Joint Multi-Flow Routing and Power Allocation via Deep-Guided K-Shortest Paths and KKT Water-Filling

### 📋 Mô tả
Dự án nghiên cứu bài toán đồng thời lựa chọn tuyến đường cho M luồng dữ liệu trên đồ thị mạng và phân bổ tổng công suất phát giới hạn. Sử dụng thuật toán lai 2 tầng:
- **Tầng 1:** MLP + Yen K-shortest paths → sinh tập cấu hình định tuyến ứng viên
- **Tầng 2:** KKT Water-Filling + Bisection → giải tối ưu phân bổ công suất

### ⚡ Cài đặt nhanh

```bash
# Tạo môi trường ảo
py -3.12 -m venv .venv
.venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt

# Kiểm tra
python -c "import numpy, networkx, cvxpy, torch; print('OK')"
```

### 🚀 Chạy chương trình

```bash
cd C:\hehe\Chuyende2\Ck

# Chạy đầy đủ (sinh dữ liệu + train + benchmark)
python main.py

# Chạy nhanh (ít dữ liệu, để test pipeline)
python main.py --quick

# Bỏ qua train, dùng checkpoint sẵn
python main.py --skip-train

# Bỏ qua kiểm chứng toán
python main.py --skip-validation
```

### 🧪 Chạy từng module riêng

```bash
cd src

# Kiểm chứng Water-Filling vs CVXPY
python cvx_validation.py

# Test sinh đồ thị và tuyến
python graph_builder.py
python path_generation.py

# So sánh baseline
python baseline.py

# Sinh dữ liệu
python dataset.py

# Test kiến trúc model
python model.py
```

### 📁 Cấu trúc thư mục

```
Ck/
├── configs/config.yaml          # Cấu hình thí nghiệm
├── src/
│   ├── graph_builder.py         # Sinh đồ thị mạng NetworkX
│   ├── path_generation.py       # Yen K-shortest paths
│   ├── objective.py             # Hàm mục tiêu J(x, p)
│   ├── water_filling.py         # KKT Water-Filling (Bisection)
│   ├── cvx_validation.py        # Kiểm chứng bằng CVXPY
│   ├── baseline.py              # Random, Shortest, Greedy, Exhaustive
│   ├── dataset.py               # Trích xuất đặc trưng & sinh dữ liệu
│   ├── model.py                 # MLP Softmax M×K
│   ├── train.py                 # Huấn luyện Cross-Entropy
│   ├── candidate_search.py      # Sinh L ứng viên & chọn nghiệm
│   └── benchmark.py             # Đánh giá tổng hợp
├── main.py                      # Pipeline end-to-end
├── data/                        # Dataset sinh tự động
├── results/
│   ├── figures/                 # Biểu đồ
│   ├── benchmark_results.csv    # Bảng kết quả
│   └── best_mlp.pt             # Checkpoint MLP
└── requirements.txt
```

### 📊 Kết quả mong đợi
- Bảng so sánh J (Objective) giữa: Random, Shortest Path, Greedy, Exhaustive Search, DNN-Guided
- Biểu đồ Optimality Gap (%) so với Exhaustive Search
- Biểu đồ Runtime khi tăng M
- Training curves (Loss & Accuracy)

### 👥 Phân công nhóm
| TV | Nhiệm vụ | Files |
|----|----------|-------|
| TV1 | Graph & Baseline | `graph_builder.py`, `path_generation.py`, `baseline.py` |
| TV2 | Optimization | `objective.py`, `water_filling.py`, `cvx_validation.py` |
| TV3 | Deep Learning | `dataset.py`, `model.py`, `train.py`, `candidate_search.py` |
# cd2Final
