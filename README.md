# Thí nghiệm 1: "Object đã xuất hiện trước đó bao nhiêu bước?"

> **Mục tiêu nghiên cứu:** Khảo sát xác suất và độ ưu ái $S$ của một object tại các vị trí *sớm hơn* ($m = 0..K$) bước sinh thực sự của nó trên mô hình thị giác-ngôn ngữ **LLaVA-1.5-7B** (greedy captioning trên COCO val2014), so sánh giữa object **hallucinated** và object **thật** (ghép cặp đối chứng 1:1 theo vị trí tương đối). Kèm phân tích tái hiện thước đo **PMC** (Preceding Minimum Confidence) của TruthPrInt (ICCV 2025).

---

## 📁 Cấu trúc thư mục

```text
.
├── EXPERIMENT1_SPEC.md              # Bản đặc tả kỹ thuật chi tiết
├── README.md                        # Tài liệu hướng dẫn sử dụng
├── CHANGELOG.md                     # Nhật ký triển khai và thiết kế
├── requirements.txt                 # Danh sách thư viện Python cần thiết
├── .gitignore                       # Bỏ qua checkpoints, ảnh nặng, venv
├── data/
│   ├── synonyms.txt                 # Từ điển từ đồng nghĩa 80 category COCO (chuẩn CHAIR)
│   └── README.md                    # Hướng dẫn tải và đặt dữ liệu COCO 2014
├── src/
│   ├── __init__.py
│   ├── common.py                    # Tiện ích căn chỉnh token, trích xuất mention, thống kê
│   ├── 01_generate.py               # Step 1: Sinh caption greedy với LLaVA-1.5-7B
│   ├── 02_label_chair.py            # Step 2: Gán nhãn CHAIR, kiểm tra word-start, tính CHAIR_S / CHAIR_I
│   ├── 03_lag_curve.py              # Step 3: Ghép cặp đối chứng, forward teacher-forcing, tính S & thống kê A
│   ├── 04_pmc.py                    # Step 4: Phân tích B (PMC) & kiểm soát độ dài n_prec
│   └── 05_visualize_report.py       # Step 5: Vẽ biểu đồ, xuất run_manifest.json và REPORT.md tự động
├── tests/
│   ├── __init__.py
│   ├── test_common.py               # Unit tests T1, T2, T3 (không cần GPU)
│   └── test_gpu.py                  # Sanity checks T4, T5, T6, T7 (cần GPU)
└── notebooks/
    └── run_kaggle.ipynb             # Kaggle Notebook chuyên nghiệp, sạch sẽ, có mục lục tương tác
```

---

## 🚀 Hướng dẫn chạy trên Kaggle

Notebook [`notebooks/run_kaggle.ipynb`](notebooks/run_kaggle.ipynb) được thiết kế chuyên biệt để chạy trên Kaggle GPU (P100 hoặc T4 x 2):

1. **Đẩy codebase lên GitHub**:
   ```bash
   git init
   git add .
   git commit -m "feat: complete experiment 1 pipeline and kaggle notebook"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```
2. **Mở Kaggle**: Tạo một Notebook mới (chọn **GPU P100** hoặc **GPU T4 x 2** trong phần *Accelerator*).
3. **Mở notebook**: Import file `notebooks/run_kaggle.ipynb` vào Kaggle.
4. **Điền URL GitHub**: Cập nhật biến `GITHUB_REPO_URL` ở Cell 2.
5. **Chạy tuần tự**: Chạy lần lượt các ô lệnh từ Cổng G0 đến G5.
6. **Tải kết quả**: Cell cuối cùng sẽ tự động nén toàn bộ thư mục `results/` thành file zip để tải về trực tiếp từ tab *Output*.

---

## 💻 Hướng dẫn chạy bằng dòng lệnh (CLI)

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu COCO 2014
Đặt các file sau vào thư mục `data/`:
- `instances_val2014.json`
- `captions_val2014.json`
- `val2014/` (thư mục ảnh)

### 3. Quy trình thực nghiệm từng bước

#### Bước 0: Chạy kiểm thử Unit Test (Cổng G0)
```bash
python3 -m unittest discover -s tests -p "test_common.py" -v
```

#### Bước 1: Sinh Caption với LLaVA-1.5-7B
```bash
# Chạy smoke test 3 ảnh:
python3 src/01_generate.py --split smoke --n_images 3 --output_file data/smoke_captions.jsonl

# Chạy tập dev (500 ảnh [0, 500)):
python3 src/01_generate.py --split dev --offset 0 --n_images 500 --output_file data/captions.jsonl
```
*(Thêm cờ `--load_8bit` nếu GPU có VRAM $\le 16$ GB).*

#### Bước 2: Gán nhãn CHAIR và trích xuất mentions
```bash
python3 src/02_label_chair.py \
  --captions_file data/captions.jsonl \
  --output_file data/labels.jsonl
```

#### Bước 3: Forward Teacher-Forcing & Phân tích đường cong độ trễ (Câu hỏi A)
```bash
python3 src/03_lag_curve.py \
  --labels_file data/labels.jsonl \
  --output_dir results/exp1
```

#### Bước 4: Tái hiện thước đo PMC (Câu hỏi B)
```bash
python3 src/04_pmc.py \
  --labels_file data/labels.jsonl \
  --captions_file data/captions.jsonl \
  --output_dir results/exp1
```

#### Bước 5: Trực quan hóa & Sinh báo cáo tự động
```bash
python3 src/05_visualize_report.py \
  --captions_file data/captions.jsonl \
  --labels_file data/labels.jsonl \
  --output_dir results/exp1
```

---

## 📊 Định dạng dữ liệu đầu ra (`results/exp1/`)

| File | Mô tả |
|---|---|
| `lag_records.csv` | Dữ liệu từng object qua độ trễ $m=0..10$ (`pair_id, image_id, canon, group, t, G, rel_pos, m, logp, rank, S, conf, ent, logp_actual`) |
| `summary.csv` | Bảng thống kê tổng hợp chính theo $m$ (trung bình $S$, hiệu số $\Delta$, 95% Bootstrap CI, Wilcoxon p, Holm p, AUROC, top10, median rank) |
| `pmc_records.csv` | Thước đo PMC và khoảng cách `argmin_dist` cho toàn bộ mentions đầu tiên |
| `pmc_summary.csv` | Tổng hợp PMC theo nhóm, kiểm soát độ dài $n_{prec}$ và trên tập ghép cặp |
| `funnel.json` | Số lượng mẫu tại từng tầng lọc (captions $\to$ mentions $\to$ first $\to$ $t \ge K$ $\to$ word start $\to$ matched pairs) |
| `run_manifest.json` | Hash SHA-256 dữ liệu, thông số GPU, seed, commit, phiên bản thư viện |
| `REPORT.md` | Báo cáo đầy đủ 10 mục khoa học với số liệu thực tế |
| `figures/*.png` | Các đồ thị publication-ready: `s_vs_m.png`, `delta_vs_m.png`, `auroc_vs_m.png`, `pmc_by_group.png`, `pmc_by_nprec.png` |

---

## 🔒 Quy tắc nghiên cứu (Tiêu chí diễn giải)

- Kết quả được phân loại thành 1 trong 4 nhóm:
  1. **Tín hiệu sớm:** Holm-p < 0,05 tại ít nhất hai giá trị $m$ liên tiếp trong $m \ge 2$.
  2. **Chỉ ở bước cuối:** Holm-p < 0,05 tại $m=0$ (hoặc $m=1$), nhưng không có tại $m \ge 2$.
  3. **Không tín hiệu:** Không $m$ nào có ý nghĩa.
  4. **Ngược chiều:** Có ý nghĩa nhưng hallucinated có điểm $S$ cao hơn thật.
- **Không kết luận nhân quả:** Báo cáo chỉ diễn giải tương quan ở đầu ra phân phối token.
