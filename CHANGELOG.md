# CHANGELOG: Thí nghiệm 1 - "Object đã xuất hiện trước đó bao nhiêu bước?"

Tài liệu này ghi nhận toàn bộ các quyết định thiết kế, cấu trúc file, và các cải tiến/sửa đổi so với bản nháp ban đầu theo yêu cầu của `EXPERIMENT1_SPEC.md`.

---

## [1.0.0] - 2026-10-01

### 1. Khởi tạo cấu trúc dự án chuẩn mực
- **`src/common.py`**:
  - Triển khai `pieces_to_text(pieces)`: chuyển đổi chính xác các mảnh token SentencePiece/LLaMA thành văn bản thô, xử lý tiền tố space ` ` (`\u2581`), byte newline `<0x0A>` và các byte `<0xNN>`. Trả về vị trí ký tự span `(char_start, char_end)` cho từng token.
  - Triển khai `find_mentions(text, syn2canon)`: trích xuất object bằng regex `[A-Za-z]+`, ưu tiên cụm 2 từ (như `teddy bear`, `hot dog`), sau đó 1 từ, xử lý số nhiều (`-ies -> -y`, `-es -> -`, `-s -> -`).
  - Triển khai `check_word_start(spans, tok_idx, char_start)`: kiểm tra điều kiện word-start (`spans[tok_idx].start == char_start - 1` hoặc `char_start`) nhằm tránh các từ bắt đầu ở giữa token (loại trừ và ghi nhận vào funnel).
  - Triển khai `build_coco_gt`: hợp nhất categories từ `instances_val2014` và 5 human reference captions.
  - Triển khai thuật toán ghép cặp 1:1 không hoàn lại (`match_object_pairs`) theo `rel_pos` với caliper $\le 0.1$, random tie-breaking theo seed.
  - Triển khai bộ thư viện thống kê chuẩn xác: Paired Bootstrap CI ($B=2000$), Cohen's $d_z$, kiểm định Wilcoxon 2 phía, hiệu chỉnh Holm-Bonferroni qua $m = 0..K$, và Bootstrap AUROC.
- **`src/01_generate.py`**:
  - Hỗ trợ đầy đủ tham số `--offset` và `--n_images` để phân chia tập độc lập `dev` `[0, 500)`, `confirm` `[500, 2500)` và `smoke` `[0, 3)`.
  - Hỗ trợ resume: tự động đọc `captions.jsonl`, bỏ qua các `image_id` đã có và gọi `flush()` từng dòng.
  - Hỗ trợ `--load_8bit` và `--load_4bit` giúp chạy mượt mà trên GPU Kaggle 16GB (P100 / T4).
- **`src/02_label_chair.py`**:
  - Gán nhãn CHAIR, kiểm tra word-start, đánh dấu `first=True` cho mention đầu tiên của mỗi category trong caption.
  - Tính toán và in ra `CHAIR_S` và `CHAIR_I` toàn cục.
  - Lưu file `data/labels.jsonl` chuẩn schema Mục 5.
- **`src/03_lag_curve.py`**:
  - Gom nhóm object theo `image_id` để mỗi ảnh chỉ forward pass teacher-forcing **đúng 1 lần**.
  - Kiểm tra tiền tố tokenizer của mô hình khi tạo tập $V_{obj}$.
  - Tính toán và lưu trữ đầy đủ: `logp`, `rank`, `S`, `conf`, `ent`, `logp_actual`.
  - Ghi nhận `funnel.json` đầy đủ các tầng lọc.
  - Phân tích thống kê theo cặp (`summary.csv`), độ ổn định seed (seeds 0..19), độ nhạy S1 (caliper 0.05), S2 (unmatched) và S3 (category fixed effect $S_c$).
- **`src/04_pmc.py`**:
  - Triển khai phân tích B: tái hiện thước đo Preceding Minimum Confidence (PMC) của TruthPrInt.
  - Kiểm soát độ dài với các bin $n_{prec}$ (1–3, 4–6, 7–10, $\ge 11$) và trên tập ghép cặp.
  - Ghi nhận `pmc_records.csv` và `pmc_summary.csv`.
- **`src/05_visualize_report.py`**:
  - Tự động vẽ 5 biểu đồ chất lượng cao (300 DPI): `s_vs_m.png`, `delta_vs_m.png`, `auroc_vs_m.png`, `pmc_by_group.png`, `pmc_by_nprec.png`.
  - Ghi `run_manifest.json` chứa thông tin môi trường, GPU, seed, SHA-256 của các file data.
  - Tự động điền dữ liệu thực tế vào `REPORT.md` (10 mục theo Mục 9), tuyệt đối không bịa số và không kết luận nhân quả.
- **`tests/test_common.py`**:
  - Kiểm thử T1: chuỗi chuẩn SentencePiece mẫu, ánh xạ chính xác token index cho `man`, `dogs`, `teddy bear`, `mugs`.
  - Kiểm thử T2: kiểm tra thuật toán ghép cặp 1:1, caliper, không tái sử dụng object, tính xác định qua seed.
  - Kiểm thử T3: kiểm tra bootstrap CI, Holm-Bonferroni correction, AUROC.
- **`tests/test_gpu.py`**:
  - Kiểm thử T4: căn chỉnh vị trí `argmax(pred[i]) == gen_ids[i]` ($\ge 98\%$), phát hiện lỗi lệch off-by-one.
  - Kiểm thử T5: tại $m=0$, `rank == 1` cho $\ge 99\%$ object.
- **`data/synonyms.txt`**:
  - Bộ từ điển từ đồng nghĩa đầy đủ cho 80 danh mục COCO chuẩn nghiên cứu CHAIR.
- **`notebooks/run_kaggle.ipynb`**:
  - Notebook Kaggle hoàn chỉnh, sạch đẹp, có mục lục tương tác, thanh tiến trình, chạy tuần tự qua các Cổng G0 $\to$ G5, hiển thị biểu đồ và nén kết quả tải về.
