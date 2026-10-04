# BÁO CÁO TỔNG HỢP DỮ LIỆU THỰC NGHIỆM VÀ PHƯƠNG PHÁP
## Khảo sát Đa chiều (Thời gian và Chiều sâu Mạng) về Cơ chế Ảo giác Đối tượng trong LLaVA-1.5-7B

> **Mục đích tài liệu:** Cung cấp đầy đủ cơ sở phương pháp luận, thiết lập kỹ thuật, bảng dữ liệu thống kê thô và các quy luật hiện tượng đo lường được từ Thí nghiệm 1 và Thí nghiệm 2. Tài liệu được cấu trúc chuẩn hóa để Agent AI tiếp nhận có đầy đủ dữ liệu đưa ra các nhận định lý thuyết, cơ chế nơ-ron hoặc đề xuất giải pháp kỹ thuật tiếp theo.

---

## 1. THIẾT LẬP KỸ THUẬT VÀ MÔ HÌNH THỰC NGHIỆM

* **Mô hình nghiên cứu:** `llava-hf/llava-1.5-7b-hf` (Kiến trúc kết hợp: Visual Encoder CLIP ViT-L/14 + Linear Projector 2 tầng MLP + LLM Backbone LLaMA-2-7B gồm 32 khối Transformer).
* **Phần cứng và Môi trường:** $2 \times \text{NVIDIA Tesla T4}$ (32GB VRAM tổng), cấu hình `device_map="auto"`, độ chính xác `torch.float16`.
* **Cấu hình giải mã:** Greedy decoding (`do_sample=False`, `temperature=1.0`, `max_new_tokens=512`).
* **Prompt tiêu chuẩn:** `USER: <image>\nPlease describe this image in detail. ASSISTANT:`
* **Tập dữ liệu ảnh:** MS-COCO 2014 Validation set (`val2014`).
* **Nguồn nhãn chuẩn (Ground Truth):** Tổng hợp từ cả hai tệp nhãn của COCO:
  * `annotations/instances_val2014.json` (Bounding box vật thể do con người gắn).
  * `annotations/captions_val2014.json` (5 chú thích văn bản của con người cho mỗi ảnh).
* **Giao thức gán nhãn ảo giác:** Chuẩn CHAIR (Rohrbach et al., EMNLP 2018):
  * **Real (Thực):** Danh từ chuẩn hóa (canonical category) có mặt trong Ground Truth của ảnh.
  * **Hallucinated (Ảo giác):** Danh từ chuẩn hóa thuộc 80 lớp COCO nhưng không tồn tại trong Ground Truth của ảnh.

---

## 2. PHƯƠNG PHÁP GHÉP CẶP ĐỐI CHỨNG 1:1 (MATCHED PAIRS)

Để triệt tiêu hoàn toàn biến gây nhiễu về vị trí câu (vật thể thật thường là chủ ngữ ở đầu câu, vật thể ảo giác thường tích tụ ở cuối câu):
* **Chuẩn hóa vị trí tương đối:**
  $$\text{rel\_pos} = \frac{t}{G} \in (0, 1]$$
  *(với $t$ là chỉ số token xuất hiện trong chuỗi sinh, $G$ là tổng số token sinh ra của câu).*
* **Bộ lọc ứng viên:**
  * Chỉ xét lần đầu tiên vật thể được nhắc đến (`first=True`).
  * Vị trí token xuất hiện phải có đủ ngữ cảnh: $t \ge 10$.
  * Ranh giới bắt đầu từ phải hợp lệ (`word_start_valid=True`, ký tự đứng trước là khoảng trắng).
* **Thuật toán ghép cặp:**
  * Ghép tham lam không hoàn lại (without replacement, `seed=0`) mỗi vật thể ảo giác $h$ với một vật thể thật $r$ có khoảng cách vị trí tương đối thỏa mãn ngưỡng dung sai (caliper):
    $$|\text{rel\_pos}(h) - \text{rel\_pos}(r)| \le 0.10$$
* **Quy mô mẫu qua các giai đoạn:**
  * **Pilot cohort:** 100 ảnh.
  * **Development cohort:** 500 ảnh $\implies$ Ghép được **$343$ cặp đối chứng**.
  * **Confirmation cohort:** 2,000 ảnh (15,480 mentions, 6,948 first mentions) $\implies$ Ghép được **$1,430$ cặp đối chứng**.

---

## 3. CÁC ĐỊNH NGHĨA TOÁN HỌC VÀ CHỈ SỐ THỐNG KÊ

### 3.1. Điểm số Ưu tiên Chuẩn hóa $S(o, z)$
Để loại trừ ảnh hưởng của việc mô hình đang chuẩn bị nói một danh từ chung chung, điểm $S$ đo mức độ ưu tiên tương đối của vật thể $o$ so với toàn bộ 80 danh mục vật thể COCO và từ đồng nghĩa ($V_{obj}$):
$$S(o, z) = \log P(o \mid z) - \log \sum_{v \in V_{obj} \cup \{o\}} P(v \mid z)$$
* $z$ là vector logits $32{,}000$ chiều.
* $P(v \mid z) = \text{softmax}(z)_v$.
* Hiệu số cặp: $\Delta = S(h) - S(r)$. $\Delta < 0$ đồng nghĩa với việc mô hình ưu tiên vật thể thật hơn vật thể ảo giác.

### 3.2. Kiểm định Thống kê
* **Paired Bootstrap 95% CI:** Lấy mẫu lại có hoàn lại $B = 2{,}000$ lần trên các hiệu số $\Delta_i$ (`seed=0`). Lấy phân vị 2.5% và 97.5%.
* **Kích thước hiệu ứng Cohen's $d_z$:**
  $$d_z = \frac{\text{Mean}(\Delta)}{\text{SD}(\Delta)}$$
* **Kiểm định có dấu Wilcoxon (Wilcoxon signed-rank test):** Hai phía (two-sided) kiểm tra phân phối hiệu số cặp.
* **Hiệu chỉnh Đa giả thuyết Holm-Bonferroni ($p_{holm}$):** Kiểm soát tỷ lệ lỗi Family-Wise Error Rate (FWER) trên toàn bộ các bước hoặc các tầng được kiểm định.
* **Bootstrap AUROC:** Đo lường khả năng phân loại nhị phân của điểm $S$ giữa hai nhóm ($B = 2{,}000$).
* **Preceding Minimum Confidence (PMC):**
  $$\text{PMC} = \min_{i \in [t - n_{prec}, t - 1]} \max_{v} P(v \mid y_{<i})$$

---

## 4. THÍ NGHIỆM 1: KHẢO SÁT THEO CHIỀU THỜI GIAN (TEMPORAL LAG CURVE $m = 0 \to 10$)

### 4.1. Thiết lập
Khảo sát tại các bước lùi $t - m$ ($m \in \{0, 1, \dots, 10\}$) trước khi token vật thể $y_t$ được phát ngôn. Vector logits $z^{(m)} = \text{pred}[t - m]$ lấy từ tầng giải mã cuối cùng của mô hình.

### 4.2. Bảng Dữ liệu Thống kê Thô Thí nghiệm 1 (Confirmation Cohort: $N = 1{,}430$ cặp)

| Bước $m$ | Mean $S(h)$ | Mean $S(r)$ | Mean $\Delta$ | 95% Bootstrap CI | Cohen's $d_z$ | Holm-adjusted $p$ | Median Rank ($h$ vs. $r$) | Top-10 Rate ($h$ vs. $r$) | AUROC $S$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$m = 0$** | -0.7705 | -0.4943 | **-0.2762** | [-0.3077, -0.2439] | -0.4504 | **$5.34 \times 10^{-53}$** | 1.0 vs. 1.0 | 100.0% vs. 100.0% | 0.3358 |
| **$m = 1$** | -3.1336 | -2.3087 | **-0.8249** | [-1.0101, -0.6328] | -0.2329 | **$3.54 \times 10^{-22}$** | 22.0 vs. 13.0 | 26.6% vs. 42.3% | 0.3765 |
| **$m = 2$** | -6.6366 | -6.1801 | **-0.4565** | [-0.7217, -0.1974] | -0.0931 | **$0.001355$** | 219.0 vs. 152.0 | 7.4% vs. 9.7% | 0.4523 |
| **$m = 3$** | -6.3587 | -5.9050 | **-0.4536** | [-0.7494, -0.1669] | -0.0829 | **$0.004811$** | 213.0 vs. 193.5 | 8.9% vs. 10.9% | 0.4652 |
| **$m = 4$** | -7.1384 | -6.8941 | -0.2442 | [-0.5370, +0.0210] | -0.0457 | 0.280513 | 301.0 vs. 326.0 | 9.6% vs. 9.2% | 0.4772 |
| **$m = 5$** | -7.2960 | -7.0771 | -0.2189 | [-0.4951, +0.0402] | -0.0411 | 0.280513 | 346.0 vs. 388.5 | 7.1% vs. 7.6% | 0.4801 |
| **$m = 6$** | -7.9231 | -7.6452 | -0.2779 | [-0.5617, +0.0151] | -0.0496 | 0.044012 | 417.5 vs. 410.0 | 6.7% vs. 7.6% | 0.4706 |
| **$m = 7$** | -8.0760 | -7.9077 | -0.1683 | [-0.4874, +0.1317] | -0.0286 | 0.280513 | 432.5 vs. 445.5 | 5.6% vs. 6.4% | 0.4823 |
| **$m = 8$** | -8.1470 | -7.8941 | -0.2529 | [-0.5743, +0.0581] | -0.0404 | 0.280513 | 411.0 vs. 371.0 | 6.6% vs. 8.2% | 0.4814 |
| **$m = 9$** | -8.0986 | -7.6296 | **-0.4690** | [-0.7855, -0.1390] | -0.0766 | **$0.008993$** | 441.5 vs. 396.5 | 6.5% vs. 6.8% | 0.4671 |
| **$m = 10$** | -8.5209 | -7.9526 | **-0.5683** | [-0.9090, -0.2491] | -0.0922 | **$0.001572$** | 569.0 vs. 452.0 | 4.7% vs. 5.2% | 0.4559 |

### 4.3. Đặc trưng Phân phối tại Điểm Phát ngôn ($m = 0$)
* **Top-1 Confidence:** Thật = $53.08\%$ vs. Ảo giác = $39.45\%$ (Giảm $-13.63$ điểm phần trăm).
* **Shannon Entropy:** Thật = $1.8384\text{ nats}$ vs. Ảo giác = $2.4023\text{ nats}$ (Tăng $+30.67\%$).
* **Tỷ trọng xác suất trong COCO $\exp(S)$:** Thật = $61.0\%$ vs. Ảo giác = $46.3\%$.

### 4.4. Dữ liệu PMC Phân tầng (Kiểm tra lại TruthPrInt)
* **Tổng thể:** PMC Ảo giác = $0.2470 \pm 0.0892$ vs. PMC Thật = $0.2975 \pm 0.1106$.
* **Phân tầng theo $n_{prec}$:**
  * $n_{prec} \in [1, 3]$ (câu cực ngắn): PMC Ảo giác = **$0.3143$** vs. Thật = **$0.3064$** (Đảo chiều, ảo giác cao hơn).
  * $n_{prec} \in [4, 6]$: PMC Ảo giác = $0.2692$ vs. Thật = **$0.3644$** (Chênh lệch lớn nhất $-0.0952$).
  * $n_{prec} \in [7, 10]$: PMC Ảo giác = $0.2565$ vs. Thật = $0.2884$.
  * $n_{prec} \ge 11$: PMC Ảo giác = $0.2119$ vs. Thật = $0.2387$.

---

## 5. THÍ NGHIỆM 2: KHẢO SÁT THEO CHIỀU SÂU TẦNG MẠNG (LAYER LOGIT LENS TẠI $m = 1$)

### 5.1. Thiết lập và Kỹ thuật Logit Lens
Tại vị trí thời gian có phân hóa mạnh nhất $t - 1$ ($m = 1$), trích xuất vector trạng thái ẩn $h^{(l)}_{t-1} \in \mathbb{R}^{4096}$ của từng tầng $l \in [1, 32]$ và chiếu qua bộ giải mã cuối:
$$h_{normed}^{(l)} = \text{RMSNorm}(h^{(l)}_{t-1}), \quad z^{(l)} = \text{lm\_head}(h_{normed}^{(l)}) \in \mathbb{R}^{32000}$$
Đo lường $S(o, l)$ và thứ hạng từ điển qua toàn bộ 32 tầng.

### 5.2. Bảng Dữ liệu Thống kê Thô Thí nghiệm 2 ($N = 343$ cặp, $21{,}952$ bản ghi đa tầng)

| Tầng $l$ | Mean $S(h)$ | Mean $S(r)$ | Mean $\Delta$ | 95% Bootstrap CI | Cohen's $d_z$ | Holm-adjusted $p$ | Median Rank ($h$) | Median Rank ($r$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | -1.1211 | -1.5635 | **+0.4423** | [+0.2150, +0.6714] | 0.1992 | **$4.49 \times 10^{-3}$** | 9,005.0 | 11,453.0 |
| **2** | -1.2076 | -1.8010 | **+0.5934** | [+0.3078, +0.8779] | 0.2197 | **$1.70 \times 10^{-3}$** | **6,715.0** | **9,513.0** |
| **3** | -1.1064 | -1.5836 | **+0.4772** | [+0.2282, +0.7300] | 0.2009 | **$2.01 \times 10^{-3}$** | 6,283.0 | 9,000.0 |
| **4** | -0.8762 | -1.4362 | **+0.5600** | [+0.3380, +0.7841] | 0.2682 | **$1.87 \times 10^{-4}$** | 7,718.0 | 8,105.0 |
| **5** | -0.7294 | -1.2151 | **+0.4858** | [+0.3277, +0.6593] | 0.3072 | **$1.95 \times 10^{-6}$** | 6,208.0 | 7,977.0 |
| **6** | -0.4423 | -0.7685 | **+0.3262** | [+0.1950, +0.4763] | 0.2439 | **$6.28 \times 10^{-4}$** | 6,995.0 | 8,015.0 |
| **7** | -0.3998 | -0.7254 | **+0.3256** | [+0.1982, +0.4682] | 0.2593 | **$2.49 \times 10^{-4}$** | 7,319.0 | 8,566.0 |
| **8** | -0.3628 | -0.6243 | **+0.2615** | [+0.1553, +0.3842] | 0.2413 | **$3.07 \times 10^{-3}$** | 7,926.0 | 8,340.0 |
| **9** | -0.2829 | -0.5061 | **+0.2232** | [+0.1218, +0.3415] | 0.2199 | **$5.85 \times 10^{-4}$** | 8,962.0 | 8,954.0 |
| **10** | -0.2508 | -0.4407 | **+0.1900** | [+0.0994, +0.2921] | 0.2128 | **$1.14 \times 10^{-3}$** | **7,533.0** | **7,232.0 (Crossover)** |
| **11** | -0.2561 | -0.4172 | **+0.1611** | [+0.0643, +0.2633] | 0.1744 | **$3.85 \times 10^{-3}$** | 6,775.0 | 6,276.0 |
| **12** | -0.4771 | -0.7340 | **+0.2569** | [+0.1156, +0.4076] | 0.1849 | **$3.80 \times 10^{-3}$** | 6,575.0 | 5,195.0 |
| **13** | -0.4728 | -0.6843 | **+0.2116** | [+0.0747, +0.3566] | 0.1596 | **$1.83 \times 10^{-2}$** | 5,835.0 | 5,226.0 |
| **14** | -0.2955 | -0.4916 | **+0.1962** | [+0.0771, +0.3143] | 0.1727 | **$1.83 \times 10^{-2}$** | 6,652.0 | 6,192.0 |
| **15** | -0.2935 | -0.5119 | +0.2184 | [+0.0993, +0.3423] | 0.1913 | 0.056674 | 3,681.0 | **2,280.0** |
| **16** | -0.1739 | -0.2726 | +0.0987 | [+0.0134, +0.1925] | 0.1168 | 0.075365 | 3,688.0 | **2,328.0** |
| **17** | -0.1113 | -0.1507 | +0.0394 | [-0.0264, +0.1119] | 0.0620 | 0.279124 | 1,583.0 | **707.0** |
| **18** | -0.0979 | -0.1488 | +0.0509 | [-0.0105, +0.1172] | 0.0846 | 1.000000 | **479.0** | **156.0 (Gap > 320)** |
| **19** | -0.0562 | -0.0760 | +0.0198 | [-0.0175, +0.0622] | 0.0539 | 1.000000 | 158.0 | **32.0** |
| **20** | -0.0334 | -0.0542 | +0.0208 | [-0.0069, +0.0527] | 0.0715 | 0.624559 | 63.0 | **16.0** |
| **21** | -0.0104 | -0.0243 | **+0.0140** | [-0.0018, +0.0337] | 0.0812 | **$1.83 \times 10^{-2}$** | 20.0 | **4.0** |
| **22** | -0.0043 | -0.0084 | **+0.0041** | [-0.0017, +0.0111] | 0.0667 | **$5.68 \times 10^{-4}$** | 12.0 | **3.0** |
| **23** | -0.0019 | -0.0072 | **+0.0053** | [-0.0001, +0.0119] | 0.0928 | **$1.10 \times 10^{-3}$** | 8.0 | **2.0** |
| **24** | -0.0003 | -0.0018 | **+0.0015** | [-0.0001, +0.0040] | 0.0740 | **$2.01 \times 10^{-3}$** | 4.0 | **1.0 (Top 1)** |
| **25** | -0.0002 | -0.0018 | **+0.0016** | [-0.0001, +0.0042] | 0.0779 | **$5.85 \times 10^{-4}$** | 4.0 | 2.0 |
| **26** | -0.0001 | -0.0017 | **+0.0016** | [-0.0001, +0.0043] | 0.0705 | **$7.48 \times 10^{-4}$** | 3.0 | 2.0 |
| **27** | -0.0001 | -0.0019 | **+0.0019** | [-0.0000, +0.0048] | 0.0743 | **$1.17 \times 10^{-5}$** | 2.0 | 1.0 |
| **28** | -0.0000 | -0.0020 | **+0.0020** | [-0.0000, +0.0051] | 0.0767 | **$6.90 \times 10^{-8}$** | 2.0 | 1.0 |
| **29** | -0.0000 | -0.0016 | **+0.0016** | [+0.0000, +0.0047] | 0.0590 | **$2.86 \times 10^{-10}$** | 2.0 | 1.0 |
| **30** | -0.0000 | -0.0001 | **+0.0001** | [-0.0000, +0.0002] | 0.0528 | **$2.58 \times 10^{-9}$** | 2.0 | 1.0 |
| **31** | -0.0002 | -0.0001 | **-0.0001** | [-0.0002, +0.0000] | -0.0839 | **$3.60 \times 10^{-8}$** | **1.0** | **1.0** |
| **32** | -0.0001 | -0.0001 | **-0.0000** | [-0.0000, +0.0000] | -0.0163 | **$2.05 \times 10^{-7}$** | **1.0** | **1.0** |

---

## 6. CÁC QUY LUẬT HIỆN TƯỢNG ĐÃ ĐO LƯỜNG ĐƯỢC

1. **Về động lực học thời gian:**
   * Tín hiệu phân hóa $\Delta(m) < 0$ có ý nghĩa thống kê hình thành ở hai cụm: Cụm gần $m \in [1, 3]$ và Cụm xa $m \in [9, 10]$.
   * Khoảng giữa $m \in [4, 8]$ không đạt ý nghĩa thống kê sau hiệu chỉnh Holm (khoảng tin cậy cắt qua 0).
   * Phân loại AUROC đơn biến $< 0.5$ ở tất cả các bước, chứng minh rằng sự phân hóa chỉ có ý nghĩa ở mức dịch chuyển trung bình quần thể, không thể dùng làm bộ phân loại cá thể đơn biến.
2. **Về động lực học chiều sâu mạng (tại $m = 1$):**
   * **Tầng 1 – 9:** $\Delta(l) > 0$ có ý nghĩa thống kê; vật thể ảo giác có điểm $S$ cao hơn và thứ hạng từ điển cao hơn vật thể thật.
   * **Tầng 10:** Điểm chuyển pha (Crossover) nơi thứ hạng vật thể thật bắt đầu vượt qua vật thể ảo giác.
   * **Tầng 15 – 19:** Tốc độ thăng hạng của vật thể thật vượt trội (nhảy từ hạng $2{,}280$ lên $32$), trong khi vật thể ảo giác bị tụt lại phía sau hàng trăm bậc.
   * **Tầng 24 – 32:** Vật thể thật đạt Top 1 từ Tầng 24; vật thể ảo giác duy trì hạng 2–4 và chỉ nhảy cóc lên Top 1 ở Tầng 31–32.

---

*(Tài liệu này được lưu trữ độc lập tại `FULL_EXPERIMENTS_SUMMARY.md` trong thư mục gốc của repository).*
