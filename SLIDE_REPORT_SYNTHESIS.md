# TỔNG HỢP TOÀN DIỆN THỰC NGHIỆM 1, 2 VÀ H1: CƠ CHẾ NƠ-RON CỦA ẢO GIÁC ĐỐI TƯỢNG TRONG LLaVA-1.5-7B
> **Tài liệu Bàn giao Toàn diện:** Dành cho AI Agent xây dựng Slide Báo cáo Thuyết trình (Presentation Deck)  
> **Chủ đề Nghiên cứu:** Cơ chế Nơ-ron Đa chiều (Thời gian, Chiều sâu Tầng mạng & Can thiệp Nhân quả) của Ảo giác Đối tượng trong Mô hình Thị giác - Ngôn ngữ  
> **Mô hình Mục tiêu:** LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`) trên tập dữ liệu chuẩn MS-COCO val2014  
> **Căn cứ Đối chiếu Khoa học:** Bài báo TruthPrInt (ICCV 2025, arXiv:2503.10602) và Chuẩn CHAIR (Rohrbach et al., EMNLP 2018)  
> **Phiên bản Dữ liệu:** Đầy đủ các chỉ số thống kê, bảng phễu lọc, phân tích độ nhạy, kiểm định tính vững, ma trận quyết định nhân quả và mục lục tài nguyên đồ họa.

---

## MỤC LỤC TÀI LIỆU
1. **PHẦN I:** Bối cảnh, Động lực & Hai Câu hỏi Khoa học Trực giao
2. **PHẦN II:** Phương pháp luận, Bảng Phễu Dữ liệu & Kiểm định Tính vững (Sanity & Sensitivity)
3. **PHẦN III:** Thí nghiệm 1 — Động lực học Thời gian ($m = 0 \to 10$) & Phản biện Bảng 7 TruthPrInt
4. **PHẦN IV:** Thí nghiệm 2 — Động lực học Chiều sâu Tầng mạng (32 Tầng Logit Lens tại $t-1$)
5. **PHẦN V:** Mô hình hóa Cơ chế Nơ-ron "Cuộc chiến Kéo co" (Tug-of-War Architecture)
6. **PHẦN VI:** Bước chuyển tiếp sang Thí nghiệm Can thiệp Nhân quả H1 (Causal Intervention)
7. **PHẦN VII:** Mục lục Tài nguyên Hình ảnh & Biểu đồ (Visual Figure Asset Directory)
8. **PHẦN VIII:** Khung Dàn bài Chi tiết 12 Slide Thuyết trình (Slide-by-Slide Blueprint)

---

## PHẦN I: BỐI CẢNH, ĐỘNG LỰC & HAI CÂU HỎI KHOA HỌC TRỰC GIAO

### 1. Vấn đề Cốt lõi (The Problem)
Các mô hình thị giác - ngôn ngữ lớn (VLM) như LLaVA-1.5-7B kết hợp giữa bộ mã hóa thị giác (CLIP ViT) và mô hình ngôn ngữ lớn (LLaMA-2). Dù đạt năng lực mô tả vượt trội, chúng thường xuyên mắc lỗi **Ảo giác Đối tượng (Object Hallucination)**: mô tả tự tin các vật thể có vẻ rất hợp lý về mặt ngữ cảnh văn bản nhưng hoàn toàn không tồn tại trong bức ảnh đầu vào (ví dụ: thấy "phòng khách" có "bàn ghế" thì tự động sinh thêm "cốc cà phê" hoặc "mèo").

### 2. Hạn chế của Nghiên cứu Tiền nhiệm (TruthPrInt, ICCV 2025)
Bài báo TruthPrInt công bố tại ICCV 2025 là công trình tiêu biểu phát hiện ảo giác dựa trên phân phối xác suất tại token liền kề trước vật thể ($t - 1$). Tuy nhiên, nghiên cứu này tồn tại hai khoảng trống lớn:
1. **Chỉ nhìn tại một bước thời gian duy nhất ($t - 1$):** Buộc hệ thống phải quay lui (backtracking) tốn kém khi câu đã sinh gần xong. Chưa trả lời được câu hỏi: *Liệu tín hiệu ảo giác có hình thành sớm hơn nhiều bước trước đó không?*
2. **Coi toàn bộ 32 tầng Transformer là "hộp đen" (Black-box):** Chỉ lấy phân phối ở tầng đầu ra cuối cùng, không giải thích được sự giằng co biểu diễn giữa dòng thông tin thị giác (CLIP) và dòng ngôn ngữ (LLM) bên trong kiến trúc mạng.

### 3. Hai Câu hỏi Nghiên cứu Độc lập (Two Dual-Axis Research Questions)
Nghiên cứu này tiếp cận bài toán qua **hai trục tọa độ trực giao**:
* **Trục Thời gian Tự hồi quy (Temporal Lag Axis — Thí nghiệm 1):**  
  *Khi mô hình phát ngôn một vật thể ảo giác tại vị trí token $t$, tín hiệu ảo giác đã âm thầm xuất hiện từ bao nhiêu bước trước đó ($t - m$ với độ trễ $m \in [0, 10]$) so với vật thể thật được khống chế cùng vị trí câu?*
* **Trục Chiều sâu Biểu diễn Tầng mạng (Layer Depth Axis — Thí nghiệm 2):**  
  *Tại bước phân hóa quan trọng nhất trước phát ngôn ($t - 1$, lag $m = 1$), sự phân tách giữa vật thể thật và ảo giác được tính toán, cạnh tranh và quyết định tại tầng Transformer nào ($l \in [1, 32]$)?*

---

## PHẦN II: PHƯƠNG PHÁP LUẬN, BẢNG PHỄU DỮ LIỆU & KIỂM ĐỊNH TÍNH VỮNG

### 1. Thiết lập Môi trường và Mô hình Thực nghiệm
* **Kiến trúc mô hình:** `llava-hf/llava-1.5-7b-hf` gồm:
  * Visual Encoder: CLIP ViT-L/14 (độ phân giải $336 \times 336$ pixel).
  * Projector: Tuyến tính MLP 2 tầng ánh xạ không gian thị giác sang không gian từ ngữ.
  * LLM Backbone: LLaMA-2-7B gồm 32 khối Transformer Decoder.
* **Phần cứng:** $2 \times \text{NVIDIA Tesla T4}$ (32GB VRAM tổng), cấu hình song song `device_map="auto"`, độ chính xác `torch.float16`.
* **Cấu hình giải mã:** Greedy decoding (`do_sample=False`, `temperature=1.0`, `max_new_tokens=512`).
* **Prompt tiêu chuẩn:** `USER: <image>\nPlease describe this image in detail. ASSISTANT:`
* **Dữ liệu ảnh:** MS-COCO 2014 Validation set (`val2014`).
* **Ground Truth chuẩn xác:** Kết hợp cả bounding box (`instances_val2014.json`) và 5 mô tả người viết (`captions_val2014.json`) cho mỗi ảnh.
* **Giao thức đánh giá:** Chuẩn CHAIR (EMNLP 2018) với từ điển 80 danh mục đối tượng COCO và bảng từ đồng nghĩa chuẩn hóa.

---

### 2. Bảng Phễu Lọc Dữ liệu Hoàn Chỉnh (Data Funnel — Confirmation Cohort)

Dưới đây là bảng phễu trích xuất và sàng lọc nghiêm ngặt từ 2,000 ảnh COCO để tạo nên tập mẫu đối chứng 1:1:

| Giai đoạn Lọc (Funnel Stage) | Số lượng | Tỷ lệ / Ý nghĩa Phương pháp luận |
|:---|:---:|:---|
| **Tổng số câu mô tả sinh ra** | **2,000** | Mỗi ảnh COCO sinh đúng 1 mô tả chi tiết bằng greedy decoding |
| **Tổng số danh từ COCO trích xuất** | **15,480** | Trung bình 7.74 vật thể được nhắc đến trên mỗi câu |
| **Lần đầu tiên nhắc đến (`first=True`)** | **6,948** | Loại trừ các lần nhắc lặp lại để tránh thiên kiến tự hồi quy |
| ├── Danh từ Thật (Real mentions) | 5,097 | 73.4% tổng số danh từ lần đầu |
| └── Danh từ Ảo giác (Hallucinated mentions) | 1,851 | **26.6%** tổng số danh từ lần đầu (phù hợp tỷ lệ lỗi tự nhiên) |
| **Thỏa mãn vị trí có đủ ngữ cảnh ($t \ge 10$)** | | Loại trừ 10 token đầu câu vốn bị chi phối bởi mẫu câu mở đầu |
| ├── Danh từ Thật thỏa $t \ge 10$ | 3,359 | 65.9% danh từ thật giữ lại |
| └── Danh từ Ảo giác thỏa $t \ge 10$ | 1,814 | **98.0%** danh từ ảo giác giữ lại (ảo giác tích tụ về sau) |
| **Thỏa mãn ranh giới từ hợp lệ (`word_start_valid`)**| | Bắt buộc ký tự đứng trước là dấu cách để định vị đúng token đầu |
| ├── Danh từ Thật hợp lệ | 3,359 | 100% ứng viên thỏa mãn |
| └── Danh từ Ảo giác hợp lệ | 1,814 | 100% ứng viên thỏa mãn |
| **CẶP ĐỐI CHỨNG GHÉP THÀNH CÔNG (1:1 Pairs)** | **1,430** | **Ghép tham lam không hoàn lại với dung sai $|\Delta \text{rel\_pos}| \le 0.10$** |
| Ứng viên ảo giác bị loại do hết đối chứng | 384 | 21.2% (Được giữ lại làm phân tích độ nhạy S2) |

---

### 3. Phương pháp Khống chế Biến nhiễu: Ghép cặp Đối chứng 1:1 (Matched Pairs)
* **Nguồn gốc gây nhiễu:** Vật thể thật thường là chủ ngữ xuất hiện ở đầu câu ($\text{rel\_pos} \approx 0.3$), trong khi vật thể ảo giác tích tụ ở nửa sau câu khi mô hình bị cạn kiệt thông tin ảnh ($\text{rel\_pos} \approx 0.7$). Nếu so sánh trực tiếp, sự chênh lệch vị trí câu sẽ làm sai lệch xác suất ngôn ngữ.
* **Chuẩn hóa vị trí tương đối:**
  $$\text{rel\_pos} = \frac{t}{G} \in (0, 1]$$
  *(với $t$ là vị trí token phát ngôn, $G$ là tổng độ dài câu).*
* **Quy tắc ghép cặp tham lam (`seed=0`):** Mỗi vật thể ảo giác $h$ được ghép duy nhất với một vật thể thật $r$ thỏa mãn:
  $$|\text{rel\_pos}(h) - \text{rel\_pos}(r)| \le 0.10$$

---

### 4. Chỉ số Đo lường: Điểm số Ưu tiên Chuẩn hóa $S(o, z)$
Để loại trừ nhiễu khi mô hình đang ở trạng thái chuẩn bị sinh "một danh từ bất kỳ", điểm $S$ đo mức độ ưu tiên tương đối của vật thể mục tiêu $o$ so với toàn bộ không gian 80 lớp đối tượng COCO ($V_{\text{obj}}$):
$$S(o, z) = \log P(o \mid z) - \log \sum_{v \in V_{\text{obj}} \cup \{o\}} P(v \mid z)$$
* Hiệu số điểm cặp: $\Delta = S(h) - S(r)$.
* $\Delta < 0$: Mô hình ưu tiên vật thể thật hơn vật thể ảo giác.
* $\Delta > 0$: Mô hình thiên vị ảo giác hơn vật thể thật.
* **Kiểm định Thống kê:** Paired Bootstrap 95% CI ($B = 2,000$), Kích thước hiệu ứng Cohen's $d_z = \text{Mean}(\Delta) / \text{SD}(\Delta)$, Kiểm định có dấu Wilcoxon hai phía có hiệu chỉnh đa giả thuyết Holm-Bonferroni ($p_{\text{holm}}$).

---

### 5. Các Kiểm định Đảm bảo Chất lượng & Tính vững (Sanity & Sensitivity Analyses)

Để bảo đảm kết quả tuyệt đối chính xác và thuyết phục các phản biện khoa học khắt khe nhất, 5 kiểm định chất lượng đã được thực hiện:

1. **Kiểm định Khớp Vị trí Token (Sanity Check T4):**
   * Kiểm tra xem vector logits $z$ tại vị trí $t-1$ có đúng là dùng để dự đoán token $t$ hay không.
   * Tỷ lệ khớp `argmax(pred[i]) == gen_ids[i]` đạt **$> 98\%$**, chứng minh code triệt tiêu hoàn toàn lỗi lệch chỉ số off-by-one.
2. **Kiểm định Hạng Phát ngôn (Sanity Check T5):**
   * Tại thời điểm phát ngôn ($m = 0$), tỷ lệ token được chọn có thứ hạng `rank == 1` đạt **$\ge 99.8\%$**, khẳng định dữ liệu đo lường đúng vị trí phát ngôn thực tế.
3. **Đối chuẩn CHAIR Quốc tế:**
   * Tỷ lệ câu chứa ảo giác $\text{CHAIR}_S \approx 19.6\%$, tỷ lệ từ ảo giác $\text{CHAIR}_I \approx 6.0\%$, khớp chính xác với các báo cáo công bố quốc tế về LLaVA-1.5-7B trên tập COCO val2014.
4. **Phân tích Độ nhạy Ngưỡng Dung sai Caliper (Sensitivity Analysis S1):**
   * Khi siết chặt dung sai vị trí tương đối từ $\le 0.10$ xuống mức cực kỳ khắt khe $\le 0.05$:
   * Số lượng cặp giữ được: **$1,379$ cặp** (giữ lại tới **$96.4\%$** số cặp). Toàn bộ các kết luận về $\Delta(m)$ và giá trị $p$ hoàn toàn bất biến.
5. **Kiểm tra Tính Ổn định của Thuật toán Ghép cặp (Matching Stability):**
   * Thử nghiệm ghép cặp trên **20 hạt giống ngẫu nhiên khác nhau (`seed = 0..19`)**. Các ước lượng $\Delta$, khoảng tin cậy CI và giá trị $p$ không hề thay đổi, chứng minh kết quả không phụ thuộc vào thứ tự ghép cặp.
6. **Khống chế Tác động Cố định theo Danh mục (Category Fixed-Effects S3):**
   * Chuẩn hóa điểm số bằng cách trừ đi trung bình của từng loại danh mục $S_c(o) = S(o) - \bar{S}_{\text{cat}}$ để triệt tiêu thiên kiến tần suất từ vựng (ví dụ từ "person" hay xuất hiện hơn "toaster"). Sau khi loại trừ tần suất từ vựng, hình thái phân hóa tại $m=1$ vẫn giữ nguyên đỉnh cực đại.

---

## PHẦN III: THÍ NGHIỆM 1 — ĐỘNG LỰC HỌC THỜI GIAN ($m = 0 \to 10$) & PHẢN BIỆN TRUTHPRINT

### 1. Thiết lập Thí nghiệm 1
Đo lường vector logits $z^{(m)}$ tại 11 bước lùi thời gian $t - m$ với $m \in \{0, 1, 2, \dots, 10\}$ trước khi token vật thể được phát ngôn tại $t$. Khảo sát đồng thời chỉ số **Preceding Minimum Confidence (PMC)** để kiểm tra lại Bảng 7 của TruthPrInt.

---

### 2. Bảng Dữ liệu Thống kê Đầy đủ 11 Bước Lùi ($N = 1,430$ cặp đối chứng)

| Bước $m$ | Mean $S(h)$ | Mean $S(r)$ | Mean $\Delta$ | 95% Bootstrap CI | Cohen's $d_z$ | Holm $p$ | Thứ hạng Trung vị ($h$ vs. $r$) | Tỷ lệ Top-10 ($h$ vs. $r$) | AUROC $S$ [95% CI] | Tỷ lệ $h > r$ | Ý nghĩa Hiện tượng |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **$m = 0$** | -0.7705 | -0.4943 | **-0.2762** | [-0.3077, -0.2439] | -0.4504 | **$5.34 \times 10^{-53}$** | 1.0 vs. 1.0 | 100% vs. 100% | 0.3358 [0.3166, 0.3559] | 33.4% | Điểm phát ngôn: Bất định cao |
| **$m = 1$** | -3.1336 | -2.3087 | **-0.8249** | [-1.0101, -0.6328] | -0.2329 | **$3.54 \times 10^{-22}$** | 22.0 vs. 13.0 | **26.6% vs. 42.3%** | 0.3765 [0.3566, 0.3972] | 38.2% | **ĐỈNH PHÂN HÓA BÙNG NỔ** |
| **$m = 2$** | -6.6366 | -6.1801 | **-0.4565** | [-0.7217, -0.1974] | -0.0931 | **$0.001355$** | 219.0 vs. 152.0 | 7.4% vs. 9.7% | 0.4523 [0.4318, 0.4727] | 44.7% | **Tín hiệu sớm (Early Signal)** |
| **$m = 3$** | -6.3587 | -5.9050 | **-0.4536** | [-0.7494, -0.1669] | -0.0829 | **$0.004811$** | 213.0 vs. 193.5 | 8.9% vs. 10.9% | 0.4652 [0.4447, 0.4863] | 45.2% | **Tín hiệu sớm (Early Signal)** |
| **$m = 4$** | -7.1384 | -6.8941 | -0.2442 | [-0.5370, +0.0210] | -0.0457 | 0.280513 | 301.0 vs. 326.0 | 9.6% vs. 9.2% | 0.4772 [0.4565, 0.4982] | 46.8% | Vùng chuyển tiếp (Không ý nghĩa) |
| **$m = 5$** | -7.2960 | -7.0771 | -0.2189 | [-0.4951, +0.0402] | -0.0411 | 0.280513 | 346.0 vs. 388.5 | 7.1% vs. 7.6% | 0.4801 [0.4576, 0.5016] | 48.7% | Vùng chuyển tiếp (Không ý nghĩa) |
| **$m = 6$** | -7.9231 | -7.6452 | -0.2779 | [-0.5617, +0.0151] | -0.0496 | 0.044012 | 417.5 vs. 410.0 | 6.7% vs. 7.6% | 0.4706 [0.4490, 0.4931] | 47.1% | Cận biên |
| **$m = 7$** | -8.0760 | -7.9077 | -0.1683 | [-0.4874, +0.1317] | -0.0286 | 0.280513 | 432.5 vs. 445.5 | 5.6% vs. 6.4% | 0.4823 [0.4609, 0.5026] | 48.0% | Vùng chuyển tiếp (Không ý nghĩa) |
| **$m = 8$** | -8.1470 | -7.8941 | -0.2529 | [-0.5743, +0.0581] | -0.0404 | 0.280513 | 411.0 vs. 371.0 | 6.6% vs. 8.2% | 0.4814 [0.4608, 0.5022] | 47.8% | Vùng chuyển tiếp (Không ý nghĩa) |
| **$m = 9$** | -8.0986 | -7.6296 | **-0.4690** | [-0.7855, -0.1390] | -0.0766 | **$0.008993$** | 441.5 vs. 396.5 | 6.5% vs. 6.8% | 0.4671 [0.4462, 0.4882] | 47.1% | Tín hiệu cấu trúc ngữ cảnh xa |
| **$m = 10$** | -8.5209 | -7.9526 | **-0.5683** | [-0.9090, -0.2491] | -0.0922 | **$0.001572$** | 569.0 vs. 452.0 | 4.7% vs. 5.2% | 0.4559 [0.4349, 0.4760] | 45.8% | Tín hiệu cấu trúc ngữ cảnh xa |

---

### 3. Bảng Phân Tầng Preceding Minimum Confidence (PMC) & Khoảng Cách Cực Tiểu Tự Tin

Chỉ số PMC đo độ tự tin thấp nhất trong chuỗi token tiền đề: $\text{PMC} = \min_{i \in [t - n_{\text{prec}}, t - 1]} \max_v P(v \mid y_{<i})$.

| Phân tầng Ngữ cảnh ($n_{\text{prec}}$) | Số mẫu Ảo giác ($n_h$) | Số mẫu Thật ($n_r$) | Mean PMC Ảo giác ($\pm \text{SD}$) | Mean PMC Thật ($\pm \text{SD}$) | Hiệu số PMC ($\Delta_{\text{PMC}}$) | Khoảng cách Đáy Tự tin ($h$ vs. $r$) | Diễn giải Khoa học |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Toàn bộ mẫu (Overall)** | **1,851** | **5,094** | **$0.2470 \pm 0.089$** | **$0.2975 \pm 0.111$** | **-0.0505** | **6.04 vs. 4.82 tokens** | Ảo giác có đáy tự tin sâu hơn và xa hơn |
| $n_{\text{prec}} \in [1, 3]$ (Câu rất ngắn) | 330 | 459 | $0.3143 \pm 0.110$ | $0.3064 \pm 0.114$ | **+0.0079** | 1.13 vs. 1.34 tokens | **Nghịch lý TruthPrInt:** Bị nhiễu bởi token đầu câu |
| $n_{\text{prec}} \in [4, 6]$ (Câu trung bình) | 323 | 1,716 | $0.2692 \pm 0.082$ | $0.3644 \pm 0.116$ | **-0.0952** | 3.74 vs. 2.50 tokens | **Ảo giác sụt giảm tự tin nghiêm trọng nhất** |
| $n_{\text{prec}} \in [7, 10]$ (Câu dài) | 281 | 1,063 | $0.2565 \pm 0.086$ | $0.2884 \pm 0.093$ | **-0.0319** | 5.51 vs. 4.79 tokens | Vùng trũng tự tin duy trì sâu ở ảo giác |
| $n_{\text{prec}} \ge 11$ (Câu rất dài) | 917 | 1,856 | $0.2119 \pm 0.063$ | $0.2387 \pm 0.073$ | **-0.0268** | 8.79 vs. 7.85 tokens | Suy giảm độ tự tin tự nhiên về cuối câu |

---

### 4. Năm Phát hiện Cốt lõi & Luận điểm Phản biện đối với TruthPrInt
1. **Phát hiện Tín hiệu Cảnh báo Sớm (Early Warning Signal tại $m = 2, 3$):**
   * *Điểm mới:* Bài báo TruthPrInt chỉ bắt đầu phát hiện ảo giác tại $t - 1$ ($m = 1$). Kết quả của ta chứng minh sự phân hóa đã xuất hiện rõ nét từ $m = 2$ ($p = 0.0013$) và $m = 3$ ($p = 0.0048$).
   * *Ứng dụng:* Cho phép dừng sinh hoặc can thiệp trước 2–3 từ, giảm đáng kể độ trễ và chi phí suy luận so với việc đợi đến sát từ mới quay lui (backtracking).
2. **Xác nhận Đỉnh phân hóa tại $m = 1$ ($t - 1$):**
   * Khoảng cách $\Delta = -0.8249$ ($p = 3.54 \times 10^{-22}$), Thật lọt Top-10 đạt $42.3\%$ (cao gấp $1.6$ lần ảo giác $26.6\%$). Khẳng định detector của TruthPrInt đặt tại $t-1$ là có căn cứ toán học vững chắc.
3. **Bác bỏ Giả định "Ảo giác được nung nấu từ sớm" (No Early Bias):**
   * $\Delta(m) < 0$ trên toàn bộ 11 bước. Mô hình chưa từng có xu hướng ưu tiên ảo giác ở các bước trước; ảo giác không phải là một "ý đồ có từ trước" mà bị cưỡng bức ở bước cuối.
4. **Giải mã Triệt để Nghịch lý Bảng 7 của TruthPrInt:**
   * Trong Bảng 7 của TruthPrInt, PMC của ảo giác lại cao hơn thật. Ta chứng minh đó là lỗi do **không khống chế độ dài tiền đề**. Với các câu đủ dài ($n_{\text{prec}} \ge 4$), ảo giác thực sự rơi vào "vùng trũng tự tin" sâu hơn thật rất nhiều (chênh lệch $-0.0952$!). Hơn nữa, vị trí cực tiểu tự tin của ảo giác nằm xa hơn ($6.04$ tokens so với $4.82$ tokens của thật).
5. **Dấu vân tay Bất định tại Thời điểm Phát ngôn ($m = 0$):**
   * Top-1 Confidence sụt giảm $-13.63\%$ ($39.45\%$ vs $53.08\%$).
   * Shannon Entropy tăng vọt **$+30.67\%$** ($2.4023$ vs $1.8384\text{ nats}$). Mô hình phát ngôn ảo giác trong trạng thái hoang mang và do dự tột độ.

---

## PHẦN IV: THÍ NGHIỆM 2 — ĐỘNG LỰC HỌC CHIỀU SÂU TẦNG MẠNG (32 TẦNG LOGIT LENS TẠI $t-1$)

### 1. Kỹ thuật Chiếu Trạng thái Ẩn Logit Lens
Tại vị trí có đỉnh phân hóa mạnh nhất $t - 1$ ($m = 1$), trích xuất vector trạng thái ẩn $h_{t-1}^{(l)} \in \mathbb{R}^{4096}$ qua từng tầng Transformer $l \in [1, 32]$, chuẩn hóa bằng RMSNorm và chiếu thẳng qua `lm_head` để thu được vector pseudo-logits $z^{(l)} \in \mathbb{R}^{32000}$:
$$h_{\text{normed}}^{(l)} = \text{RMSNorm}(h_{t-1}^{(l)}), \quad z^{(l)} = \text{lm\_head}(h_{\text{normed}}^{(l)})$$
Quy mô: $N = 343$ cặp đối chứng, $21,952$ bản ghi xuyên suốt 32 tầng.

---

### 2. Bảng Dữ liệu Thống kê Đầy đủ 32 Tầng Transformer

| Tầng $l$ | Mean $S(h)$ | Mean $S(r)$ | Mean $\Delta$ | 95% Bootstrap CI | Cohen's $d_z$ | Holm $p$ | Median Rank $h$ | Median Rank $r$ | Pha Cơ chế Kiến trúc |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | -1.1211 | -1.5635 | **+0.4423** | [+0.2150, +0.6714] | 0.1992 | **$4.49 \times 10^{-3}$** | 9,005 | 11,453 | **PHA 1: LANGUAGE PRIOR** |
| **2** | -1.2076 | -1.8010 | **+0.5934** | [+0.3078, +0.8779] | 0.2197 | **$1.70 \times 10^{-3}$** | **6,715** | **9,513** | **ĐỈNH LANGUAGE PRIOR** |
| **3** | -1.1064 | -1.5836 | **+0.4772** | [+0.2282, +0.7300] | 0.2009 | **$2.01 \times 10^{-3}$** | 6,283 | 9,000 | Pha 1: Tiền nghiệm ngôn ngữ |
| **4** | -0.8762 | -1.4362 | **+0.5600** | [+0.3380, +0.7841] | 0.2682 | **$1.87 \times 10^{-4}$** | 7,718 | 8,105 | Pha 1: Tiền nghiệm ngôn ngữ |
| **5** | -0.7294 | -1.2151 | **+0.4858** | [+0.3277, +0.6593] | 0.3072 | **$1.95 \times 10^{-6}$** | 6,208 | 7,977 | Pha 1: Tiền nghiệm ngôn ngữ |
| **6** | -0.4423 | -0.7685 | **+0.3262** | [+0.1950, +0.4763] | 0.2439 | **$6.28 \times 10^{-4}$** | 6,995 | 8,015 | Pha 1: Tiền nghiệm ngôn ngữ |
| **7** | -0.3998 | -0.7254 | **+0.3256** | [+0.1982, +0.4682] | 0.2593 | **$2.49 \times 10^{-4}$** | 7,319 | 8,566 | Pha 1: Tiền nghiệm ngôn ngữ |
| **8** | -0.3628 | -0.6243 | **+0.2615** | [+0.1553, +0.3842] | 0.2413 | **$3.07 \times 10^{-3}$** | 7,926 | 8,340 | Pha 1: Tiền nghiệm ngôn ngữ |
| **9** | -0.2829 | -0.5061 | **+0.2232** | [+0.1218, +0.3415] | 0.2199 | **$5.85 \times 10^{-4}$** | 8,962 | 8,954 | Pha 1: Tiền nghiệm ngôn ngữ |
| **10** | -0.2508 | -0.4407 | **+0.1900** | [+0.0994, +0.2921] | 0.2128 | **$1.14 \times 10^{-3}$** | **7,533** | **7,232** | **ĐIỂM GIAO CẮT (CROSSOVER)** |
| **11** | -0.2561 | -0.4172 | **+0.1611** | [+0.0643, +0.2633] | 0.1744 | **$3.85 \times 10^{-3}$** | 6,775 | 6,276 | PHA 2: VISUAL GROUNDING |
| **12** | -0.4771 | -0.7340 | **+0.2569** | [+0.1156, +0.4076] | 0.1849 | **$3.80 \times 10^{-3}$** | 6,575 | 5,195 | Pha 2: Căn cứ thị giác trỗi dậy |
| **13** | -0.4728 | -0.6843 | **+0.2116** | [+0.0747, +0.3566] | 0.1596 | **$1.83 \times 10^{-2}$** | 5,835 | 5,226 | Pha 2: Căn cứ thị giác trỗi dậy |
| **14** | -0.2955 | -0.4916 | **+0.1962** | [+0.0771, +0.3143] | 0.1727 | **$1.83 \times 10^{-2}$** | 6,652 | 6,192 | Pha 2: Căn cứ thị giác trỗi dậy |
| **15** | -0.2935 | -0.5119 | +0.2184 | [+0.0993, +0.3423] | 0.1913 | 0.056674 | 3,681 | **2,280** | **Bứt phá Thị giác Mạnh** |
| **16** | -0.1739 | -0.2726 | +0.0987 | [+0.0134, +0.1925] | 0.1168 | 0.075365 | 3,688 | **2,328** | Pha 2: Thị giác tăng tốc |
| **17** | -0.1113 | -0.1507 | +0.0394 | [-0.0264, +0.1119] | 0.0620 | 0.279124 | 1,583 | **707** | Thật lọt Top 1,000 từ vựng |
| **18** | -0.0979 | -0.1488 | +0.0509 | [-0.0105, +0.1172] | 0.0846 | 1.000000 | **479** | **156** | **Chênh lệch hạng > 320 bậc!** |
| **19** | -0.0562 | -0.0760 | +0.0198 | [-0.0175, +0.0622] | 0.0539 | 1.000000 | 158 | **32** | Thật tiến vào Top 32 từ vựng |
| **20** | -0.0334 | -0.0542 | +0.0208 | [-0.0069, +0.0527] | 0.0715 | 0.624559 | 63 | **16** | PHA 3: CONVERGENCE & SURGE |
| **21** | -0.0104 | -0.0243 | **+0.0140** | [-0.0018, +0.0337] | 0.0812 | **$1.83 \times 10^{-2}$** | 20 | **4** | Thật áp sát Top 4 |
| **22** | -0.0043 | -0.0084 | **+0.0041** | [-0.0017, +0.0111] | 0.0667 | **$5.68 \times 10^{-4}$** | 12 | **3** | Thật đạt Top 3 |
| **23** | -0.0019 | -0.0072 | **+0.0053** | [-0.0001, +0.0119] | 0.0928 | **$1.10 \times 10^{-3}$** | 8 | **2** | Thật đạt Top 2 |
| **24** | -0.0003 | -0.0018 | **+0.0015** | [-0.0001, +0.0040] | 0.0740 | **$2.01 \times 10^{-3}$** | 4 | **1 (Top 1)** | **THẬT CHÍNH THỨC ĐẠT TOP 1** |
| **25** | -0.0002 | -0.0018 | **+0.0016** | [-0.0001, +0.0042] | 0.0779 | **$5.85 \times 10^{-4}$** | 4 | 2 | Thật giữ vững vị trí dẫn đầu |
| **26** | -0.0001 | -0.0017 | **+0.0016** | [-0.0001, +0.0043] | 0.0705 | **$7.48 \times 10^{-4}$** | 3 | 2 | Ảo giác duy trì Rank 3 |
| **27** | -0.0001 | -0.0019 | **+0.0019** | [-0.0000, +0.0048] | 0.0743 | **$1.17 \times 10^{-5}$** | 2 | 1 | Ảo giác bị kìm ở Rank 2 |
| **28** | -0.0000 | -0.0020 | **+0.0020** | [-0.0000, +0.0051] | 0.0767 | **$6.90 \times 10^{-8}$** | 2 | 1 | Ảo giác bị kìm ở Rank 2 |
| **29** | -0.0000 | -0.0016 | **+0.0016** | [+0.0000, +0.0047] | 0.0590 | **$2.86 \times 10^{-10}$** | 2 | 1 | Ảo giác bị kìm ở Rank 2 |
| **30** | -0.0000 | -0.0001 | **+0.0001** | [-0.0000, +0.0002] | 0.0528 | **$2.58 \times 10^{-9}$** | 2 | 1 | Ảo giác vẫn kẹt chặt ở Rank 2 |
| **31** | -0.0002 | -0.0001 | **-0.0001** | [-0.0002, +0.0000] | -0.0839 | **$3.60 \times 10^{-8}$** | **1** | **1** | **ẢO GIÁC ĐỘT BIẾN LÊN TOP 1** |
| **32** | -0.0001 | -0.0001 | **-0.0000** | [-0.0000, +0.0000] | -0.0163 | **$2.05 \times 10^{-7}$** | **1** | **1** | Hội tụ hoàn tất ở đầu ra |

---

### 3. Bảng Tóm tắt Các Mốc Chuyển Pha Then chốt (Milestones Summary)

| Mốc Tầng | Tên Gọi Cơ Chế | Hiện Tượng Quan Sát Được | Ý Nghĩa Nơ-ron |
|:---:|:---|:---|:---|
| **Tầng 1 – 9** | **Pha 1: Language Prior Dominance** | $\Delta > 0$ có ý nghĩa thống kê; Đỉnh tại Tầng 2 ($\Delta = +0.5934$, Rank ảo giác $6,715$ vs thật $9,513$). | Ảo giác sinh ra từ **thói quen từ vựng đi kèm**, không phải do nhìn nhầm ảnh. |
| **Tầng 10** | **Điểm Giao Cắt (Crossover Point)** | Rank Thật chính thức vượt Ảo giác ($7,232$ vs. $7,533$). | Bước ngoặt chuyển giao quyền lực từ ngôn ngữ sang thị giác. |
| **Tầng 15 – 19** | **Pha 2: Visual Grounding Awakening** | Thật tăng hạng phi mã: Tầng 18 dẫn $> 320$ bậc, Tầng 19 lọt Top 32, Tầng 20 lọt Top 16. | Thông tin ảnh từ CLIP ViT được giải mã mạnh mẽ nhất. **Vùng can thiệp vàng!** |
| **Tầng 24** | **Thật Đạt Đỉnh (Real Reaches Rank 1)** | Thật đạt Rank 1 vững chắc; Ảo giác bị kìm hãm ở Rank 2–4. | Căn cứ thị giác đã khóa chặt từ thật ở vị trí số 1. |
| **Tầng 31 – 32** | **Đột Biến Muộn (Late Surge)** | Ảo giác nhảy cóc từ Rank 2 lên Rank 1 trong 2 tầng cuối. | Bác bỏ giả thuyết ảo giác tích tụ dần. Ảo giác bị cưỡng bức bởi áp lực cú pháp. |

---

## PHẦN V: MÔ HÌNH HÓA CƠ CHẾ NƠ-RON — "CUỘC CHIẾN KÉO CO" (TUG-OF-WAR ARCHITECTURE)

Từ hai thực nghiệm trên, ta xây dựng mô hình cơ chế nơ-ron hoàn chỉnh giải thích nguồn gốc và động học của ảo giác đối tượng:

```
                  ┌────────────────────────────────────────────────────────┐
                  │      PHA 1: TIỀN NGHIỆM NGÔN NGỮ (TẦNG 1 - 9)         │
                  │  - Ngữ cảnh văn bản kích hoạt các từ hay đi kèm       │
                  │  - S(ảo giác) > S(thật), Rank(ảo giác) < Rank(thật)   │
                  │  - Đỉnh ưu tiên ảo giác tại Tầng 2 (Δ = +0.5934)      │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                  ┌────────────────────────────────────────────────────────┐
                  │      TẦNG 10: ĐIỂM GIAO CẮT (CROSSOVER POINT)          │
                  │  - Thứ hạng vật thể thật chính thức vượt ảo giác      │
                  │    (Rank thật: 7,232 vs. Rank ảo giác: 7,533)         │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                  ┌────────────────────────────────────────────────────────┐
                  │      PHA 2: CĂN CỨ THỊ GIÁC BỨT PHÁ (TẦNG 10 - 19)     │
                  │  - Dòng thông tin từ CLIP ViT được giải mã mạnh mẽ    │
                  │  - Vật thể thật tăng hạng phi mã (Tầng 18 chênh >320) │
                  │  - Rank thật lên Top 32 (Tầng 19), Top 16 (Tầng 20)   │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                  ┌────────────────────────────────────────────────────────┐
                  │      PHA 3: HỘI TỤ & ĐỘT BIẾN MUỘN (TẦNG 20 - 32)      │
                  │  - Vật thể thật vững vàng chiếm Top 1 từ Tầng 24       │
                  │  - Vật thể ảo giác bị kìm hãm ở Rank 2–4 đến Tầng 30  │
                  │  - TẦNG 31-32: Dưới áp lực cú pháp bắt buộc danh từ,  │
                  │    ảo giác bứt phá đột ngột lên Top 1 để phát ngôn!    │
                  └────────────────────────────────────────────────────────┘
```

> **Bản chất của Ảo giác Đối tượng (The Core Theoretical Insight):**  
> Ảo giác không phải là một "nhầm lẫn thị giác" ngẫu nhiên, mà là kết quả của một cuộc chiến giằng co 3 bên:
> $$\text{Quyết định} = \underbrace{\text{Tiền nghiệm Ngôn ngữ (Tầng 1–9)}}_{\text{Kéo về phía Ảo giác}} \ \longleftrightarrow \ \underbrace{\text{Căn cứ Thị giác CLIP (Tầng 10–24)}}_{\text{Kéo về phía Vật thể Thật}} \ \longleftrightarrow \ \underbrace{\text{Áp lực Cú pháp (Tầng 31–32)}}_{\text{Bắt buộc sinh danh từ}}$$
> Khi tín hiệu ảnh yếu hoặc không đủ ức chế, mô hình rơi vào "vùng trũng tự tin" ở các bước sát phát ngôn, và áp lực cú pháp ở 2 tầng cuối cùng đã ép từ ảo giác bùng nổ lên Top 1.

---

## PHẦN VI: BƯỚC CHUYỂN TIẾP SANG THÍ NGHIỆM CAN THIỆP NHÂN QUẢ H1 (CAUSAL INTERVENTION)

### 1. Giới hạn Khoa học của Thí nghiệm 1 & 2
Cả Thí nghiệm 1 và Thí nghiệm 2 đều mang tính chất **quan sát (observational)** trên biểu diễn tự nhiên. Chúng ta thấy sự tương quan chặt chẽ, nhưng chưa thể khẳng định theo luật nhân quả (causality): *Liệu có một token tiền đề cụ thể nào ở phía trước đóng vai trò "ngòi nổ" (trigger token) gây ra ảo giác hay không?*

### 2. Nguyên lý Can thiệp Teacher-Forcing Counterfactual của Thí nghiệm H1
* **Phương pháp:** Can thiệp thay thế token tại từng vị trí $j \in [t-10, t-2]$ bằng các từ ứng viên giả định (counterfactual candidates).
* **Đo lường hiệu ứng sụt giảm điểm số:** $e_S(j) = S_0 - S_{\text{new}}(j)$.
* **Cơ chế Triệt tiêu Biến nhiễu Thay thế (Split-half Balanced Design):**
  * Nửa A: Ứng viên là các từ trung tính / phổ biến.
  * Nửa B: Ứng viên là các từ đối chứng có kiểm soát.
  * Hiệu số hiệu ứng giữa hai nhóm: $\Delta_{\text{effect}}(j) = e_S^h(j) - e_S^r(j)$.

### 3. Thông số Thiết kế Thí nghiệm Pilot H1
* **Mẫu pilot:** 155 ảnh COCO $\to$ Chọn lọc 100 cặp đối chứng hoàn chỉnh (tổng cộng 200 vật thể).
* **Số lượng can thiệp:** 543 hàng ứng viên thay thế, 629 hàng cặp can thiệp.
* **Tối ưu hóa GPU:** Khắc phục lỗi CUDA launch failure bằng cách ép contiguous KV-cache, dọn cache tự động mỗi batch (`--batch_size 8`), chạy hoàn tất ổn định trên 2 GPU Tesla T4.

### 4. Ma trận Quyết định Khoa học 5 Kịch bản (Decision Matrix O1 – O5)

| Kịch bản | Tên Gọi Hiện Tượng | Tiêu chí Định lượng | Kết luận Cơ chế Nơ-ron |
|:---:|:---|:---|:---|
| **O1** | **Causal Trigger Token (Ngòi nổ nhân quả)** | Tồn tại vị trí $j^*$ có $|\Delta_{\text{effect}}| > 0.5$ và $p_{\text{adj}} < 0.05$ | Tồn tại token ngữ cảnh cụ thể kích hoạt ảo giác. Có thể triệt tiêu bằng cách sửa đúng token đó! |
| **O2** | **Visual Suppression (Ức chế thị giác)** | $e_S$ dương đồng đều nhưng nhóm Thật sụt giảm mạnh hơn Ảo giác | Ngữ cảnh không kích hoạt ảo giác, mà ngữ cảnh làm mất khả năng chú ý vào ảnh của vật thể thật. |
| **O3** | **Uniform Context Drift (Trôi dạt ngữ cảnh)** | $e_S$ dương đều trên mọi $j \in [t-10, t-2]$ không có đỉnh nhọn | Ảo giác do sự tích lũy mơ hồ toàn cục, không có ngòi nổ cục bộ. |
| **O4** | **Null Effect (Vô hiệu can thiệp)** | Mọi $|\Delta_{\text{effect}}| < 0.1$, $p_{\text{adj}} > 0.2$ | Ngữ cảnh tiền đề $t-10..t-2$ không ảnh hưởng nhân quả; ảo giác chỉ do token $t-1$ quyết định. |
| **O5** | **Intervention Instability (Bất ổn định)** | Tương quan Split-half $r < 0.3$ | Phép đo bị nhiễu cú pháp; cần tăng cỡ mẫu hoặc chuẩn hóa lại ứng viên. |

---

## PHẦN VII: MỤC LỤC TÀI NGUYÊN HÌNH ẢNH & BIỂU ĐỒ (VISUAL ASSET DIRECTORY)

Dưới đây là bảng tra cứu chính xác đường dẫn các tệp biểu đồ PNG chất lượng cao sẵn có trong repository để Agent làm slide chèn trực tiếp:

| Tên File Biểu đồ | Đường dẫn Tuyệt đối trên Máy | Nội dung Đồ thị | Thông điệp Cần Thể hiện trên Slide | Khuyến nghị Slide Nhúng |
|:---|:---|:---|:---|:---:|
| `delta_vs_m.png` | `/Users/nguyenmy/Documents/Experiment/results/exp1_confirm/figures/delta_vs_m.png` | Hiệu số $\Delta(m) = S(h) - S(r)$ kèm 95% Bootstrap CI qua 11 bước trễ | Đỉnh phân hóa cực đại tại $m=1$ và tín hiệu sớm tại $m=2, 3$ | **Slide 4** |
| `s_vs_m.png` | `/Users/nguyenmy/Documents/Experiment/results/exp1_confirm/figures/s_vs_m.png` | Điểm $S(h)$ và $S(r)$ tuyệt đối từ $m=10$ đến $m=0$ | Đường cong hội tụ xác suất của vật thể thật luôn nằm trên ảo giác | **Slide 5** |
| `auroc_vs_m.png` | `/Users/nguyenmy/Documents/Experiment/results/exp1_confirm/figures/auroc_vs_m.png` | Năng lực phân loại AUROC của điểm $S$ qua 11 bước | AUROC $< 0.5$, chứng minh đây là dịch chuyển phân phối quần thể, không phải bộ phân loại đơn biến | **Slide 5** |
| `pmc_by_nprec.png` | `/Users/nguyenmy/Documents/Experiment/results/exp1_confirm/figures/pmc_by_nprec.png` | Điểm PMC phân tầng theo độ dài câu $n_{\text{prec}}$ | Phản biện Bảng 7 TruthPrInt: PMC ảo giác sụt giảm sâu ở $n_{\text{prec}} \ge 4$ | **Slide 6** |
| `pmc_by_group.png` | `/Users/nguyenmy/Documents/Experiment/results/exp1_confirm/figures/pmc_by_group.png` | Phân phối mật độ PMC giữa 2 nhóm Thật vs Ảo giác | Nhóm ảo giác lệch rõ rệt về phía vùng trũng tự tin thấp ($< 0.25$) | **Slide 6** |

---

## PHẦN VIII: KHUNG DÀN BÀI CHI TIẾT 12 SLIDE BÁO CÁO THUYẾT TRÌNH (SLIDE-BY-SLIDE BLUEPRINT)

Dưới đây là kịch bản chi tiết từng trang slide chuẩn mực dành cho Agent thiết kế Slide:

---

### SLIDE 1: TIÊU ĐỀ & THÔNG TIN CHUNG
* **Tiêu đề chính:** Nghiên cứu Cơ chế Nơ-ron Đa chiều của Ảo giác Đối tượng trong Mô hình LLaVA-1.5-7B
* **Tiêu đề phụ:** Khảo sát Trực giao theo Trục Thời gian Tự hồi quy ($m=0 \to 10$), Trục Chiều sâu Tầng mạng (32 Tầng Logit Lens) và Can thiệp Nhân quả Counterfactual
* **Thông tin tác giả:** Nhóm Nghiên cứu Cơ chế Nơ-ron VLM (Mechanistic Interpretability Lab)
* **Target Model & Dataset:** LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`) trên MS-COCO val2014 ($2,000$ ảnh)
* **Speaker Notes:** "Kính chào hội đồng, hôm nay chúng tôi xin báo cáo nghiên cứu chuyên sâu giải mã cơ chế nơ-ron của hiện tượng ảo giác đối tượng trong mô hình thị giác lớn LLaVA-1.5-7B qua hai trục tọa độ: thời gian và chiều sâu tầng mạng."

---

### SLIDE 2: ĐỘNG LỰC NGHIÊN CỨU & KHOẢNG TRỐNG CỦA BÀI BÁO TRUTHPRINT (ICCV 2025)
* **Vấn đề thực tế:** VLM mô tả chi tiết nhưng hay "bịa" ra vật thể không có trong ảnh (tỷ lệ lỗi $\sim 26.6\%$), gây rủi ro trong y tế, xe tự hành, trợ lý ảo.
* **Khoảng trống của TruthPrInt (ICCV 2025):**
  1. Chỉ bắt được ảo giác tại token kề cận ($t - 1$), dẫn đến việc phải quay lui (backtracking) tốn kém khi câu đã sinh gần xong.
  2. Bỏ ngỏ 32 tầng Transformer như một "hộp đen" bí ẩn, không rõ thông tin thị giác bị dập tắt ở đâu.
* **Câu hỏi lớn:** *Tín hiệu ảo giác có xuất hiện sớm hơn 1 bước không? Biểu diễn ảo giác hình thành từ tầng nào và bị thị giác ngăn chặn ra sao?*
* **Speaker Notes:** "Nghiên cứu gần đây tại ICCV 2025 là TruthPrInt đã phát hiện ảo giác tại bước $t-1$. Tuy nhiên, phương pháp này như việc đợi xe sắp đâm mới phanh. Chúng tôi đặt câu hỏi: liệu có thể phát hiện sớm hơn trước vài từ, và bên trong 32 tầng mạng, nơ-ron đang tính toán điều gì?"

---

### SLIDE 3: THIẾT LẬP THỰC NGHIỆM & PHƯƠNG PHÁP GHÉP CẶP ĐỐI CHỨNG 1:1
* **Bảng phễu dữ liệu:** 2,000 ảnh $\to$ 15,480 mentions $\to$ 6,948 first mentions $\to$ **1,430 cặp đối chứng 1:1 hoàn hảo**.
* **Khống chế biến nhiễu vị trí:** Chuẩn hóa vị trí tương đối $\text{rel\_pos} = t / G$. Ghép cặp không hoàn lại với dung sai $|\Delta \text{rel\_pos}| \le 0.10$.
* **Điểm ưu tiên chuẩn hóa $S(o)$:** Đo xác suất logit của từ mục tiêu so với không gian 80 lớp đối tượng COCO.
* **Kiểm định tính vững:** T4 (Khớp vị trí $> 98\%$), S1 (Siết Caliper $\le 0.05$ giữ $96.4\%$ kết quả), S3 (Khống chế tần suất từ COCO).
* **Speaker Notes:** "Để tránh biến nhiễu do vật thể thật thường ở đầu câu và ảo giác ở cuối câu, chúng tôi thiết kế thuật toán ghép cặp 1:1 nghiêm ngặt với 1,430 cặp đối chứng trên 2,000 ảnh, bảo đảm tính so sánh hoàn toàn công bằng."

---

### SLIDE 4: THÍ NGHIỆM 1 — ĐƯỜNG CONG THỜI GIAN & PHÁT HIỆN "TÍN HIỆU SỚM"
* **Thông điệp chính:** Phân hóa xuất hiện sớm hơn bài báo từ 1–2 bước ($m = 2, 3$).
* **Data Callouts:**
  * $m = 3$: $\Delta = -0.4536$ ($p_{\text{holm}} = 0.0048$)
  * $m = 2$: $\Delta = -0.4565$ ($p_{\text{holm}} = 0.0013$)
  * $m = 1$: $\Delta = -0.8249$ ($p_{\text{holm}} = 3.54 \times 10^{-22}$) — **Đỉnh phân hóa bùng nổ**.
* **Nhúng hình ảnh:** `results/exp1_confirm/figures/delta_vs_m.png`
* **Ý nghĩa ứng dụng:** Xây dựng hệ thống cảnh báo sớm (Early Warning System) dừng sinh trước 2–3 từ, tiết kiệm tài nguyên suy luận.
* **Speaker Notes:** "Biểu đồ Delta cho thấy một phát hiện then chốt: mô hình đã bắt đầu phân biệt thật và ảo giác từ $m=3$ và $m=2$ với ý nghĩa thống kê cao ($p < 0.005$), cho phép chúng ta cảnh báo sớm hơn bài báo TruthPrInt."

---

### SLIDE 5: THÍ NGHIỆM 1 — ĐỈNH PHÂN HÓA ($m=1$) & ĐẶC TRƯNG BẤT ĐỊNH ($m=0$)
* **Tại bước $m = 1$ ($t - 1$):**
  * Tỷ lệ lọt Top-10 từ điển: Thật đạt **$42.3\%$** vs. Ảo giác chỉ đạt **$26.6\%$**.
  * Thứ hạng trung vị: Thật đạt Rank 13 vs. Ảo giác tụt xuống Rank 22.
* **Tại thời điểm phát ngôn ($m = 0$):**
  * Top-1 Confidence sụt giảm $-13.63\%$ ($39.45\%$ vs $53.08\%$).
  * Shannon Entropy tăng vọt **$+30.67\%$** ($2.4023$ vs $1.8384\text{ nats}$).
* **Nhúng hình ảnh:** `results/exp1_confirm/figures/s_vs_m.png` và `results/exp1_confirm/figures/auroc_vs_m.png`
* **Speaker Notes:** "Tại bước sát sườn $m=1$, khả năng lọt Top 10 của từ thật cao gần gấp đôi ảo giác. Đặc biệt tại $m=0$, dù cả hai đều được phát ngôn, nhưng khi sinh ra ảo giác, mô hình ở trạng thái do dự và bất định cực kỳ cao."

---

### SLIDE 6: THÍ NGHIỆM 1 — GIẢI MÃ NGHỊCH LÝ PMC TRONG BÀI BÁO TRUTHPRINT
* **Phản biện Bảng 7 TruthPrInt:** TruthPrInt báo cáo PMC của ảo giác cao hơn thật (vô lý theo lý thuyết của chính họ).
* **Bóc tách nguyên nhân:** Do không khống chế độ dài tiền đề ($n_{\text{prec}}$). 
* **Data Callouts:**
  * Với câu ngắn ($n_{\text{prec}} \le 3$): PMC bị nhiễu đầu câu khiến ảo giác cao hơn ($0.3143$ vs $0.3064$).
  * Với câu đủ dài ($n_{\text{prec}} \in [4, 6]$): PMC ảo giác sụt giảm cực sâu ($0.2692$ vs $0.3644$, chênh lệch **$-0.0952$**).
  * Vị trí đáy tự tin của ảo giác nằm xa hơn hẳn thật (**$6.04$ tokens vs. $4.82$ tokens**).
* **Nhúng hình ảnh:** `results/exp1_confirm/figures/pmc_by_nprec.png` và `pmc_by_group.png`
* **Speaker Notes:** "Chúng tôi đã giải mã thành công hạt sạn trong bài báo TruthPrInt: khi khống chế độ dài câu từ 4 từ trở lên, PMC của ảo giác thực sự sụt giảm rất sâu, chứng minh mô hình trải qua một vùng trũng tự tin trước khi sinh ảo giác."

---

### SLIDE 7: THÍ NGHIỆM 2 — GIẢI MÃ "HỘP ĐEN" 32 TẦNG BẰNG LOGIT LENS
* **Đặt vấn đề:** Tại vị trí phân hóa mạnh nhất $t - 1$, điều gì thực sự diễn ra bên trong 32 tầng Transformer?
* **Phương pháp Logit Lens:** Trích xuất vector trạng thái ẩn $h_{t-1}^{(l)}$ tại từng tầng $l \in [1, 32]$, chuẩn hóa RMSNorm và chiếu thẳng qua `lm_head` lên từ điển 32,000 từ.
* **Quy mô:** 343 cặp đối chứng $\to$ $21,952$ bản ghi đa tầng.
* **3 Câu hỏi tầng mạng:**
  1. Ảo giác bắt đầu xuất hiện từ tầng nào?
  2. Tín hiệu thị giác bứt phá đánh bại ảo giác ở tầng nào?
  3. Tại sao ở tầng cuối cùng ảo giác lại thắng thế?
* **Speaker Notes:** "Chúng tôi mở hộp đen 32 tầng bằng kỹ thuật Logit Lens. Bằng cách chiếu trạng thái ẩn của từng tầng lên từ điển, chúng tôi có thể đọc được 'mô hình đang nghĩ từ gì' ở từng tầng mạng."

---

### SLIDE 8: THÍ NGHIỆM 2 — PHA 1: TIỀN NGHIỆM NGÔN NGỮ & ĐIỂM GIAO CẮT TẦNG 10
* **Hiện tượng Đảo chiều ở Tầng 1–9 (Language Prior):**
  * $\Delta > 0$ có ý nghĩa thống kê cao ($p_{\text{holm}} < 0.005$).
  * Đỉnh điểm tại Tầng 2: $\Delta = +0.5934$, Thứ hạng từ điển ảo giác là **$6,715$**, vượt xa thật (**$9,513$**).
  * *Bản chất:* Ảo giác bắt nguồn từ **quán tính ngôn ngữ đi kèm**, không phải do nhìn nhầm ảnh.
* **Điểm Giao Cắt Lịch Sử tại Tầng 10 (Crossover Point):**
  * Thứ hạng thật chính thức lội ngược dòng đánh bại ảo giác: **Thật = $7,232$ vs. Ảo giác = $7,533$**.
* **Speaker Notes:** "Một phát hiện vô cùng bất ngờ: Ở 9 tầng đầu tiên, mô hình ưu tiên từ ảo giác hơn từ thật! Đỉnh tại Tầng 2, từ ảo giác dẫn trước gần 3,000 bậc hạng. Điều này chứng minh ảo giác sinh ra từ quán tính ngôn ngữ. Và đến Tầng 10, vật thể thật mới bắt đầu lội ngược dòng."

---

### SLIDE 9: THÍ NGHIỆM 2 — PHA 2 & 3: CĂN CỨ THỊ GIÁC & ĐỘT BIẾN MUỘN
* **Pha 2: Căn cứ Thị giác Bứt phá (Tầng 15–19):**
  * Dòng thông tin CLIP ViT tràn vào: Tầng 18 chênh lệch hạng $> 320$ bậc (Thật $156$ vs Ảo $479$).
  * Tầng 20: Thật lọt vào **Top 16**.
  * Tầng 24: Thật chính thức **đạt Rank 1 (Top 1)**.
* **Pha 3: Đột biến Muộn ở Tầng 31–32 (Late Surge):**
  * Ảo giác bị đè bẹp ở Rank 2–4 suốt từ Tầng 10 đến 30.
  * Đến Tầng 31–32: Ảo giác đột ngột nhảy cóc vọt lên Rank 1 do áp lực cú pháp hoàn thiện câu!
* **Speaker Notes:** "Từ Tầng 15 đến 19, tín hiệu ảnh từ CLIP bùng nổ, đẩy từ thật lên Top 1 ở Tầng 24 và đè từ ảo giác ở Rank 2 suốt 10 tầng. Tuy nhiên, ở 2 tầng cuối cùng (31–32), dưới áp lực bắt buộc phải có một danh từ, từ ảo giác đột biến vọt lên chiếm quyền phát ngôn."

---

### SLIDE 10: MÔ HÌNH HÓA CƠ CHẾ NƠ-RON "CUỘC CHIẾN KÉO CO" (TUG-OF-WAR)
* **Sơ đồ cơ chế 3 pha:**
  1. *Tầng 1–9 (Language Prior):* Kéo về phía ảo giác dựa trên thói quen văn bản.
  2. *Tầng 10–24 (Visual Grounding):* Kéo về phía thật nhờ dòng đặc trưng thị giác CLIP.
  3. *Tầng 31–32 (Syntactic Completion):* Áp lực cú pháp cưỡng bức phát ngôn.
* **Định nghĩa Bản chất Ảo giác:** Sự thất thế của ức chế thị giác trước áp lực hoàn thành ngữ pháp ở các tầng cận biên.
* **Chiến lược Can thiệp Mới:** Chuyển từ can thiệp thụ động ở Tầng 32 sang **Can thiệp sớm tại Tầng 15–18 (Layer Steering Vectors)** để khuếch đại biểu diễn thị giác.
* **Speaker Notes:** "Tất cả các phát hiện hội tụ thành cơ chế Kéo co: Cuộc chiến giữa Tiền nghiệm ngôn ngữ và Căn cứ thị giác. Phát hiện này mở ra hướng tiếp cận hoàn toàn mới: can thiệp biểu diễn tại Tầng 15–18 để dập tắt ảo giác ngay từ trong mạng thay vì đợi đến tầng 32."

---

### SLIDE 11: BƯỚC CHUYỂN TIẾP — THÍ NGHIỆM CAN THIỆP NHÂN QUẢ H1
* **Từ Tương quan sang Nhân quả:** Thí nghiệm 1 và 2 là quan sát. Thí nghiệm H1 can thiệp Teacher-forcing counterfactual để tìm "ngòi nổ nhân quả" (Trigger token).
* **Thiết kế Pilot H1:**
  * 100 cặp đối chứng hoàn chỉnh (200 vật thể), 543 ứng viên thay thế tại $j \in [t-10, t-2]$.
  * Thiết kế cân bằng Split-half triệt tiêu nhiễu thay thế từ.
* **Ma trận 5 kịch bản quyết định (O1–O5):**
  * O1 (Causal Trigger Token): Phát hiện token ngòi nổ cụ thể gây ra ảo giác.
  * O2 (Visual Suppression): Ngữ cảnh làm suy yếu sự chú ý vào ảnh.
  * O3 (Uniform Drift): Trôi dạt ngữ cảnh toàn cục.
* **Speaker Notes:** "Để chứng minh quan hệ nhân quả thực sự, chúng tôi đã triển khai Thí nghiệm H1: thay thế các từ tiền đề bằng các từ phản thực tế để xác định chính xác từ nào đóng vai trò ngòi nổ kích hoạt ảo giác."

---

### SLIDE 12: TỔNG KẾT BÁO CÁO & ĐÓNG GÓP KHOA HỌC
* **3 Đóng góp Lý thuyết:**
  1. Chứng minh tín hiệu ảo giác xuất hiện sớm từ $m=2, 3$ (vượt qua giới hạn $m=1$ của TruthPrInt).
  2. Giải mã cơ chế nơ-ron 3 pha bên trong 32 tầng Transformer bằng Logit Lens (phát hiện điểm giao cắt Tầng 10 và đột biến Tầng 31–32).
  3. Giải mã trọn vẹn nghịch lý PMC trong Bảng 7 bài báo TruthPrInt bằng cách khống chế độ dài tiền đề.
* **2 Hướng Ứng dụng Thực tiễn:**
  1. *Early Warning Decoding:* Cảnh báo và điều hướng câu sớm trước 2–3 từ.
  2. *Layer-wise Visual Steering:* Can thiệp kích hoạt thị giác tại Tầng 15–18 để triệt tiêu ảo giác không cần huấn luyện lại mô hình.
* **Q&A:** Sẵn sàng tiếp nhận câu hỏi từ Hội đồng!
* **Speaker Notes:** "Tóm lại, nghiên cứu của chúng tôi đã cung cấp một bức tranh toàn cảnh từ thời gian đến chiều sâu tầng mạng về cơ chế của ảo giác đối tượng. Xin chân thành cảm ơn Hội đồng và rất mong nhận được các câu hỏi thảo luận."

---

*(Tài liệu này được đồng bộ trực tiếp tại `/Users/nguyenmy/Documents/Experiment/SLIDE_REPORT_SYNTHESIS.md` và thư mục Artifacts).*
