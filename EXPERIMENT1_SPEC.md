# ĐẶC TẢ THÍ NGHIỆM 1: "Object đã xuất hiện trước đó bao nhiêu bước?"

Tài liệu này dành cho **agent lập trình** triển khai và chạy thử Thí nghiệm 1. Người viết (Claude) **chưa chạy được** code trên mô hình thật. Mọi con số về thời gian/quy mô là **ước lượng**, cần đo lại.

---

## 0. Đọc trước (TL;DR cho agent)

**Việc cần làm:** với caption do LLaVA-1.5-7B sinh (greedy) cho ảnh COCO, đo xem xác suất của một object, khi đọc tại các vị trí *sớm hơn* bước sinh thật của nó, khác nhau thế nào giữa object **hallucinated** và object **thật** (đã ghép cặp theo vị trí trong caption). Kèm một phân tích phụ tái hiện thước đo "PMC" của paper TruthPrInt.

**Điều phải giữ:**
1. Làm đúng định nghĩa ở Mục 4 (đặc biệt chỉ số độ trễ `m` và chuẩn hóa `S`).
2. Chạy theo thứ tự các cổng (gate) ở Mục 8. **Không** qua cổng bằng cách nới ngưỡng.
3. Đây là thí nghiệm **mô tả tương quan ở đầu ra**. **Không** được kết luận nhân quả (xem Mục 10).
4. Không bịa số. Nếu thiếu GPU, mạng, hoặc dữ liệu: dừng và báo cáo chính xác thiếu gì (Mục 10).

**Đã có code khởi tạo** trong `src/` (`common.py`, `01_generate.py`, `02_label_chair.py`, `03_lag_curve.py`). Code này **chưa được chạy trên mô hình thật** và có các điểm yếu đã biết ở Mục 11. Bạn được phép sửa hoặc viết lại, nhưng **giữ nguyên schema file ở Mục 5** và ghi mọi thay đổi vào `CHANGELOG.md`.

---

## 1. Bối cảnh ngắn

Dự án nghiên cứu giảm hallucination object trong captioning **sớm**, từ các token đứng trước object. Paper TruthPrInt (ICCV 2025, arXiv 2503.10602) dùng detector đọc hidden state của token **liền trước** object (cảnh báo sớm một bước) và quay lui tới token "độ tin cậy thấp nhất" trong câu. Câu hỏi của dự án: tín hiệu cho thấy một object sắp bị hallucinate có xuất hiện **sớm hơn một bước** không?

Thí nghiệm 1 là bước rẻ nhất: chỉ cần teacher-forcing caption đã sinh, không cần patching hay huấn luyện. Nó trả lời: *ở đầu ra mô hình, object hallucinated có "lộ diện" sớm hơn object thật không?*

---

## 2. Câu hỏi, giả thuyết và cách diễn giải (đặt trước khi nhìn dữ liệu)

**Câu hỏi A (chính):** với độ trễ `m = 0..K`, điểm `S` (Mục 4) của object hallucinated khác object thật (cặp đối chứng theo vị trí) thế nào?

**Câu hỏi B (phụ):** tái hiện PMC (Preceding Minimum Confidence) của TruthPrInt trên dữ liệu của chúng ta; hallucinated và thật khác nhau theo chiều nào? (Bảng 7 trong paper cho PMC của hallucinated **cao hơn** thật, trái với câu chữ "token confidence thấp đứng trước hallucination". Cần kiểm tra lại.)

**Câu hỏi C (tùy chọn, sau khi A và B xong):** bỏ ảnh (ảnh xám) thì `S` thay đổi thế nào? Tách đóng góp của language prior khỏi thị giác.

### Tiêu chí diễn giải (đã cố định)

Dùng kiểm định hai phía, hiệu chỉnh Holm trên `m = 0..K` (K+1 phép thử).

| Kết quả | Định nghĩa | Gợi ý diễn giải |
|---|---|---|
| **Tín hiệu sớm** | Holm-p < 0,05 tại **ít nhất hai giá trị m liên tiếp** trong `m ≥ 2` | Object hallucinated khác thật ở đầu ra từ sớm hơn 1 bước |
| **Chỉ ở bước cuối** | Có ý nghĩa tại `m = 0` (và có thể `m = 1`) nhưng **không** có tại `m ≥ 2` | Phù hợp giả thuyết "chỉ xuất hiện ngay trước" |
| **Không tín hiệu** | Không m nào có ý nghĩa | Tín hiệu đầu ra không đủ phân biệt; cần probe hidden state |
| **Ngược chiều** | Có ý nghĩa nhưng hallucinated có `S` **cao hơn** | Báo cáo trung thực, phân tích theo loại object, không "sửa" giả thuyết |

Không được đổi `K`, caliper hay định nghĩa `S` sau khi đã thấy kết quả. Mọi phân tích thêm phải ghi là **exploratory**.

---

## 3. Môi trường và đầu vào

| Mục | Yêu cầu |
|---|---|
| Phần cứng | 1 GPU ≥ 24 GB (LLaVA-1.5-7B ở fp16) |
| Mô hình | `llava-hf/llava-1.5-7b-hf` |
| Thư viện | `torch`, `transformers` (≥ 4.40), `accelerate`, `pillow`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm`, `pytest` |
| Dữ liệu COCO 2014 | `val2014/` (ảnh), `instances_val2014.json`, `captions_val2014.json` |
| Synonyms | `data/synonyms.txt` kiểu CHAIR (mỗi dòng một nhóm, từ đầu là tên category COCO) |
| Seed | 0 (cố định cho chọn ảnh, ghép cặp, bootstrap) |

**Nếu thiếu `synonyms.txt`:** **không** tự bịa. Có thể tạo bản dự phòng chỉ gồm tên 80 category COCO và dạng số nhiều, **đặt tên `synonyms_FALLBACK.txt`**, và in cảnh báo rõ ràng rằng CHAIR_I sẽ bị đẩy cao, kết quả **không dùng cho báo cáo chính**.

**Ghi lại** (vào `run_manifest.json`): phiên bản `transformers`/`torch`/CUDA, tên GPU, git commit, toàn bộ tham số CLI, seed, thời gian chạy từng bước, hash (sha256) của `captions.jsonl` và `labels.jsonl`.

---

## 4. Ký hiệu và định nghĩa (phần quan trọng nhất)

### 4.1. Chuỗi token

- Prompt cố định: `"USER: <image>\nPlease describe this image in detail. ASSISTANT:"`.
- Caption sinh bằng **greedy** (`do_sample=False`), `max_new_tokens=512`, bỏ EOS cuối.
- Gọi các token **đã sinh** là `y_0, y_1, …, y_{G-1}` (không tính prompt).
- `dist_i := softmax(logits_i)` là phân phối next-token **dùng để chọn `y_i`**, tức `P(· | ảnh, prompt, y_{<i})`.
  - `dist_i` được tính từ hidden state tại vị trí của token `y_{i-1}` (hoặc token cuối của prompt nếu `i = 0`).

### 4.2. Object mention và độ trễ

- Một object mention nằm ở token `t` (token **đầu tiên** của từ object), `o := y_t`.
- **Độ trễ `m ∈ {0, 1, …, K}`** (mặc định `K = 10`): đọc phân phối `dist_{t-m}` và xét xác suất của `o` trong phân phối đó.
  - `m = 0`: bước dự đoán thật sự của `o` (hidden state của token liền trước object). Đây là trạng thái mà detector của TruthPrInt dùng.
  - `m = 1`: sớm hơn một token, v.v.
- Ý nghĩa: `P(o | y_{<t-m})` là xác suất để `o` là token **kế tiếp** tại thời điểm `t-m` (không phải xác suất `o` xuất hiện đúng ở vị trí `t`). Nó đo "mô hình đã nghĩ tới `o` chưa".
- **Ví dụ minh họa (tránh lệch 1):** `y = [The, image, shows, a, dog, …]`, `t = 4`.
  - `m = 0` dùng `dist_4`, tính sau khi đọc `a` (và là phân phối chọn ra `dog`).
  - `m = 1` dùng `dist_3`, tính sau khi đọc `shows` (phân phối chọn ra `a`).
  - `m = 2` dùng `dist_2`, v.v.
- Cần `t ≥ K` để mọi độ trễ tồn tại.

### 4.3. Các đại lượng ghi cho mỗi `(object, m)`

Cho logits `z = logits_{t-m}` (float32):

| Tên | Định nghĩa |
|---|---|
| `logp` | `log_softmax(z)[o]` |
| `rank` | `1 + #{v : z[v] > z[o]}` |
| `S` | `logp − logsumexp_{v ∈ V_obj ∪ {o}} log_softmax(z)[v]` |
| `conf` | `max_v softmax(z)[v]` (độ tin cậy top-1) |
| `ent` | entropy của `softmax(z)` (nats) |
| `logp_actual` | `log_softmax(z)[y_{t-m}]` (xác suất của token thực sự được chọn tại `t-m`) |

**`V_obj`:** tập id token đầu của **mọi từ trong `synonyms.txt`** (với cụm nhiều từ, lấy từ đầu tiên của cụm). Phải kiểm tra tokenizer (Mục 6.4).

**Vì sao dùng `S`:** các từ như "a", "to", "including" mở ra vị trí cho danh từ. `logp` thô sẽ trộn hai thứ: "mô hình sắp sinh danh từ" và "danh từ đó là `o`". `S` chuẩn hóa theo tập token object nên đo chủ yếu **"trong các object, `o` được ưu ái cỡ nào"**. Chuẩn hóa này chưa loại bỏ hoàn toàn tần suất của category; xem phân tích nhạy cảm S3 ở Mục 6.5.

### 4.4. Nhãn hallucinated / thật

- `GT(image)` = {category trong `instances_val2014` qua `syn2canon`} ∪ {object nhắc trong 5 caption người viết, trích bằng cùng hàm `find_mentions`}.
- Một mention là **hallucinated** nếu `canon ∉ GT(image)`, ngược lại là **thật**.
- Mỗi object chỉ dùng **mention đầu tiên** của `canon` trong caption (`first = true`). (CHAIR_S/CHAIR_I vẫn tính trên **mọi** mention.)

> Đây là CHAIR **đơn giản hóa**. Nhãn có nhiễu (COCO bỏ sót object có thật). Phải ước lượng tỷ lệ nhiễu bằng kiểm tay (Mục 8, Cổng 3).

### 4.5. Ghép cặp đối chứng

Hallucination hay xảy ra ở cuối caption, nên không so với tất cả object thật. Với mỗi object hallucinated `h`, chọn **một** object thật `r`:
- `r` chưa được dùng (ghép 1:1, không hoàn lại),
- `|rel_pos(r) − rel_pos(h)| ≤ 0,1` (caliper), với `rel_pos = t / G`,
- chọn `r` có chênh lệch nhỏ nhất; nếu hòa, chọn ngẫu nhiên theo seed,
- thứ tự duyệt các `h` được xáo ngẫu nhiên theo seed.

Cả `h` và `r` phải có `t ≥ K`. Mỗi cặp có `pair_id`. `h` không ghép được thì bị loại (và được đếm trong funnel).

---

## 5. Đặc tả file dữ liệu (giữ nguyên schema)

### `data/captions.jsonl` (mỗi dòng một ảnh)
```json
{"image_id": 123, "file_name": "COCO_val2014_000000000123.jpg",
 "gen_ids": [450, 1967, ...], "pieces": ["▁The", "▁image", ...]}
```
`pieces[i]` = `tokenizer.convert_ids_to_tokens(gen_ids[i])`. Không chứa EOS.

### `data/labels.jsonl` (mỗi dòng một ảnh)
```json
{"image_id": 123, "file_name": "...", "gen_ids": [...],
 "mentions": [{"canon": "dog", "word": "dogs", "tok_idx": 17, "tok_id": 11203,
               "halluc": false, "first": true, "rel_pos": 0.12}]}
```

### `results/exp1/lag_records.csv` (dạng dài)
`pair_id, image_id, canon, group (halluc|real), t, G, rel_pos, m, logp, rank, S, conf, ent, logp_actual`

### `results/exp1/summary.csv` (mỗi dòng một m)
`m, n_pairs, mean_S_halluc, mean_S_real, mean_delta, delta_ci_lo, delta_ci_hi, dz, wilcoxon_p, holm_p, frac_halluc_higher, auroc_S, auroc_ci_lo, auroc_ci_hi, top10_halluc, top10_real, median_rank_halluc, median_rank_real`

### `results/exp1/pmc_records.csv`
`image_id, canon, halluc, t, n_prec, pmc, argmin_dist`

### `results/exp1/funnel.json`
Số lượng ở từng tầng lọc (Mục 6.3).

### `results/exp1/run_manifest.json`, `REPORT.md`, `figures/*.png`

---

## 6. Thuật toán chi tiết

### 6.1. Sinh caption (`01_generate.py`)

1. Lấy danh sách ảnh từ `instances_val2014.json`, sắp theo `id`, xáo bằng `random.Random(seed)`.
2. **Chia tập không giao nhau theo vị trí sau khi xáo:**
   - `dev`: vị trí `[0, 500)` (dùng để debug và gate),
   - `confirm`: vị trí `[500, 2500)` (chạy xác nhận, chỉ khi dev đã qua mọi gate).
   - Thêm tham số `--offset` và `--n_images` để chọn đúng đoạn.
3. Với mỗi ảnh: `processor(images=img, text=PROMPT, return_tensors="pt").to("cuda", torch.float16)`, rồi `generate(do_sample=False, max_new_tokens=512)`. Lấy phần token sau độ dài `input_ids` của prompt, bỏ EOS cuối, ghi một dòng JSON.
4. Hỗ trợ **chạy lại** (bỏ qua `image_id` đã có), `flush` sau mỗi dòng.

### 6.2. Gán nhãn (`02_label_chair.py`) và hàm trích mention (`common.py`)

`find_mentions(text, syn2canon)`:
- Tách từ bằng `[A-Za-z]+`, hạ chữ thường.
- Ưu tiên cụm 2 từ liền kề (cách đúng một dấu cách), rồi mới đến 1 từ.
- Chỉ **số nhiều hóa/số ít hóa** ở từ cuối: thử từ gốc, rồi `-ies→-y`, `-es→-`, `-s→-`.

`pieces_to_text(pieces)`: ghép `pieces`, thay `▁` bằng dấu cách, byte-token `<0x0A>` thành xuống dòng, `<0xNN>` khác thành dấu cách; trả về `text` và `spans[i] = (start, end)` ký tự của token `i`.

`char_to_token(spans, c)`: token chứa ký tự `c`.

Với mỗi mention: `tok_idx = char_to_token(spans, char_start)`.
- **Kiểm tra word-start:** `spans[tok_idx].start` phải bằng `char_start − 1` (có dấu cách đứng trước) hoặc `char_start` (đầu text/sau xuống dòng). Nếu không, **loại** mention và đếm vào funnel (`dropped_not_word_start`).

In ra `CHAIR_S`, `CHAIR_I` (trên mọi mention).

### 6.3. Chọn object và funnel (`03_lag_curve.py`, hàm `select_objects`)

Ghi `funnel.json` với các đếm theo thứ tự:
```
captions → mentions → first_mentions → (halluc, real) → t>=K → word_start_valid → matched_pairs
```
Cảnh báo (không dừng) nếu `matched_pairs < 30`; ghi rõ trong báo cáo nếu `< 200` là **thiếu công suất**.

### 6.4. Forward teacher-forcing và thu thập

Gom object theo `image_id` để mỗi ảnh chỉ forward **một lần**:

```python
inputs = processor(images=img, text=PROMPT, return_tensors="pt").to("cuda", torch.float16)
gen = torch.tensor([gen_ids], device="cuda")
ids = torch.cat([inputs["input_ids"], gen], dim=1)
logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
               pixel_values=inputs["pixel_values"]).logits[0]
G, Lout = gen.shape[1], logits.shape[0]
pred = logits[Lout - G - 1 : Lout - 1].float()   # pred[i] = logits dùng để chọn y_i
```

Dùng `Lout = logits.shape[0]` (không dùng `ids.shape[1]`) vì một số phiên bản `transformers` mở rộng token ảnh bên trong mô hình.

**Kiểm tra tokenizer cho `V_obj`:**
- Với ≥ 20 từ mẫu, `tok.convert_ids_to_tokens(tok.encode(w, add_special_tokens=False)[0])` phải bắt đầu bằng `▁`. Nếu không, đổi sang cách ghép `"▁" + w` rồi `convert_tokens_to_ids`, và ghi vào `CHANGELOG.md`.
- In 10 từ ví dụ và id tương ứng vào log.

Với mỗi object và mỗi `m = 0..K`: tính các đại lượng ở Mục 4.3 tại `pred[t − m]`. Chỉ tính trên `V_obj ∪ {o}` cho phần `logsumexp` của `S`.

### 6.5. Thống kê (`summarize`)

**Phân tích chính (theo cặp):**
- Với mỗi `m`, `Δ_p(m) = S(h_p, m) − S(r_p, m)`.
- Trung bình `Δ`, CI 95% bằng **paired bootstrap** (resample cặp, `B = 2000`, seed cố định).
- `dz = mean(Δ) / sd(Δ)`.
- Kiểm định Wilcoxon signed-rank hai phía (`scipy.stats.wilcoxon`), sau đó **hiệu chỉnh Holm** trên `m = 0..K`.
- `frac_halluc_higher` = tỷ lệ cặp có `S(h) > S(r)`.
- `auroc_S` (nhãn hallucinated, điểm = `S`), CI bằng bootstrap. 0,5 là ngẫu nhiên; < 0,5 nghĩa là ngược chiều so với điểm `S`.
- `top10_*` = tỷ lệ `rank ≤ 10`. **Lưu ý:** vì sinh bằng greedy, **tại `m = 0` rank luôn bằng 1** (xem kiểm thử T4); các chỉ số dựa trên rank chỉ có ý nghĩa với `m ≥ 1`.

**Độ ổn định ghép cặp:** lặp lại ghép cặp với seed `0..19`, báo cáo trung bình và độ lệch chuẩn của `mean_delta(m)` qua các lần ghép.

**Phân tích nhạy cảm (đánh dấu rõ trong báo cáo):**
- **S1:** caliper 0,05 thay vì 0,1.
- **S2:** dùng **toàn bộ** object thật (không ghép), so với hallucinated bằng AUROC; kỳ vọng bị nhiễu bởi vị trí.
- **S3:** trừ trung bình theo `canon` (hiệu ứng cố định theo category, chỉ cho category có ≥ 3 object): `S_c = S − mean_{cùng canon, cùng m}(S)`. Diễn giải thận trọng vì hiệu ứng category một phần là tín hiệu thật.

**Hình:**
1. `S` trung bình theo `m` cho hai nhóm (kèm CI).
2. `mean_delta(m)` kèm CI và đường 0.
3. `auroc_S` theo `m` kèm CI.

### 6.6. Phân tích B: PMC (`04_pmc.py`)

Dùng cùng forward pass (hoặc forward lại; không thay đổi kết quả).

Với mỗi object (mention đầu tiên, **toàn bộ**, không chỉ cặp ghép):
- `sent_start` = chỉ số token ngay sau token cuối cùng có `piece` kết thúc bằng `.` ở vị trí `< t`; nếu không có, `sent_start = 0`.
- `n_prec = t − sent_start` (bỏ qua object nếu `n_prec < 1`).
- `PMC = min_{i ∈ [sent_start, t−1]} conf_i`, với `conf_i = max_v softmax(pred[i])[v]`.
- `argmin_dist = t − argmin_i conf_i`.

Báo cáo:
- Trung bình PMC cho hallucinated vs thật (toàn bộ), và so với giá trị paper (LLaVA-1.5: 0,29 vs 0,22; **không kỳ vọng khớp chính xác**, vì thiết lập khác).
- **Kiểm soát độ dài:** PMC giảm một cách cơ học khi có nhiều token phía trước hơn. Phải báo cáo thêm (a) theo bin `n_prec` (1–3, 4–6, 7–10, ≥ 11), và (b) trên **tập ghép cặp** (caliper thêm trên `n_prec`).
- Phân bố `argmin_dist` (khoảng cách từ token độ tin cậy thấp nhất tới object) cho hallucinated vs thật.

### 6.7. Phân tích C (tùy chọn): bỏ ảnh

Chỉ làm sau khi A và B xong. Lặp lại 6.4 với ảnh xám đồng nhất cùng kích thước (giữ nguyên mọi thứ khác) để có `S_blank`. Báo cáo `S − S_blank` theo `m` cho hai nhóm. Ghi rõ: ảnh trống là phân phối ngoài miền, nên chỉ là chỉ báo thô của language prior.

### 6.8. Phân tích khám phá (tùy chọn, đánh dấu *exploratory*)

Chia `Δ(m)` theo loại token tại `t − 1`: mạo từ (`a`, `an`, `the`), giới từ (`of`, `on`, `in`, `with`, `to`, `at`, `near`, `by`, `for`), liên từ/dấu câu, khác. Chỉ báo cáo mô tả, không kết luận.

---

## 7. Kiểm thử bắt buộc

### Không cần GPU (`tests/`, dùng `pytest`)

| Mã | Nội dung | Kỳ vọng |
|---|---|---|
| T1 | `pieces_to_text` + `find_mentions` + `char_to_token` trên chuỗi mẫu `['▁The','▁image','▁shows','▁a','▁man','▁and','▁two','▁dog','s','▁near','▁a','▁teddy','▁bear','<0x0A>','▁Mugs','.']` với synonyms giả | `man→tok 4`, `dogs→tok 7 (▁dog)`, `teddy bear→tok 11`, `mugs→cup, tok 14` |
| T2 | `select_objects` trên dữ liệu giả | Mọi cặp có `|Δrel_pos| ≤ caliper`; `t ≥ K`; không tái sử dụng object thật; kết quả giống nhau khi cùng seed |
| T3 | Thống kê trên dữ liệu giả có `Δ` biết trước | Bootstrap CI chứa giá trị thật; Holm đúng; AUROC đúng khi tách hoàn toàn |

### Cần GPU

| Mã | Nội dung | Ngưỡng |
|---|---|---|
| T4 | **Kiểm tra căn chỉnh vị trí:** với 20 ảnh, tỷ lệ `argmax(pred[i]) == gen_ids[i]` trên mọi `i` | ≥ 98% = đạt; 90–98% = xem lại dtype/KV cache; **< 90% gần như chắc chắn lệch vị trí (off-by-one), dừng và sửa** |
| T5 | Tại `m = 0`, `rank == 1` cho ≥ 99% object được chọn | Nếu không, có lỗi căn chỉnh |
| T6 | Với 5 ảnh, `logp` tại `m = 0` từ pipeline bằng `logp` lấy từ `generate(output_scores=True)` (sai số ≤ 0,05 do fp16) | Đạt |
| T7 | Chạy hai lần cùng seed cho cùng `summary.csv` (sai số số học ≤ 1e-3) | Đạt |

T4 là kiểm tra quan trọng nhất; nó phát hiện phần lớn lỗi lệch chỉ số.

---

## 8. Các cổng và lộ trình chạy

Mỗi cổng phải đạt **trước khi** sang cổng sau. Nếu không đạt, **dừng và báo cáo**, không nới ngưỡng.

| Cổng | Việc | Điều kiện qua | Quy mô (ước lượng) |
|---|---|---|---|
| **G0** | Unit test T1–T3 | Tất cả đạt | phút |
| **G1** | Smoke 3 ảnh: sinh caption, gán nhãn, in tay vài mention | Đọc tay xác nhận mapping token đúng; T4–T6 đạt trên 3–20 ảnh | nửa ngày |
| **G2** | Pilot `dev` 100 ảnh chạy hết pipeline | Ra đủ mọi file ở Mục 5; `funnel.json` hợp lý | vài chục phút |
| **G3** | Chạy `dev` 500 ảnh: **CHAIR sanity** | `CHAIR_I` cùng bậc với paper TruthPrInt cho LLaVA-1.5 greedy (≈ 6, CHAIR_S ≈ 19,6; **chỉ đối chiếu thô**) **và** đã kiểm tay 100–200 object, ghi tỷ lệ nhãn sai | 1–3 giờ chạy sinh caption + vài chục phút thống kê |
| **G4** | Kết quả `dev` đầy đủ (A + B) và báo cáo nháp | Mọi phân tích Mục 6.5–6.6 có số liệu | |
| **G5** | Chạy `confirm` (vị trí `[500, 2500)`) với **mọi lựa chọn đã cố định** | Báo cáo cuối dùng số liệu của `confirm`, và nêu số `dev` làm tham chiếu | vài giờ đến vài chục giờ |

**Ước lượng số cặp ghép:** với `CHAIR_I ≈ 6%` và vài object mỗi caption, `dev` 500 ảnh có thể cho khoảng một–hai trăm cặp; `confirm` 2.000 ảnh cho khoảng vài trăm đến nghìn cặp. **Hãy dùng `funnel.json` để biết con số thật.** Nếu `n_pairs < 200` ở `confirm`, báo cáo phải nêu rõ là thiếu công suất.

Chỉ chạy Phân tích C (6.7) và 6.8 sau G4, và chỉ khi còn thời gian.

---

## 9. Đầu ra bắt buộc

```
results/exp1/
├── lag_records.csv
├── summary.csv
├── pmc_records.csv
├── pmc_summary.csv
├── funnel.json
├── run_manifest.json
├── REPORT.md
└── figures/
    ├── s_vs_m.png
    ├── delta_vs_m.png
    ├── auroc_vs_m.png
    ├── pmc_by_group.png
    └── pmc_by_nprec.png
CHANGELOG.md
tests/
```

### Khung `REPORT.md` (điền số thật, không để trống)

1. **Tóm tắt (≤ 10 dòng):** thiết lập, số cặp, kết quả chính theo bảng diễn giải ở Mục 2 (nêu một trong bốn loại).
2. **Thiết lập:** phiên bản, tham số, seed, kích thước `dev`/`confirm`.
3. **Funnel:** bảng từ `funnel.json`.
4. **Kiểm tra chất lượng:** kết quả T4–T7; `CHAIR_S/CHAIR_I` so với paper; tỷ lệ nhiễu nhãn từ kiểm tay.
5. **Kết quả A:** bảng `summary.csv`, ba hình; độ ổn định ghép cặp; phân tích nhạy cảm S1–S3.
6. **Kết quả B:** PMC theo nhóm, theo `n_prec`, trên tập ghép cặp; nhận xét về chiều của PMC so với Bảng 7 của paper.
7. **Kết quả C và khám phá (nếu có):** ghi rõ *exploratory*.
8. **Diễn giải:** chỉ nói điều dữ liệu cho phép (Mục 10).
9. **Hạn chế:** CHAIR đơn giản hóa, nhiễu nhãn, một mô hình, một tập ảnh, số cặp, fp16, teacher forcing.
10. **Việc tiếp theo đề xuất.**

---

## 10. Quy tắc và điều kiện dừng

### Cấm
- Đổi `K`, caliper, định nghĩa `S`, hoặc loại outlier **sau khi thấy kết quả**.
- Loại object ngoài các bộ lọc đã liệt kê trong funnel.
- Dùng dữ liệu `confirm` để chọn tham số.
- Kết luận **nhân quả** ("token X gây ra hallucination"). Kết quả này chỉ là tương quan ở đầu ra. Câu hợp lệ: "object hallucinated có / không có điểm `S` khác object thật ở độ trễ m".
- Báo cáo kết quả chọn lọc (chỉ m "đẹp"). Phải báo cáo mọi `m = 0..K`.
- Âm thầm dùng phương án dự phòng (ví dụ synonyms thiếu); mọi dự phòng phải in cảnh báo và ghi vào `CHANGELOG.md`/`REPORT.md`.

### Dừng và báo cáo khi
- Không có GPU hoặc không tải được mô hình/dữ liệu: nêu chính xác thiếu gì.
- G1 (T4) thất bại và không sửa được trong phạm vi hợp lý.
- Nhãn CHAIR sai quá nhiều (tỷ lệ nhiễu kiểm tay cao bất thường); nêu số liệu và đề xuất.
- Số cặp ghép quá ít cho mọi phân tích có ý nghĩa.

---

## 11. Rủi ro đã biết trong code khởi tạo

Code ở `src/` do Claude viết, **chưa chạy trên mô hình thật**. Cần chú ý và sửa nếu cần:

1. `03_lag_curve.py` đang dùng bootstrap **độc lập** cho hai nhóm; đặc tả này yêu cầu phân tích **theo cặp** (Mục 6.5) và cột `pair_id`. Cần thêm.
2. `03_lag_curve.py` chưa ghi `conf`, `ent`, `logp_actual`, `t`, `G`, `rel_pos`, `pair_id`, chưa có `funnel.json`, kiểm định Wilcoxon/Holm, `dz`, độ ổn định ghép cặp, S1–S3.
3. Chưa có `04_pmc.py` và các test T1–T7.
4. `01_generate.py` chưa có `--offset` cho tách `dev`/`confirm`.
5. `02_label_chair.py` chưa kiểm tra điều kiện word-start (Mục 6.2).
6. `V_obj` dùng `tok.encode(w)`: cần xác nhận tiền tố `▁` (Mục 6.4).
7. `load_synonyms` dùng `setdefault`: từ xuất hiện ở nhiều dòng sẽ giữ nhóm đầu tiên; kiểm tra có xung đột đáng kể không.
8. CHAIR đơn giản hóa: số ít hóa thô; chưa xử lý các trường hợp đặc biệt của CHAIR gốc. Nếu có thể, so sánh với `chair.py` gốc trên ≥ 50 caption và báo cáo mức khớp.
9. Cách `processor(images=..., text=...)` xử lý `<image>` thay đổi giữa các phiên bản `transformers`; kiểm tra bằng T4.
10. So sánh `float16` giữa `generate` (có KV cache) và forward một lần có thể làm lệch nhẹ các trường hợp gần hòa; T4–T6 sẽ phát hiện.

---

## 12. Phụ lục: checklist hoàn thành

- [ ] G0: T1–T3 đạt
- [ ] G1: smoke 3 ảnh; T4–T6 đạt; đã kiểm tay mapping token
- [ ] G2: pilot 100 ảnh chạy hết, đủ file
- [ ] G3: `CHAIR_S/I` trên 500 ảnh `dev`; đã kiểm tay 100–200 object, ghi tỷ lệ nhiễu
- [ ] G4: kết quả `dev` đầy đủ (A, B), báo cáo nháp
- [ ] G5: chạy `confirm`, báo cáo cuối
- [ ] `run_manifest.json`, `CHANGELOG.md`, `funnel.json` đầy đủ
- [ ] `REPORT.md` đủ 10 mục, không có số bịa, không có khẳng định nhân quả
- [ ] Đã nêu rõ các giới hạn và điểm chưa kiểm chứng

*Kết thúc đặc tả.*
