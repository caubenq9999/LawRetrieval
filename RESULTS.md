# DSC 2026 — Task 1: Legal Information Retrieval — Kết quả

Ghi lại toàn bộ số liệu đã đo, kèm cỡ mẫu và điều kiện đo. Mọi con số trong tài liệu này
đều từ lần chạy thật, không phải ước lượng — trừ chỗ ghi rõ là ước lượng.

Cập nhật: 2026-08-07

---

## 1. Kết quả chính

| Chỉ số | Giá trị | Đo trên |
|---|---|---|
| **Recall@5** | **0.8233** | 300 câu train, chấm bằng `scoring.py` của BTC |
| Precision@5 | 0.1740 | như trên |
| Bài nộp | `Retrieval-LegalIR/submission.zip` | 1.000 câu public test, validator pass |

Recall là độ đo chính; Precision chỉ dùng phá hoà. Precision thấp là **cố ý**: trả đủ 5 ID
cho câu thường chỉ có 1 đáp án thì trần Precision là 0.20. Trả 1 ID thì Precision ~1.0
nhưng Recall tụt còn ~0.51.

> `0.8233` đo trên câu **lấy từ train**, không phải điểm public test. Đáp án public nằm ở
> phía BTC (`answer: null` trong đề bài) nên không thể chấm ở local.

---

## 2. Pipeline

```
context_*.json  ──chunk 3 tầng──►  487.194 chunk
                                        │
                                   BM25 (k1=3.0, b=0.75)
                                        │
                                   điểm từng chunk
                                        │
                                   gộp top-3 mean ──►  điểm document
                                        │
                                   top-5 document
```

### 2.1 Chunking — 3 tầng

| Tầng | Điều kiện | Số chunk |
|---|---|---|
| 1 | cắt theo `Điều N` | 122.520 |
| 2 | Điều > 350 từ → cắt theo `Khoản`; vẫn dài → cửa sổ cố định | 333.416 |
| 3 | văn bản không có `Điều` (công văn, TCVN) → cửa sổ 256 từ, overlap 25% | 31.238 |
| 0 | `passage` rỗng → vớt tiêu đề từ slug trong `link` | 20 |

```
8.532 văn bản → 487.194 chunk    1m31s, 592 MB
độ dài: median 165 từ · p90 266 · p99 336 · max 407   (từ max gốc 1.242.409)
chunk/văn bản: median 33 · p90 126 · max 6.514
```

Đối chiếu toàn vẹn: **0/8.532 văn bản bị mất**, **0/3.105 đáp án gold thiếu chunk**.

Mỗi chunk được prepend header `tên văn bản > Điều > Khoản`. Tiêu đề Điều được nhân bản
vào mọi mảnh con, nếu không mảnh thứ 2 trở đi mất hết ngữ cảnh.

### 2.2 BM25

Inverted index trên đĩa qua `np.memmap` (RAM máy chỉ còn ~0.5 GB nên không dùng sparse
matrix trong bộ nhớ được). Build 2 lượt: lượt 1 đếm df để biết offset, lượt 2 ghi postings.

```
từ điển 69.920 · 48.339.443 postings · index 299 MB · build 202s · query 85ms
```

### 2.3 Gộp chunk → document

Đo trên 200 câu train, `k1=3.0 b=0.75`:

| Cách gộp | Recall@5 |
|---|---|
| **top3** (trung bình 3 chunk cao nhất) | **0.8250** |
| max | 0.7950 |
| avg toàn bộ chunk | 0.4175 |
| top3_log (chia log số chunk) | 0.1706 |

`avg` toàn bộ hỏng vì văn bản `68843` có 6.514 chunk — một câu hỏi chỉ khớp 2–3 chunk,
6.511 chunk còn lại điểm 0 kéo trung bình về sát 0. Càng dài càng bị dìm.

---

## 3. Tinh chỉnh BM25

Quét lưới `k1`/`b` trên 200 câu (đổi tham số không cần build lại index — postings lưu tf thô):

```
k1 \ b     0.4      0.6     0.75      0.9
0.6     0.7788   0.7788   0.7963   0.8013
0.9     0.7887   0.8187   0.8263   0.8263
1.2     0.8029   0.8237   0.8363   0.8363
1.5     0.8079   0.8279   0.8379   0.8454
2.0     0.8087   0.8379   0.8479   0.8479
3.0        —        —    0.8579*     —
```

Kiểm chứng trên 500 câu **seed khác** để loại overfit vào lưới quét:

| Cấu hình | Recall@5 | Hit@5 | MRR@20 |
|---|---|---|---|
| mặc định k1=1.5 b=0.75 | 0.8220 | 0.8420 | 0.6456 |
| tune k1=3.0 b=0.75 | 0.8290 | 0.8480 | 0.6585 |

Mức tăng tụt từ **+0.020 xuống +0.007** — phần lớn ở lưới quét là nhiễu mẫu. Tune BM25
coi như đã cạn.

**Hai giả thuyết ban đầu bị dữ liệu bác bỏ:**

- `top3_log`: phạt văn bản nhiều chunk → 0.1706, tệ hơn `max` 5 lần. BM25 đã chuẩn hoá độ
  dài ở mức chunk rồi, phạt thêm ở mức document chỉ làm hỏng.
- `k1` thấp: tưởng làm từ lặp bão hoà nhanh sẽ triệt tiêu lợi thế "nhồi từ khoá" của văn
  bản dài. Thực tế ngược — `k1=0.6` → 0.796, `k1=3.0` → 0.858. Tần suất lặp là tín hiệu
  chủ đề, không phải nhiễu.

---

## 4. Phân bố thứ hạng và trần recall

300 câu train, gộp `top3`:

```
hạng 1      : 51.0%        hạng 6-10   :  5.7%
hạng 2-3    : 23.3%        hạng 11-20  :  6.0%
hạng 4-5    :  9.3%        ngoài top20 :  4.7%
```

Trần recall của pool BM25 (300 câu):

| Pool (chunk) | Gold nằm trong pool | Số văn bản riêng |
|---|---|---|
| 200 | 0.9767 | 70 |
| 1.000 | 0.9900 | 304 |
| 2.000 | 0.9967 | 563 |
| 10.000 | 0.9967 | 2.142 |

**Đây là con số quyết định kiến trúc.** Ở mức document 4.7% gold rớt ngoài top-20, nhưng ở
mức chunk pool 1.000 đã chứa 99.0% gold. Cùng một văn bản có thể đứng hạng 50 ở mức doc
nhưng vẫn có chunk lọt top-1.000. Tức **BM25 tìm ra rồi, chỉ xếp sai chỗ** — vấn đề là
ranking chứ không phải recall.

Dư địa còn lại: **+0.168** (0.829 → 0.9967), lấy được bằng rerank, không cần tìm thêm.

---

## 5. Dense — đang chạy

Model `hiieu/halong_embedding`: xlm-roberta base, 278M tham số, 768 chiều.

Ba thuộc tính đọc từ config, không đoán:

- **mean pooling** (`pooling_mode_mean_tokens: true`, cls `false`)
- **L2 normalize** sẵn → dot product = cosine
- **không cần prefix** `query:`/`passage:` (`prompts: {}`; README encode query và doc y hệt).
  Chỗ này dễ sai vì kiến trúc giống multilingual-e5 vốn *bắt buộc* có prefix.
- train bằng `MatryoshkaLoss` → cắt 768→256 rồi normalize lại vẫn dùng được, không cần
  encode lại (chỉ là slice cột).

Benchmark trên 8.192 chunk (RTX 3050 6GB):

```
226 chunk/s  →  487.194 chunk ≈ 36 phút
độ dài token: median 209 · p90 389 · p99 507 · max 512
bị cắt ở 512 token: 1.0%
vector: 487.194 × 768 fp16 = 714 MB trên đĩa
```

Chunking 350 từ khớp đẹp với giới hạn 512 token của model — chỉ 1% mất mát.

### Kiến trúc hybrid

Dense đóng vai **reranker**, không phải retriever — hệ quả trực tiếp của trần pool ở mục 4.
BM25 lấy pool 1.000 chunk → đọc vector từ memmap (~3 MB/query) → cosine → hoà điểm → gộp
`top3` về document.

Nhờ vậy không cần faiss và không phải nạp 714 MB vector vào RAM.

Sẽ so 6 cấu hình: `bm25` thuần, `dense` thuần, `rrf`, `linear` ở alpha 0.3/0.5/0.7.

**Chưa có kết quả.** Trần +0.168 là dư địa, không phải mức tăng sẽ đạt.

---

## 6. Phát hiện về dữ liệu

| Phát hiện | Con số | Hệ quả |
|---|---|---|
| Văn bản khổng lồ | median 4.813 từ, max 1.242.409, **99.6% vượt 512 token** | bắt buộc chunk |
| Tín hiệu lexical loãng | từ khoá câu hỏi có trong văn bản đúng 0.944, trong văn bản **ngẫu nhiên** 0.683 | BM25 trên nguyên văn bản sẽ hỏng; chunk là để khôi phục tín hiệu phân biệt |
| Văn bản rỗng | 20 văn bản `passage` rỗng, **6 là gold trong train** | cần fallback tầng 0 |
| Thiếu `name` | 1.125/8.532 văn bản | `d["name"]` sẽ `KeyError`, phải `.get()` |
| Lệch popularity | chỉ 36.4% corpus từng là đáp án; `245154` là đáp án 109 lần | 5.427 văn bản chưa từng xuất hiện trong train là nơi chứa đáp án public/private — đừng prune |
| Cấu trúc pháp lý | 85.9% có `Điều N`, 95.8% có `Khoản` | chunk theo cấu trúc khả thi |
| Viết tắt | `cccd` có trong từ điển nhưng hiếm → idf cao → kéo kết quả sang văn bản sai | cần từ điển viết tắt ở phía query |

Ví dụ ca viết tắt:

```
'lệ phí làm cccd là bao nhiêu'   →  #1 doc 20457 (sai), #2 doc 20457, #4 doc 20457
'lệ phí làm căn cước công dân'   →  #1 doc 165290 (GOLD)
```

---

## 7. Hạ tầng

### `Validator-Task-LegalIR/`
Validator local cho bài nộp: 14 mã lỗi + 5 cảnh báo, gom hết lỗi rồi in một lần.
Bắt được cả `question_id` trùng lặp — thứ `json.load` mặc định nuốt mất.

### `Mock-Eval-LegalIR/`
Chạy **nguyên văn** `scoring.py` của BTC ở local: đọc file rồi `exec` trong namespace riêng
với `__name__ = 'btc_scoring'`, ghi đè 3 biến `reference_dir`/`prediction_dir`/`score_dir`.
Không sửa một ký tự nào trong file của BTC.

Phát hiện về `scoring.py`:

- **crash** với `answer: null`, thiếu key `answer`, sai question_id (`TypeError`/`KeyError`
  trần, không có thông điệp)
- **0 điểm âm thầm** khi `document_id` là `int` — `context_*.json` lưu `id` kiểu số nguyên
  nhưng đáp án bắt buộc là chuỗi
- file reference phải **phẳng** `{qid: [id]}`, không phải cấu trúc của `train.json`. Đưa
  nhầm thì **toàn bộ thí sinh 0 điểm mà không báo lỗi gì**

Đối chứng 7 bài nộp mẫu: `perfect` 1.0/1.0 · `padded` recall 1.0 precision 0.22 ·
`int_ids` **0.0/0.0**.

### `EDA-LegalIR/`
`eda_corpus.py`, `eda_chunking.py` — số liệu ở mục 6.

### `Retrieval-LegalIR/`

| File | Việc |
|---|---|
| `bm25.py` | build/query inverted index trên đĩa |
| `eval_bm25.py` | đo Recall/Precision/MRR theo công thức BTC |
| `sweep_bm25.py` | quét lưới k1/b |
| `predict.py` | sinh `submission.json` + `.zip` |
| `encode.py` | trích vector halong_embedding ra memmap |
| `hybrid.py` | BM25 + dense, hoà bằng RRF hoặc linear |
| `serve.py` | web UI local, không cần cài gì thêm |

---

## 8. Việc tiếp theo

1. **Chạy bảng so hybrid** khi encode xong — 6 cấu hình.
2. **Từ điển viết tắt** (CCCD, BHXH, GTGT, TNCN, UBND, ATGT…) mở rộng ở phía query. Rẻ hơn
   dense rất nhiều, đo được ngay.
3. **Cross-encoder tầng 3** nếu bi-encoder ăn được: BM25 (1.000) → bi-encoder (50) →
   cross-encoder (5). Bi-encoder nén cả chunk thành 1 vector trước khi thấy câu hỏi nên
   không "chú ý" được vào đúng đoạn; cross-encoder thì có.
4. Khoảng trống ngữ nghĩa thật: câu hỏi `"... là bao nhiêu"` ↔ đáp án `"30.000 đồng/thẻ"`
   không chung một token nào. Không tham số BM25 nào vá được.
