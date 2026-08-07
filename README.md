# DSC 2026 — Task 1: Legal Information Retrieval

Hệ thống truy xuất văn bản pháp luật tiếng Việt. Cho một câu hỏi, trả về **5 văn bản**
có khả năng chứa câu trả lời nhất.

**Public leaderboard: Recall@5 = 0.8637**

| Thành phần | Recall@5 | Ghi chú |
|---|---|---|
| BM25 thuần | 0.8112 | |
| Dense thuần | 0.8307 | |
| **Hybrid (đang dùng)** | **0.8633** | khớp gần như tuyệt đối với public 0.8637 |
| + cross-encoder | +0.03 | đã bỏ: nặng, lợi ích bị trần ứng viên triệt tiêu |

*Đo trên 1.000 câu validation tách riêng từ `train.json`, không dùng để huấn luyện.*

---

## Pipeline

```
8.532 văn bản pháp luật (353M ký tự, median 4.813 từ/văn bản)
        │
        ▼  chunker 3 tầng theo cấu trúc pháp lý
487.194 chunk (median 165 từ, max 407)
        │
        ├──────────────────────┬─────────────────────────┐
        ▼                      ▼                         │
   BM25 (sparse)         halong_embedding (dense)        │
   inverted index        487k × 768 fp16                 │
   trên đĩa, 299 MB      trên đĩa, 714 MB                │
        │                      │                         │
        │  top-1.000 chunk     │  cosine trên pool       │
        └──────────┬───────────┘                         │
                   ▼                                     │
       hoà điểm linear: 0.7 × dense + 0.3 × BM25          │
                   ▼                                     │
       gộp chunk → document: trung bình 3 chunk cao nhất  │
                   ▼                                     │
              top-5 văn bản  ◄──────────────────────────┘
```

### 1. Chunking 3 tầng

Văn bản gốc quá dài để nhét vào encoder — **99,6% vượt 512 token**, cá biệt có văn bản
1,24 triệu từ. Nhưng chunk không chỉ để lách giới hạn token:

```
Tỷ lệ từ khoá câu hỏi xuất hiện trong văn bản ĐÚNG      : 0.944
Tỷ lệ từ khoá câu hỏi xuất hiện trong văn bản NGẪU NHIÊN : 0.683
```

Một văn bản lấy ngẫu nhiên đã chứa sẵn 68% từ khoá chỉ vì nó dài. Chunk là để **khôi phục
tín hiệu phân biệt**.

| Tầng | Điều kiện | Số chunk |
|---|---|---|
| 1 | cắt theo `Điều N` (85,9% văn bản có) | 122.520 |
| 2 | Điều > 350 từ → cắt theo `Khoản`; vẫn dài → cửa sổ cố định | 333.416 |
| 3 | văn bản không có `Điều` (công văn, TCVN) → cửa sổ 256 từ, overlap 25% | 31.238 |
| 0 | `passage` rỗng → vớt tiêu đề từ slug trong `link` | 20 |

Tầng 0 không phải chi tiết vụn: có **20 văn bản `passage` rỗng, 6 trong đó là đáp án gold**.
Thiếu tầng này là mất trắng 6 câu.

Mỗi chunk được prepend header `tên văn bản > Điều > Khoản`, và tiêu đề Điều được nhân bản
vào mọi mảnh con — nếu không, mảnh thứ 2 trở đi mất hết ngữ cảnh.

### 2. BM25 trên đĩa

Máy phát triển chỉ còn ~0,5 GB RAM trống, không đủ cho ma trận sparse 487k × 70k trong bộ
nhớ. Index được ghi ra đĩa dưới dạng `np.memmap`, build 2 lượt (lượt 1 đếm df để biết
offset, lượt 2 ghi postings vào đúng ô). Lúc query chỉ đọc postings của vài từ trong câu
hỏi → RAM gần như bằng 0.

Tham số: `k1=2.5, b=0.9` (xác minh trên public test).

### 3. Dense — vai trò reranker, không phải retriever

Model: [`hiieu/halong_embedding`](https://huggingface.co/hiieu/halong_embedding)
(xlm-roberta base, 278M, 768 chiều). Ba thuộc tính đọc từ config chứ không đoán:
**mean pooling**, **L2 normalize sẵn**, và **không cần prefix** `query:`/`passage:` —
dễ sai vì kiến trúc giống multilingual-e5 vốn *bắt buộc* có prefix.

Dense không search toàn cục vì phép đo trần nói rằng không cần:

| Pool (chunk) | Gold nằm trong pool |
|---|---|
| 200 | 0.9767 |
| 1.000 | 0.9900 |
| 2.000 | 0.9967 |

BM25 tìm ra rồi, chỉ **xếp sai chỗ**. Nên dense chỉ đọc ~1.000 vector từ memmap
(~3 MB/query) để xếp lại — không cần faiss, không cần nạp 714 MB vào RAM.

### 4. Hoà điểm

BM25 (16–40, không chặn trên) và cosine (0.07–0.58) không cộng thẳng được. Chuẩn hoá
min-max trong phạm vi pool rồi trộn:

```
điểm = 0.7 × dense_norm + 0.3 × bm25_norm
```

RRF (chỉ dùng thứ hạng) thua linear 0.8375 vs 0.8528 — vì nó vứt bỏ thông tin *khoảng
cách*, mà "thắng áp đảo" khác hẳn "thắng sát nút".

### 5. Gộp chunk → document

Nhãn ở mức document nhưng chấm điểm ở mức chunk, nên phải gộp:

| Cách gộp | Recall@5 |
|---|---|
| **top-3 mean** | **0.8250** |
| max | 0.7950 |
| avg toàn bộ chunk | 0.4175 |
| top3 / log(số chunk) | 0.1706 |

`avg` hỏng vì văn bản `68843` có 6.514 chunk — một câu hỏi chỉ khớp 2–3 chunk, phần còn
lại điểm 0 kéo trung bình về sát 0.

---

## Cài đặt

```bash
pip install numpy scipy torch transformers
# tuỳ chọn, chỉ cần khi fine-tune:
pip install sentence-transformers datasets
```

Cần GPU cho bước encode (RTX 3050 6GB là đủ). Các bước khác chạy CPU được.

Giải nén dữ liệu của BTC vào thư mục gốc — repo này **không** chứa dữ liệu cuộc thi:

```
LegalIR - Public Test/
  train.json                    7.000 câu hỏi + đáp án
  public-official.json          1.000 câu hỏi, answer = null
  selected-contexts/            8.532 file context_*.json
```

---

## Chạy

Bốn bước, mỗi bước tạo ra thứ bước sau cần. Tất cả sản phẩm trung gian đều bị `.gitignore`
(tổng ~1,7 GB) nhưng tái tạo được hoàn toàn.

```bash
# 1. Chunk corpus                              1m31s  → 592 MB
cd Chunking-LegalIR
python chunker.py --contexts "../LegalIR - Public Test/selected-contexts" \
                  --out chunks.jsonl

# 2. Index BM25                                 202s  → 299 MB
cd ../Retrieval-LegalIR
python bm25.py build --chunks ../Chunking-LegalIR/chunks.jsonl --index index

# 3. Trích vector                             36 phút  → 714 MB
python encode.py --chunks ../Chunking-LegalIR/chunks.jsonl --out emb

# 4. Sinh bài nộp                                74s  → submission.zip
python predict.py --index index --emb emb \
    --questions "../LegalIR - Public Test/public-official.json" \
    --train "../LegalIR - Public Test/train.json" \
    --k1 2.5 --b 0.9 --alpha 0.7 --out submission
```

Kiểm tra bài nộp trước khi upload:

```bash
python ../Validator-Task-LegalIR/validate_submission.py submission.zip \
    --ref "../LegalIR - Public Test/public-official.json" \
    --corpus "../LegalIR - Public Test/selected-contexts"
```

### Tìm kiếm thủ công

```bash
python bm25.py query --index index "mức lương cơ sở là bao nhiêu" --topk 10
python hybrid.py --query "lệ phí làm căn cước công dân"
```

### Giao diện web

```bash
python serve.py --index index --contexts "../LegalIR - Public Test/selected-contexts"
```

Mở http://127.0.0.1:8000 — tìm kiếm, bấm vào kết quả để đọc toàn văn render theo
Chương/Điều/Khoản, tự nhảy tới Điều khớp, tô vàng từ khoá. Không cần cài thêm gì
(chỉ dùng `http.server` của stdlib).

### Đo lại

```bash
python eval_bm25.py --index index --train "../LegalIR - Public Test/train.json" -n 300
python hybrid.py --eval "../LegalIR - Public Test/train.json" -n 300
python sweep_bm25.py --index index --train "../LegalIR - Public Test/train.json" -n 200
```

---

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `Chunking-LegalIR/` | `chunker.py` — chunker 3 tầng |
| `Retrieval-LegalIR/` | `bm25.py` · `encode.py` · `hybrid.py` · `rerank.py` · `predict.py` · `serve.py` + script đo |
| `Finetune-LegalIR/` | `mine_data.py` · `train.py` — fine-tune bi-encoder (đang làm) |
| `Validator-Task-LegalIR/` | Kiểm tra bài nộp trước khi upload, 14 mã lỗi |
| `Mock-Eval-LegalIR/` | Chạy `scoring.py` gốc của BTC ở local |
| `EDA-LegalIR/` | Phân tích corpus |
| `RESULTS.md` | Toàn bộ số liệu đã đo, kèm cỡ mẫu và điều kiện |

### Validator

Chặn bài nộp lỗi trước khi tốn lượt upload. Gom hết lỗi rồi in một lần thay vì dừng ở lỗi
đầu. Bắt được cả `question_id` trùng lặp — thứ `json.load` mặc định nuốt mất.

Lỗi nguy hiểm nhất là **E010**: `context_*.json` lưu `id` kiểu số nguyên nhưng đáp án bắt
buộc là chuỗi. Sai chỗ này thì `scoring.py` cho **0 điểm mà không báo lỗi gì**.

### Mock eval

Chạy **nguyên văn** `scoring.py` của BTC ở local: đọc file rồi `exec` trong namespace riêng
với `__name__ = 'btc_scoring'`, ghi đè 3 biến đường dẫn `/app/...`. Không sửa một ký tự nào
trong file của BTC, nên nó crash chỗ nào thì local cũng crash y chỗ đó.

⚠️ File reference của BTC phải ở dạng **phẳng** `{qid: [doc_id]}`, không phải cấu trúc của
`train.json`. Đưa nhầm thì **toàn bộ thí sinh 0 điểm mà không có lỗi nào báo ra**.

---

## Phát hiện về dữ liệu

| Phát hiện | Con số | Hệ quả |
|---|---|---|
| Văn bản khổng lồ | median 4.813 từ, max 1.242.409 | bắt buộc chunk |
| Tín hiệu lexical loãng | 0.944 vs 0.683 (đúng vs ngẫu nhiên) | BM25 trên nguyên văn bản sẽ hỏng |
| Văn bản rỗng | 20 văn bản, **6 là gold** | cần fallback tầng 0 |
| Thiếu `name` | 1.125/8.532 văn bản | phải dùng `.get()` |
| Lệch popularity | 36,4% corpus từng là đáp án; `245154` xuất hiện 109 lần | 5.427 văn bản chưa từng là đáp án chính là nơi chứa đáp án public/private — đừng prune |
| Viết tắt | `cccd` hiếm → idf cao → lái sang văn bản sai | cần từ điển viết tắt phía query |

Ví dụ ca viết tắt:

```
'lệ phí làm cccd là bao nhiêu'   →  #1 doc 20457 (sai)
'lệ phí làm căn cước công dân'   →  #1 doc 165290 (đúng)
```

---

## Bài học đo đạc

Ở cỡ mẫu 200–500 câu, **chênh lệch dưới ~0.01 không đọc được**. Đã ba lần chọn nhầm tham số
vì tin vào đỉnh của một lần quét lưới:

- `α=0.8` — đỉnh 0.9061 trên seed 777, tụt xuống 0.8445 trên seed 2024
- `k1/b` — phải tới 2.000 câu + bootstrap mới thấy khoảng tin cậy chứa 0, hai cấu hình
  không phân biệt được
- cross-encoder — `beta=1.0` cho 0 điểm cải thiện, nhưng `beta=0.5` cho +0.03

Quy trình đã chốt: quét lưới trên một mẫu → **kiểm chứng lại trên mẫu khác seed** → chỉ tin
**delta**, không tin giá trị tuyệt đối.

Và bằng chứng từ public test luôn thắng mọi phép đo trên train.

---

## Việc tiếp theo

1. **Fine-tune `halong_embedding`** trên 6.000 cặp đã đào (`Finetune-LegalIR/`) — mọi thành
   phần hiện tại đều zero-shot, chưa cái nào nhìn thấy một dòng văn bản pháp luật của cuộc
   thi. Đây là lực đòn bẩy lớn nhất còn lại.
2. **Từ điển viết tắt** phía query — rẻ, đo được trong 10 phút.
3. **Word segmentation + bigram cho BM25** — "mức lương cơ sở" đang bị xé thành 4 âm tiết rời.
4. **Bật lại cross-encoder** với `ndocs=50` (trần 0.9463 thay vì 0.9177 của `ndocs=20`).
