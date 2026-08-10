# DSC 2026 — Task 1: Legal Information Retrieval

Hệ thống truy xuất văn bản pháp luật tiếng Việt. Cho một câu hỏi, trả về **5 văn bản**
có khả năng chứa câu trả lời nhất.

**Public leaderboard: Recall@5 = 0.8637**

---

## Chạy

```bash
pip install -r requirements.txt

python run_all.py --data "LegalIR - Public Test"
```

Một lệnh, ra thẳng `submission.zip` đã được kiểm tra hợp lệ.

Pipeline gồm 5 bước, **bước nào đã có sản phẩm thì tự bỏ qua** — nên chạy lại an toàn,
đứt giữa chừng không phải làm lại từ đầu:

```
1. chunk corpus          1m31s   →  chunks.jsonl     592 MB
2. index BM25             202s   →  index/           299 MB
3. trích vector        36 phút   →  emb/             714 MB   (cần GPU)
4. sinh bài nộp            74s   →  submission.zip    21 KB
5. kiểm tra bài nộp
```

Không có GPU thì bước 3 rất lâu. Bỏ qua bằng `--skip-encode` để chạy BM25 thuần
(nhanh, nhưng mất khoảng 0.05 Recall).

### Dữ liệu

Repo **không** chứa dữ liệu cuộc thi. Giải nén vào thư mục gốc:

```
LegalIR - Public Test/
  selected-contexts/            8.532 file context_*.json
  public-official.json          1.000 câu hỏi cần trả lời
  train.json                    7.000 câu hỏi + đáp án (tuỳ chọn)
```

### Tuỳ chọn

```bash
python run_all.py --data "..." --alpha 0.7 --k1 2.5 --b 0.9   # tham số đang dùng
python run_all.py --data "..." --skip-encode                  # chỉ BM25
python run_all.py --data "..." --force                        # làm lại từ đầu
```

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

| Thành phần | Recall@5 |
|---|---|
| BM25 thuần | 0.8112 |
| Dense thuần | 0.8307 |
| **Hybrid (đang dùng)** | **0.8633** |

*Đo trên 1.000 câu validation tách riêng từ `train.json`. Con số 0.8633 khớp gần như
tuyệt đối với public leaderboard 0.8637 — tập validation này là proxy đáng tin.*

### 1. Chunking 3 tầng

Văn bản gốc quá dài để nhét vào encoder — **99,6% vượt 512 token**, cá biệt có văn bản
1,24 triệu từ. Nhưng chunk không chỉ để lách giới hạn token:

```
Tỷ lệ từ khoá câu hỏi xuất hiện trong văn bản ĐÚNG       : 0.944
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

Kết quả: **0/8.532 văn bản bị mất, 0/3.105 đáp án gold thiếu chunk.**

### 2. BM25 trên đĩa

Máy phát triển chỉ còn ~0,5 GB RAM trống, không đủ cho ma trận sparse 487k × 70k trong bộ
nhớ. Index được ghi ra đĩa dưới dạng `np.memmap`, build 2 lượt (lượt 1 đếm df để biết
offset, lượt 2 ghi postings vào đúng ô). Lúc query chỉ đọc postings của vài từ trong câu
hỏi → RAM gần như bằng 0.

```
từ điển 69.920 · 48,3M postings · index 299 MB · query 85 ms
```

### 3. Dense — vai trò reranker, không phải retriever

Model: [`hiieu/halong_embedding`](https://huggingface.co/hiieu/halong_embedding)
(xlm-roberta base, 278M, 768 chiều). Ba thuộc tính đọc từ config chứ không đoán:
**mean pooling**, **L2 normalize sẵn**, và **không cần prefix** `query:`/`passage:` —
chỗ này dễ sai vì kiến trúc giống multilingual-e5 vốn *bắt buộc* có prefix.

Dense không search toàn cục, vì phép đo trần nói rằng không cần:

| Pool (chunk) | Gold nằm trong pool |
|---|---|
| 200 | 0.9767 |
| 1.000 | 0.9900 |
| 2.000 | 0.9967 |

BM25 tìm ra rồi, chỉ **xếp sai chỗ**. Nên dense chỉ đọc ~1.000 vector từ memmap
(~3 MB/query) để xếp lại — không cần faiss, không cần nạp 714 MB vào RAM.

Chunk 350 từ khớp đẹp với giới hạn 512 token của model: token median 209, **chỉ 1,0% bị cắt**.

### 4. Hoà điểm

BM25 (16–40, không chặn trên) và cosine (0.07–0.58) không cộng thẳng được. Chuẩn hoá
min-max trong phạm vi pool rồi trộn:

```
điểm = 0.7 × dense_norm + 0.3 × bm25_norm
```

RRF (chỉ dùng thứ hạng) thua linear **0.8375 vs 0.8528** — vì nó vứt bỏ thông tin *khoảng
cách*, mà "thắng áp đảo" khác hẳn "thắng sát nút".

### 5. Gộp chunk → document

Nhãn ở mức document nhưng chấm điểm ở mức chunk:

| Cách gộp | Recall@5 |
|---|---|
| **top-3 mean** | **0.8250** |
| max | 0.7950 |
| avg toàn bộ chunk | 0.4175 |
| top3 / log(số chunk) | 0.1706 |

`avg` hỏng vì văn bản `68843` có 6.514 chunk — một câu hỏi chỉ khớp 2–3 chunk, phần còn
lại điểm 0 kéo trung bình về sát 0.

---

## Cấu trúc

```
run_all.py                          chạy toàn bộ pipeline
requirements.txt
Chunking-LegalIR/
  chunker.py                        chunker 3 tầng
Retrieval-LegalIR/
  bm25.py                           build/query inverted index trên đĩa
  encode.py                         trích vector halong_embedding → memmap
  hybrid.py                         hoà điểm BM25 + dense
  predict.py                        sinh submission.json + .zip
  rerank.py                         cross-encoder (tuỳ chọn, xem bên dưới)
  serve.py                          giao diện web tra cứu
  eval_bm25.py, sweep_bm25.py       script đo
Validator-Task-LegalIR/
  validate_submission.py            kiểm tra bài nộp, 14 mã lỗi
RESULTS.md                          toàn bộ số liệu đã đo, kèm cỡ mẫu
```

### Dùng lẻ từng phần

```bash
cd Retrieval-LegalIR

# tìm kiếm thủ công
python bm25.py query --index index "mức lương cơ sở là bao nhiêu" --topk 10
python hybrid.py --query "lệ phí làm căn cước công dân"

# đo lại
python hybrid.py --eval "../LegalIR - Public Test/train.json" -n 300
python sweep_bm25.py --index index --train "../LegalIR - Public Test/train.json" -n 200

# giao diện web
python serve.py --index index --contexts "../LegalIR - Public Test/selected-contexts"
```

`serve.py` mở http://127.0.0.1:8000 — tìm kiếm, bấm vào kết quả để đọc toàn văn render
theo Chương/Điều/Khoản, tự nhảy tới Điều khớp, tô vàng từ khoá. Không cần cài thêm gì
(chỉ dùng `http.server` của stdlib).

### Validator

Chặn bài nộp lỗi trước khi tốn lượt upload. Gom hết lỗi rồi in một lần thay vì dừng ở lỗi
đầu. Bắt được cả `question_id` trùng lặp — thứ `json.load` mặc định nuốt mất.

Lỗi nguy hiểm nhất là **E010**: `context_*.json` lưu `id` kiểu số nguyên nhưng đáp án bắt
buộc là chuỗi. Sai chỗ này thì chương trình chấm cho **0 điểm mà không báo lỗi gì**.

`run_all.py` đã tự gọi validator ở bước 5.

---

## Phát hiện về dữ liệu

| Phát hiện | Con số | Hệ quả |
|---|---|---|
| Văn bản khổng lồ | median 4.813 từ, max 1.242.409 | bắt buộc chunk |
| Tín hiệu lexical loãng | 0.944 vs 0.683 (đúng vs ngẫu nhiên) | BM25 trên nguyên văn bản sẽ hỏng |
| Văn bản rỗng | 20 văn bản, **6 là gold** | cần fallback tầng 0 |
| Thiếu `name` | 1.125/8.532 văn bản | phải dùng `.get()`, không `d["name"]` |
| Lệch popularity | 36,4% corpus từng là đáp án; `245154` xuất hiện 109 lần | 5.427 văn bản chưa từng là đáp án chính là nơi chứa đáp án public/private — đừng prune |
| Viết tắt | 44/1000 câu có viết tắt, Recall 0.9091 vs 0.8998 câu thường | không phải vấn đề — đã đo |

Văn bản pháp luật tự định nghĩa viết tắt ngay Điều 1 rồi dùng cả hai dạng, nên chunk chứa
cả hai. Mở rộng viết tắt hoặc đồng nghĩa **không đáng làm**: trần của nó chỉ là +0.0040.

---

## Cross-encoder — đã thử, tạm bỏ

`rerank.py` cài sẵn tầng 3 dùng cross-encoder. Kết quả đo:

| Cách dùng | Delta |
|---|---|
| `beta=1.0` (thay thế hẳn xếp hạng hybrid) | ~0.000 |
| `beta=0.4–0.6` (trộn với hybrid) | **+0.030** |

Trộn thì có ăn, thay thế thì không — hybrid đã là xếp hạng tốt, cross-encoder off-the-shelf
chưa đủ mạnh để cầm lái.

Tạm bỏ vì `ndocs=20` áp trần cứng **0.9177** ở mức văn bản, nên +0.030 bị triệt tiêu bởi
−0.020 do cắt mất ứng viên. Muốn dùng thật thì phải nới `ndocs=50` (trần 0.9463), đổi lại
mỗi lần sinh bài nộp mất ~21 phút thay vì 74 giây.

Cần thêm `sentencepiece` và `protobuf` (xem `requirements.txt`).

---

## Bài học đo đạc

Ở cỡ mẫu 200–500 câu, **chênh lệch dưới ~0.01 không đọc được**. Đã ba lần suýt chọn nhầm
tham số vì tin vào đỉnh của một lần quét lưới:

- `α=0.8` — đỉnh 0.9061 trên seed 777, tụt xuống 0.8445 trên seed 2024
- `k1/b` — phải tới 2.000 câu + bootstrap mới thấy khoảng tin cậy chứa 0, hai cấu hình
  thực chất không phân biệt được
- cross-encoder — `beta=1.0` cho 0 cải thiện, `beta=0.5` cho +0.03

Quy trình đã chốt: quét lưới trên một mẫu → **kiểm chứng lại trên mẫu khác seed** → chỉ tin
**delta**, không tin giá trị tuyệt đối. Và bằng chứng từ public test luôn thắng mọi phép đo
trên train.

---

## Việc tiếp theo

1. **Fine-tune `halong_embedding`** trên 6.000 cặp (query, chunk đúng) đào từ `train.json` —
   mọi thành phần hiện tại đều zero-shot, chưa cái nào nhìn thấy một dòng văn bản pháp luật
   của cuộc thi. Đây là lực đòn bẩy lớn nhất còn lại.
2. **Từ điển viết tắt** phía query — rẻ, đo được trong 10 phút.
3. **Word segmentation + bigram cho BM25** — "mức lương cơ sở" đang bị xé thành 4 âm tiết rời.
4. **Bật lại cross-encoder** với `ndocs=50`.
