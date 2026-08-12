# Hướng dẫn chạy — DSC 2026 Task 1 (LegalIR)

Tài liệu này để bạn dựng lại **toàn bộ hệ thống từ số không**: chunk corpus, dựng index,
trích vector, fine-tune model, rerank, và sinh bài nộp.

Số liệu và lý do đằng sau từng lựa chọn nằm ở [RESULTS.md](RESULTS.md). File này chỉ nói
*làm thế nào*.

---

## 1. Hệ thống này làm gì

Cho một câu hỏi pháp luật tiếng Việt, trả về **5 văn bản** có khả năng chứa câu trả lời
nhất. Độ đo chính là **Recall@5**.

Corpus: 8.532 văn bản pháp luật, tổng 353 triệu ký tự. Văn bản dài nhất 1,24 triệu từ —
dài gấp hàng nghìn lần giới hạn của một encoder, nên **chunking là bắt buộc chứ không phải
tuỳ chọn**.

Đường đi của một câu hỏi qua hệ thống:

```
                      câu hỏi
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
   BM25 (mặt chữ)                  bi-encoder (ngữ nghĩa)
   inverted index trên đĩa         487k vector trên đĩa
        │                                 │
        │  lấy 2.000 chunk                │  chấm lại đúng 2.000 chunk đó
        └────────────────┬────────────────┘
                         ▼
              hoà điểm: 0.7×dense + 0.3×BM25
                         ▼
              50 văn bản × 2 chunk mạnh nhất
                         ▼
              cross-encoder chấm lại từng cặp
                         ▼
              trộn: 0.7×cross-encoder + 0.3×hybrid
                         ▼
              gộp chunk → văn bản (trung bình 2 chunk cao nhất)
                         ▼
              + prior độ phổ biến (λ=0.2)
                         ▼
                    top-5 văn bản
```

Ba tầng, mỗi tầng lọc hẹp dần: **487.194 chunk → 2.000 → 100 cặp → 5 văn bản**. Tầng sau
đắt hơn tầng trước hàng trăm lần, nên phải lọc mạnh trước khi tới nó.

---

## 2. Cài đặt

```bash
pip install -r requirements.txt
```

Cần thêm cho fine-tune và cross-encoder:

```bash
pip install sentence-transformers accelerate peft sentencepiece protobuf
```

**GPU gần như bắt buộc.** Toàn bộ số liệu dưới đây đo trên RTX 3050 Laptop 6 GB. Không có
GPU thì bước trích vector từ 2 tiếng thành hàng chục tiếng.

### Đổi chỗ chứa model HuggingFace

Model tải về khá nặng (bge-m3 hơn 4 GB). Nếu ổ C: chật:

```bash
# Windows PowerShell
setx HF_HOME "D:\hf_cache"
# Linux/macOS
export HF_HOME=~/hf_cache
```

---

## 3. Dữ liệu

Repo **không** chứa dữ liệu cuộc thi. Giải nén vào thư mục gốc:

```
LegalIR - Public Test/
  selected-contexts/            8.532 file context_*.json
  public-official.json          1.000 câu hỏi cần trả lời (answer = null)
  train.json                    7.000 câu hỏi + đáp án
```

Giải nén thường tạo thêm một cấp thư mục con, thành
`selected-contexts/selected-contexts/context_*.json`. Các script đều tự dò xuống một cấp,
nhưng nếu báo không tìm thấy thì trỏ thẳng vào **đúng cấp chứa file**.

---

## 4. Chạy nhanh — một lệnh

```bash
python run_all.py --data "LegalIR - Public Test"
```

Ra thẳng `Retrieval-LegalIR/submission.zip` đã kiểm tra hợp lệ. Bước nào đã có sản phẩm thì
tự bỏ qua, nên đứt giữa chừng chạy lại không mất gì.

| Bước | Thời gian | Sản phẩm |
|---|---:|---|
| 1. chunk corpus | 1m31s | `chunks.jsonl` 592 MB |
| 2. index BM25 | 202s | `index/` 286 MB |
| 3. trích vector | 36 phút | `emb/` 714 MB |
| 4. sinh bài nộp | 74s | `submission.zip` |
| 5. kiểm tra | vài giây | |

**Đường này cho khoảng Recall@5 = 0.86.** Nó dừng ở tầng hybrid, chưa có cross-encoder và
prior. Muốn điểm cao nhất thì xem mục 5.

Không có GPU: thêm `--skip-encode` để chạy BM25 thuần — nhanh, nhưng mất ~0.05 Recall.

---

## 5. Chạy đầy đủ — cấu hình đạt 0.9277

`run_all.py` chưa bao phủ hai tầng cuối. Đường đầy đủ chạy thủ công, sáu bước.

### 5.1 Chunk corpus

```bash
cd Chunking-LegalIR
python chunker.py \
  --contexts "../LegalIR - Public Test/selected-contexts/selected-contexts" \
  --out chunks.jsonl
```

**1m31s → 592 MB, 487.194 chunk.**

Chunker chia làm ba tầng theo cấu trúc pháp lý:

| Tầng | Điều kiện | Số chunk |
|---|---|---:|
| 1 | cắt theo `Điều N` (85,9% văn bản có) | 122.520 |
| 2 | Điều > 350 từ → cắt tiếp theo `Khoản`, vẫn dài → cửa sổ cố định | 333.416 |
| 3 | văn bản không có `Điều` (công văn, TCVN) → cửa sổ 256 từ, overlap 25% | 31.238 |
| 0 | `passage` rỗng → vớt tiêu đề từ slug trong `link` | 20 |

Tầng 0 nghe như chi tiết vụn nhưng **6 trong 20 văn bản rỗng đó là đáp án gold** — bỏ tầng
này là mất trắng 6 câu.

Xem chunk của một văn bản cụ thể trước khi chạy cả corpus:

```bash
python chunker.py --contexts "<...>/selected-contexts" --demo 43443 --preview 5
```

### 5.2 Dựng index BM25

```bash
cd ../Retrieval-LegalIR
python bm25.py build --chunks ../Chunking-LegalIR/chunks.jsonl --index index
```

**202s → 286 MB.**

Index là inverted index ghi thẳng ra đĩa dạng `np.memmap`, dựng bằng hai lượt quét: lượt
một đếm document frequency để biết offset, lượt hai ghi postings vào đúng ô. Lúc query chỉ
đọc postings của vài từ trong câu hỏi nên **RAM dùng gần như bằng 0** — quan trọng với máy
yếu.

Thử ngay:

```bash
python bm25.py query --index index "mức lương cơ sở là bao nhiêu" --topk 10 --group-doc
```

### 5.3 Trích vector

```bash
python encode.py \
  --chunks ../Chunking-LegalIR/chunks.jsonl \
  --out emb_v2 \
  --model "AITeamVN/Vietnamese_Embedding_V2"
```

**112 phút → 952 MB** (487.194 × 1024 chiều, fp16).

`encode.py` **tự đọc kiểu pooling từ `1_Pooling/config.json` của model**, không cắm cứng.
Chi tiết này quan trọng: họ e5 (halong) dùng mean pooling, họ BGE-M3 dùng CLS pooling. Lấy
trung bình token của một model huấn luyện theo CLS sẽ ra vector lệch hẳn mà **không báo lỗi
gì** — hỏng âm thầm, kiểu tệ nhất.

Kiểu pooling được ghi vào `emb_v2/meta.json`, và `hybrid.py` đọc lại từ đó để encode câu hỏi
đúng cách. Đừng sửa tay file này.

Các model đã thử, đo trên 500 câu validation:

| Model | Tham số | Recall@5 tầng hybrid |
|---|---:|---:|
| `AITeamVN/Vietnamese_Embedding_V2` | 568M | **0.9103** |
| `halong-ft` (halong fine-tune 6.000 cặp) | 278M | 0.9067 |
| `hiieu/halong_embedding` (gốc) | 278M | thấp hơn rõ |
| `BAAI/bge-m3` | 568M | thấp hơn V2 |

### 5.4 Fine-tune — tuỳ chọn, xem mục 6

Bỏ qua được. Cross-encoder đã fine-tune (`reranker-ft`) đóng góp **+0.027**, đáng làm nếu
bạn có ~1 tiếng GPU.

### 5.5 Sinh bài nộp

Hai lệnh: một lệnh chạy cả ba tầng và lưu 50 ứng viên kèm điểm, một lệnh áp prior rồi đóng
gói.

```bash
python prior.py dump \
  --questions "../LegalIR - Public Test/public-official.json" \
  --emb emb_v2 \
  --reranker ../Finetune-LegalIR/models/reranker-ft \
  -o cands_pub.json

python prior.py submit -c cands_pub.json --lam 0.2 -o submission
```

**27 phút** cho bước `dump` (cross-encoder chiếm gần hết), vài giây cho `submit`.

Tách làm hai bước là có chủ đích: `dump` lưu lại **50 ứng viên kèm điểm** thay vì chỉ top-5.
Nhờ vậy đổi tham số prior chỉ tốn vài giây thay vì chạy lại 27 phút.

**Prior độ phổ biến** là gì: văn bản từng là đáp án nhiều lần trong `train.json` được cộng
thêm điểm. Văn bản từng làm đáp án ≥20 lần có xác suất làm đáp án cao gấp **50 lần** văn
bản chưa từng. BM25 và dense đều không biết thông tin này — nó đến miễn phí từ nhãn train.

```
điểm cuối = minmax(điểm) + λ × log1p(số lần làm đáp án) / log1p(max)
```

λ=0.2. Lúc sinh bài nộp, tần suất đếm trên **cả 7.000 câu train** (public test rời nhau với
train nên không rò rỉ). Lúc đo trên validation thì phải loại 1.000 câu đang chấm ra khỏi
phép đếm, nếu không mỗi câu tự cộng điểm cho đáp án của chính nó.

### 5.6 Kiểm tra trước khi nộp

```bash
python ../Validator-Task-LegalIR/validate_submission.py submission.zip \
  --ref "../LegalIR - Public Test/public-official.json" \
  --corpus "../LegalIR - Public Test/selected-contexts/selected-contexts"
```

Validator gom hết lỗi rồi in một lần, có 14 mã lỗi. Nguy hiểm nhất là **E010**:
`context_*.json` lưu `id` kiểu số nguyên nhưng đáp án bắt buộc là chuỗi. Sai chỗ đó thì
chương trình chấm cho **0 điểm mà không báo gì**.

---

## 6. Fine-tune

Toàn bộ code nằm trong `Finetune-LegalIR/`. Ba bước: đào dữ liệu → train → đo.

### 6.1 Đào dữ liệu huấn luyện

```bash
cd Finetune-LegalIR
python mine_data.py \
  --train "../LegalIR - Public Test/train.json" \
  --emb ../Retrieval-LegalIR/emb_v2 \
  --out data
```

**~6 phút → `data/pairs.jsonl` (49 MB) + `data/split.json`.**

Vấn đề cốt lõi: **BTC gán nhãn ở mức văn bản, nhưng model học ở mức chunk.** Văn bản gold
có median 33 chunk, phải chọn một cái làm positive. Cách làm hiện tại là lấy chunk có điểm
hybrid cao nhất trong văn bản gold — giám sát yếu, và model có nguy cơ học đồng ý với chính
nó. Đây là điểm yếu đã biết của pipeline, chưa có cách tốt hơn.

Hard negative lấy từ chunk thuộc văn bản **không phải** gold, hạng 5–80.

```bash
--skip-top 0     # lấy cả 4 hạng đầu (đã thử: kém hơn -0.005)
--neg 6          # số negative mỗi câu
```

Bốn hạng đầu bị bỏ có lý do: corpus có 5.427 văn bản chưa từng là đáp án, rất có thể chúng
vẫn liên quan mà chỉ không được gán nhãn. Dạy model đè chúng xuống là dạy sai. **Đã thử
`--skip-top 0` và điểm public tụt 0.005** — trực giác ban đầu đúng.

`split.json` chia 6.000 câu huấn luyện / 1.000 câu validation, seed 42. **Mọi phép đo sau
này đều dùng đúng tập 1.000 câu này** — nếu bạn đào lại dữ liệu thì kiểm tra split không
đổi, nếu không mọi so sánh với số cũ đều vô nghĩa.

### 6.2 Fine-tune bi-encoder

```bash
python train.py --epochs 1 --out models/halong-ft
```

**11 phút, VRAM đỉnh 5,58/6,4 GB → +0.029 Recall.**

Loss là `CachedMultipleNegativesRankingLoss` bọc trong `MatryoshkaLoss`. MNRL dùng mọi mẫu
khác trong batch làm negative miễn phí, nên **batch càng lớn càng tốt**; bản Cached
(GradCache) chia làm hai lượt nên batch không còn bị VRAM chặn.

```bash
python train.py --smoke                  # 30 bước, đo VRAM và tốc độ trước
python train.py --batch 32 --mini 4      # hạ xuống nếu hết VRAM
```

**Đừng tăng số epoch.** Đã thử 3 epoch: loss giảm từ 3.34 xuống 1.82 nhưng chất lượng đứng
yên (7 câu tốt lên, 6 câu xấu đi). Đường loss đi bậc thang — rơi một nấc đúng mỗi ranh giới
epoch rồi nằm ngang — đó là chữ ký của học thuộc. **5.999 cặp bị vắt kiệt sau đúng một
lượt.** Muốn tiến thêm thì phải thêm dữ liệu, không phải thêm vòng lặp.

> **Chỉ chạy được với model cỡ 278M.** `train.py` đang full fine-tune, mà model 568M như
> Vietnamese_Embedding_V2 cần ~7,9 GB chỉ riêng cho trọng số và trạng thái AdamW — vượt
> 6,4 GB VRAM. Muốn fine-tune model 568M thì phải chuyển `train.py` sang LoRA; `train_ce.py`
> đã có sẵn mẫu để bê sang. **Chưa ai làm.**

Sau khi train phải encode lại toàn bộ corpus bằng model mới (mục 5.3), 112 phút.

### 6.3 Fine-tune cross-encoder

```bash
python train_ce.py --data data/pairs.jsonl --out models/reranker-ft
```

**38–51 phút, VRAM 2,54 GB → +0.027 Recall.**

Dùng LoRA (r=16) + gradient checkpointing nên model 568M vừa thoải mái trong 6 GB. Xong thì
LoRA được gộp thẳng vào trọng số gốc, nên lúc dùng không cần `peft`.

Loss mặc định là `BinaryCrossEntropy` **pointwise**: mỗi cặp (câu hỏi, chunk) là một mẫu
độc lập, nhãn 1.0 hoặc 0.0. Cách đọc con số loss: nó là `-log(xác suất gán cho nhãn đúng)`.
Với tỉ lệ 1 dương / 4 âm, một model đoán bừa 0.2 cho mọi cặp đã đạt loss **0.50** — đó là
mốc để biết loss của bạn có ý nghĩa hay không.

```bash
python train_ce.py --listwise    # đổi sang CachedMultipleNegativesRankingLoss
```

Cờ `--listwise` tối ưu **thứ tự** thay vì điểm tuyệt đối, tức đúng thứ Recall@5 cần. Chậm
hơn nhiều (~86 phút so với 38). **Chưa ai chạy nhánh này** — đây là hướng còn bỏ ngỏ đáng
thử nhất.

### 6.4 Đo lại sau khi fine-tune

```bash
cd ../Retrieval-LegalIR
python ../Finetune-LegalIR/eval_ft.py \
  --emb emb_ft --base emb \
  --train "../LegalIR - Public Test/train.json"
```

---

## 7. Đo đạc và thử nghiệm

Ba script để thử tham số hoặc model mới **mà không phải chạy lại cả pipeline**. Nguyên tắc
chung: chạy phần đắt (cross-encoder) đúng một lần, lưu điểm ra file, rồi quét tham số ngoại
tuyến trong vài giây.

### So sánh nhiều cross-encoder — `ce_bench.py`

```bash
python ce_bench.py pairs -n 500 --emb emb_v2 -o pairs_val.json          # 1 phút
python ce_bench.py score -p pairs_val.json --model <model> --kind seq -o sc_a.json   # 12 phút
python ce_bench.py eval  -p pairs_val.json -s sc_a.json sc_b.json       # vài giây
```

`--kind` chọn cách chấm: `seq` cho cross-encoder phân loại cặp thông thường, `qwen3` và
`gemma` cho reranker kiểu sinh (điểm = logit của token "yes"). Hai họ sau có khuôn prompt
bắt buộc riêng — sai khuôn là điểm vô nghĩa.

Bước `pairs` sinh tập ứng viên **dùng chung cho mọi model**, nên so sánh là **theo cặp trên
cùng câu hỏi**, nhiễu thấp hơn hẳn chạy riêng từng pipeline.

**Mẹo quan trọng: dùng AUC để sàng lọc, Recall@5 để kết luận.** AUC đọc trên hàng nghìn cặp
thay vì vài chục con số đúng/sai, nên phân biệt được model ở cỡ mẫu mà Recall@5 hoàn toàn
mù. Loại ứng viên bằng AUC trước, chỉ chạy Recall đầy đủ cho model sống sót.

### Quét α, β, λ — `sweep_alpha.py`

```bash
python sweep_alpha.py dump  -n 500 --emb emb_v2 -o pool_val.npz    # 1 phút
python sweep_alpha.py score -d pool_val.npz -o sc_union.json       # 23 phút
python sweep_alpha.py sweep -d pool_val.npz -s sc_union.json --beta 0.7 --lam 0.2
```

Mẹo: pool 2.000 chunk do **BM25** chọn nên không phụ thuộc α. Chỉ cần lưu điểm BM25 và dense
thô của cả pool, rồi tính lại `fused = α×dense + (1−α)×BM25` cho α bất kỳ ở ngoài. Bước
`score` chấm **hợp** của mọi tập ứng viên sinh bởi 8 mức α, nên chạy một lần là quét được
cả dải.

### Sàng lọc embedder — `emb_screen.py`

```bash
python emb_screen.py --models "BAAI/bge-m3" "AITeamVN/Vietnamese_Embedding_V2"
```

Chỉ encode ~32k chunk đã nằm trong tập ứng viên thay vì cả 487k — **5 phút thay vì 112**.

**Cảnh báo về script này:** nó chỉ đo khả năng *xếp lại trong tập ứng viên có sẵn*, mà tập
đó do model hiện tại chọn ra — sân nhà của model hiện tại. Thực tế đã có một lần kết quả
sàng lọc lệch hẳn so với khi chạy đủ đường ống: V2 thua 0.062 khi sàng lọc nhưng **hoà**
khi chạy thật. Dùng nó để loại ứng viên tệ, đừng dùng để chốt.

### Quét k1/b của BM25

```bash
python sweep_bm25.py --index index --train "../LegalIR - Public Test/train.json" -n 200
```

Không cần dựng lại index: postings lưu tf thô, đổi k1/b chỉ đổi mẫu số.

### Giao diện web tra cứu

```bash
python serve.py --index index --contexts "../LegalIR - Public Test/selected-contexts"
```

Mở http://127.0.0.1:8000 — tìm kiếm, bấm vào kết quả để đọc toàn văn render theo
Chương/Điều/Khoản, tự nhảy tới Điều khớp, tô vàng từ khoá. Chỉ dùng `http.server` của
stdlib, không cần cài thêm.

---

## 8. Bảng tham số

Tất cả đều đã quét và kiểm chứng trên 500–1.000 câu validation. Xem [RESULTS.md](RESULTS.md)
để biết cỡ mẫu và khoảng tin cậy từng cái.

| Tham số | Giá trị | Ở đâu | Ghi chú |
|---|---|---|---|
| `k1` | 2.5 | BM25 | |
| `b` | 0.9 | BM25 | |
| `pool` | 2000 | hybrid | chứa gold cho 99,2% câu |
| `alpha` | **0.7** | hybrid | 70% dense / 30% BM25. Đỉnh sạch, một đỉnh, đã kiểm chứng ba lần |
| `ndocs` | 50 | rerank | số văn bản đưa vào cross-encoder |
| `mchunks` | 2 | rerank | số chunk mỗi văn bản |
| `beta` | **0.7** | rerank | 70% cross-encoder / 30% hybrid. β=1.0 (đè hẳn) luôn tệ hơn |
| `agg` | top-2 mean | gộp | `top3` đúng cho BM25 thuần nhưng sai sau khi có dense |
| `lam` | 0.2 | prior | vùng 0.05–0.3 đều dương, 0.5 trở lên âm |

Hai tham số cần cẩn thận:

**`beta` không nên là 1.0.** Để cross-encoder đè hẳn thay vì trộn cho kết quả tệ hơn ở mọi
cấu hình đã thử (−0.023 đến −0.054). Hybrid đã là xếp hạng tốt, cross-encoder chỉ nên góp ý.

**`lam` đo được rất yếu.** Hiệu ứng khoảng +0.005, mà 500 câu validation chỉ phân giải được
tới ~0.01. Đường cong λ nhảy loạn chứ không trơn như α. Đừng tinh chỉnh nó — để 0.2 và đi
làm việc khác.

---

## 9. Bản đồ file

```
run_all.py                    chạy pipeline cơ bản bằng một lệnh (~0.86)
requirements.txt
RESULTS.md                    mọi số liệu đã đo, kèm cỡ mẫu và CI
RUNNING.md                    file này

Chunking-LegalIR/
  chunker.py                  chunker 3 tầng theo cấu trúc pháp lý

Retrieval-LegalIR/
  bm25.py                     dựng và truy vấn inverted index trên đĩa
  encode.py                   trích vector → memmap (tự dò pooling)
  hybrid.py                   hoà điểm BM25 + dense
  rerank.py                   tầng cross-encoder
  prior.py                    prior độ phổ biến + sinh bài nộp  ← đường sản xuất
  predict.py                  sinh bài nộp không có prior
  serve.py                    giao diện web tra cứu
  ce_bench.py                 so sánh nhiều cross-encoder
  sweep_alpha.py              quét α/β/λ, chạy CE một lần
  emb_screen.py               sàng lọc embedder nhanh
  eval_bm25.py, sweep_bm25.py đo BM25

Finetune-LegalIR/
  mine_data.py                đào cặp huấn luyện từ train.json
  train.py                    fine-tune bi-encoder
  train_ce.py                 fine-tune cross-encoder (LoRA)
  eval_ft.py                  đo model đã fine-tune
  sweep_all.py                quét pool × alpha × cách gộp

Validator-Task-LegalIR/
  validate_submission.py      kiểm tra bài nộp, 14 mã lỗi
```

### Sản phẩm trung gian (không nằm trong repo, tự sinh ra)

| Đường dẫn | Dung lượng | Sinh bởi | Thời gian |
|---|---:|---|---:|
| `Chunking-LegalIR/chunks.jsonl` | 592 MB | `chunker.py` | 1m31s |
| `Retrieval-LegalIR/index/` | 286 MB | `bm25.py build` | 202s |
| `Retrieval-LegalIR/emb_v2/` | 952 MB | `encode.py` | 112 phút |
| `Finetune-LegalIR/data/pairs.jsonl` | 49 MB | `mine_data.py` | 6 phút |
| `Finetune-LegalIR/models/` | ~2,3 GB/model | `train*.py` | 11–51 phút |

Tổng cần khoảng **5 GB đĩa trống**, chưa kể cache HuggingFace.

---

## 10. Sự cố thường gặp

### `OSError: paging file is too small` hoặc CUDA OOM khi VRAM còn trống

Windows bắt mọi cấp phát VRAM phải có commit hệ thống tương ứng, nên **hết pagefile biểu
hiện thành lỗi CUDA OOM** dù `nvidia-smi` báo GPU còn trống. Dấu hiệu: CUDA báo không xin
nổi 20 MB trong khi còn 4 GB VRAM.

Kiểm tra:

```powershell
Get-CimInstance Win32_OperatingSystem | Select-Object @{n='CommitFree_GB';e={[math]::Round($_.FreeVirtualMemory/1MB,2)}}
```

Dưới 3 GB là sẽ gặp vấn đề. Sửa (PowerShell **admin**, có hiệu lực ngay, không cần reboot
nếu đặt trên ổ khác):

```powershell
New-CimInstance -ClassName Win32_PageFileSetting -Property @{Name='D:\pagefile.sys'; InitialSize=[uint32]16384; MaximumSize=[uint32]24576}
```

Nếu báo `Value out of range` thì ổ đó không đủ chỗ trống — chọn ổ khác.

### Encode xong nhưng Recall tụt thảm

Gần như chắc chắn là **lệch pooling hoặc lệch model giữa câu hỏi và văn bản**. Kiểm tra
`emb_*/meta.json`:

```json
{"model": "...", "pooling": "cls", "dim": 1024}
```

`hybrid.py` đọc cả hai trường này để encode câu hỏi cho khớp. Nếu bạn encode bằng model A mà
truy vấn bằng model B, hai bên nằm ở hai không gian vector khác nhau — **không có lỗi nào
được ném ra**, chỉ là điểm tệ đi. Bug này đã từng tồn tại trong repo và ngốn 0.006 Recall.

### `train.py` hết VRAM

```bash
python train.py --smoke                  # đo trước, 30 bước
python train.py --batch 32 --mini 4      # mini mới là thứ ăn VRAM, không phải batch
python train.py --maxlen 192
```

`--mini` là kích thước lô con của GradCache, đó mới là thứ quyết định VRAM. `--batch` chỉ
quyết định số negative trong batch, gần như miễn phí.

### Không nạp nổi cross-encoder fp32 (2,27 GB)

Chuyển checkpoint sang fp16 một lần, còn 1,14 GB và nạp nhanh hơn hẳn. Chỉ cần nếu pagefile
chật — sau khi sửa pagefile thì nạp thẳng fp32 vẫn được.

### Tiếng Việt lỗi font trên console Windows

Mọi script đã có `sys.stdout.reconfigure(encoding='utf-8')`. Nếu viết script mới có `print()`
tiếng Việt thì nhớ thêm dòng đó, không thì `UnicodeEncodeError: 'charmap' codec`.

---

## 11. Hạn chế đã biết

**Đường dẫn dữ liệu bị cắm cứng trong ba script.** `prior.py`, `ce_bench.py`, `sweep_alpha.py`
có hằng số `BASE` trỏ tới tên thư mục dữ liệu cụ thể trên máy phát triển
(`LegalIR - Public Test-20260806T081424Z-1-001/...`). Chạy trên máy khác phải sửa hằng số
đó, hoặc đặt tên thư mục dữ liệu y hệt. Nên tham số hoá, chưa làm.

**`run_all.py` chưa bao phủ tầng cross-encoder và prior**, nên nó dừng ở ~0.86 trong khi
đường thủ công đạt 0.9277.

**Positive lúc đào dữ liệu là giám sát yếu** — chọn chunk có điểm hybrid cao nhất trong văn
bản gold, tức model học đồng ý với chính nó. Có thể là nguyên nhân khiến cross-encoder chỉ
xếp chunk gold ở phân vị 0.833 thay vì cao nhất.

**`train.py` chưa hỗ trợ LoRA**, nên không fine-tune được model 568M trên GPU 6 GB.
