# Document-level reranker experiment

Đây là pipeline **thử nghiệm** cho Task 1 LegalIR. Nó không thay thế pipeline đã đạt
`0.9456` và không sửa index, embedding hoặc checkpoint đang có.

Mục tiêu là xử lý điểm lệch quan trọng nhất của pipeline cũ:

- BTC gán nhãn ở mức **văn bản**;
- cross-encoder đang chấm ở mức **chunk**;
- pipeline cũ dùng công thức cố định để gộp chunk thành văn bản.

Pipeline này giữ nguyên BM25, dense retriever và cross-encoder, sau đó học một
LambdaMART nhỏ để xếp hạng trực tiếp 50 văn bản ứng viên.

```text
BM25 + dense -> top-50 documents -> CE scores (2 chunks/document)
             -> document features -> LambdaMART -> top-5
```

## Trạng thái và nguyên tắc an toàn

- Tất cả sản phẩm chạy nằm trong `DocumentReranker-LegalIR/artifacts/` và không được
  commit.
- Script từ chối ghi đè output đã có, trừ khi truyền `--overwrite`.
- `Retrieval-LegalIR/rerank.py` chỉ được mở rộng để nhận `max_length`; mặc định vẫn là
  512 như trước.
- Default cũ bị lệch trong `Retrieval-LegalIR/prior.py` đã được đồng bộ với cấu hình
  public tốt nhất: `beta=0.5`, `agg=max+0.4`.
- `Finetune-LegalIR/train_ce.py` vẫn giữ `maxlen=256`. Thử train CE ở 384/512 là một
  ablation riêng, không được trộn vào thí nghiệm LambdaMART đầu tiên.

Không đổi default production sang LambdaMART cho đến khi delta Recall@5 dương trên
validation sạch và được xác nhận trên public/private.

## Các file

| File | Chức năng |
|---|---|
| `export_features.py` | Chạy retrieval + CE và xuất một JSONL row cho mỗi `(qid, doc_id)` |
| `train_ranker.py` | Train LightGBM LambdaMART, early-stop theo NDCG@5 |
| `evaluate.py` | So sánh paired với baseline trên cùng candidate set |
| `predict.py` | Tạo `submission.json` và ZIP từ public features |
| `run_experiment.py` | Wrapper chạy các stage theo config |
| `common.py` | Feature schema, I/O và Recall@5/bootstrap dùng chung |
| `config.example.json` | Cấu hình mẫu; copy thành file riêng trước khi chạy |

## Feature schema

Mỗi document có các nhóm feature sau:

- hybrid: rank, max, second, mean;
- BM25: max và second chunk;
- dense: max và second chunk;
- cross-encoder: max, second, gap và mean;
- `baseline_score`: tái tạo pipeline `beta=0.5`, `max+0.4`, sau đó cộng popularity
  prior với `lambda=0.2`;
- số chunk được chấm và tổng số chunk của văn bản;
- popularity tính từ nhãn train.

Khi export validation, popularity chỉ được đếm từ train split. Khi export train,
đáp án của chính query hiện tại được trừ khỏi popularity (leave-one-query-out). Điều
này tránh feature nhìn thấy nhãn của chính hàng đang học.

## Chuẩn bị

Pipeline dùng lại các artifact hiện có:

```text
Retrieval-LegalIR/index/
Retrieval-LegalIR/emb_v2ft/
Finetune-LegalIR/models/reranker-ft/
Finetune-LegalIR/data/split.json
LegalIR - Public Test/train.json
LegalIR - Public Test/public-official.json
```

Nếu đi tiếp từ pipeline `feat/popularity-prior`, hãy chạy đến khi đã có `index/`,
`emb_v2ft/` và `reranker-ft/`, rồi chuyển sang pipeline này. Không cần chạy
`prior.py dump/sweep/submit` vì các bước export, xếp hạng và tạo submission bên dưới sẽ
thay thế phần đó.

`split.json` cần có dạng:

```json
{
  "train": ["qid-1", "qid-2"],
  "val": ["qid-3", "qid-4"]
}
```

Nếu file hiện tại chỉ có `val`, `export_features.py` vẫn suy ra tập dùng để tính prior
bằng phần bù của `val`, nhưng `run_experiment.py` cần cả key `train` để export train.

Cài dependency nền của repo rồi cài thêm LightGBM:

```bash
pip install -r requirements.txt
pip install -r DocumentReranker-LegalIR/requirements.txt
```

Trên macOS, LightGBM còn cần OpenMP runtime. Nếu import báo thiếu `libomp.dylib`, cài:

```bash
brew install libomp
```

Máy Linux/CUDA thường không cần bước riêng này. LambdaMART chạy CPU; GPU chỉ được dùng ở
bước export feature bởi dense encoder và cross-encoder.

## Chạy bằng config

Từ root repo:

```bash
cp DocumentReranker-LegalIR/config.example.json \
   DocumentReranker-LegalIR/config.local.json
```

Sửa các path trong `config.local.json` cho đúng máy. Không commit file này nếu nó chứa
đường dẫn dữ liệu cá nhân.

Chạy train/validation đầy đủ:

```bash
python DocumentReranker-LegalIR/run_experiment.py \
  --config DocumentReranker-LegalIR/config.local.json
```

Các stage có thể chạy riêng để không phải chấm CE lại:

```bash
python DocumentReranker-LegalIR/run_experiment.py \
  --config DocumentReranker-LegalIR/config.local.json \
  --stage export

python DocumentReranker-LegalIR/run_experiment.py \
  --config DocumentReranker-LegalIR/config.local.json \
  --stage train --stage evaluate
```

Sau khi validation đã đạt tiêu chí, mới export public và tạo submission:

```bash
python DocumentReranker-LegalIR/run_experiment.py \
  --config DocumentReranker-LegalIR/config.local.json \
  --stage public
```

Không dùng `--overwrite` nếu chưa chắc muốn thay artifact. Muốn chạy cấu hình khác nên
đổi `paths.artifacts`, ví dụ:

```text
DocumentReranker-LegalIR/artifacts_seed2026/
DocumentReranker-LegalIR/artifacts_maxlen384/
```

## Chạy từng bước

### 1. Export train

```bash
python DocumentReranker-LegalIR/export_features.py \
  --questions "LegalIR - Public Test/train.json" \
  --split Finetune-LegalIR/data/split.json \
  --split-name train \
  --prior-train "LegalIR - Public Test/train.json" \
  --index Retrieval-LegalIR/index \
  --emb Retrieval-LegalIR/emb_v2ft \
  --reranker Finetune-LegalIR/models/reranker-ft \
  --output DocumentReranker-LegalIR/artifacts/train_features.jsonl
```

Để smoke test, thêm `--limit 10`. Smoke test chỉ kiểm tra pipeline chạy được, không dùng
kết quả đó để kết luận điểm.

### 2. Export validation

Giống lệnh trên nhưng đổi:

```text
--split-name val
--output DocumentReranker-LegalIR/artifacts/val_features.jsonl
```

### 3. Train

```bash
python DocumentReranker-LegalIR/train_ranker.py \
  --train DocumentReranker-LegalIR/artifacts/train_features.jsonl \
  --valid DocumentReranker-LegalIR/artifacts/val_features.jsonl \
  --model-out DocumentReranker-LegalIR/artifacts/document_ranker.txt
```

### 4. Evaluate

```bash
python DocumentReranker-LegalIR/evaluate.py \
  --features DocumentReranker-LegalIR/artifacts/val_features.jsonl \
  --model DocumentReranker-LegalIR/artifacts/document_ranker.txt \
  --output DocumentReranker-LegalIR/artifacts/evaluation.json
```

Output quan trọng:

```text
candidate_recall
baseline_recall_at_5
candidate_recall_at_5
delta
improved_queries
damaged_queries
bootstrap_95_ci
```

### 5. Public submission

Export public không truyền `--split`; popularity sẽ dùng toàn bộ train:

```bash
python DocumentReranker-LegalIR/export_features.py \
  --questions "LegalIR - Public Test/public-official.json" \
  --prior-train "LegalIR - Public Test/train.json" \
  --index Retrieval-LegalIR/index \
  --emb Retrieval-LegalIR/emb_v2ft \
  --reranker Finetune-LegalIR/models/reranker-ft \
  --output DocumentReranker-LegalIR/artifacts/public_features.jsonl

python DocumentReranker-LegalIR/predict.py \
  --features DocumentReranker-LegalIR/artifacts/public_features.jsonl \
  --model DocumentReranker-LegalIR/artifacts/document_ranker.txt \
  --output DocumentReranker-LegalIR/artifacts/submission_document_ranker.zip
```

Có thể tạo submission baseline từ chính candidate file để kiểm tra tái lập:

```bash
python DocumentReranker-LegalIR/predict.py \
  --features DocumentReranker-LegalIR/artifacts/public_features.jsonl \
  --method baseline \
  --output DocumentReranker-LegalIR/artifacts/submission_baseline_check.zip
```

Submission baseline này phải gần pipeline `prior.py` hiện tại. Nếu không, dừng lại và
kiểm tra feature export trước khi tin kết quả LambdaMART.

## Tiêu chí quyết định

Chỉ coi thí nghiệm có triển vọng khi:

1. candidate recall khớp phép đo trước đó trong sai số hợp lý;
2. baseline tái lập được pipeline hiện tại;
3. delta Recall@5 dương;
4. số query được cứu lớn hơn số query bị làm hỏng;
5. bootstrap 95% CI không cho thấy mức giảm đáng kể;
6. kết quả giữ được trên một split chưa dùng để tune.

Nếu đạt, bước tích hợp sau này nên thêm lựa chọn `--document-ranker heuristic|lambdamart`
vào pipeline production. Không xóa heuristic cũ để luôn có baseline và rollback.
