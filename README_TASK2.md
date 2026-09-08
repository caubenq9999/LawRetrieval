# UIT-DSC 2026 Task 2 — LegalQA

Tài liệu này tóm tắt pipeline, kết quả thử nghiệm và cách chạy lại phần Task 2.
Toàn bộ dữ liệu, index, embedding và checkpoint của Task 2 được tách độc lập với
Task 1 theo quy định của ban tổ chức.

## Kết quả tốt nhất hiện tại

Pipeline tốt nhất đã đo được:

```text
Task 2 selected-contexts
  -> 487.194 chunks
  -> BM25 + AITeamVN/Vietnamese_Embedding_V2
  -> hybrid retrieval (alpha=0.7)
  -> AITeamVN/Vietnamese_Reranker zero-shot (beta=0.6)
  -> top 3 chunks
  -> rule-based answer, style=cite
  -> submission.json
```

| Cấu hình | Tập đo | METEOR | ROUGE-L |
|---|---:|---:|---:|
| Rule baseline, 3 chunks + cite | Public | 0.5104 | 0.4618 |
| Rule, 4 chunks + cite | Public | 0.4954 | — |
| Hybrid `alpha=0.5`, 3 chunks + cite | Public | 0.4860 | — |
| Hybrid `alpha=0.5`, 2 chunks + cite | Public | 0.4752 | — |
| BM25 experiment | Public | 0.4321 | — |
| **Hybrid + zero-shot reranker, 3 chunks + cite** | **Public** | **0.5241** | — |
| Hybrid `alpha=0.7`, chưa rerank | Validation 200 | 0.5235 | — |
| **Hybrid + zero-shot reranker `beta=0.6`** | **Validation 200** | **0.5347** | **0.4688** |

Điểm validation dùng để so sánh các cấu hình trong repo; điểm public là điểm trả
về từ hệ thống chấm của cuộc thi. Không nên so trực tiếp hai tập với nhau.

### Sweep alpha trên validation

| Alpha | METEOR |
|---:|---:|
| 0.60 | 0.5079 |
| **0.70** | **0.5235** |
| 0.75 | 0.5219 |
| 0.80 | 0.5170 |

Kết luận thực nghiệm: dùng `alpha=0.7`, sau đó rerank với `beta=0.6`, lấy ba
chunk và giữ cách dựng đáp án `cite`.

## Những phần đã implement

### 1. Baseline Task 2 độc lập

[`Retrieval-LegalIR/qa_predict.py`](Retrieval-LegalIR/qa_predict.py) hỗ trợ:

- BM25, dense và hybrid retrieval.
- Trộn điểm bằng `alpha`.
- Cross-encoder reranker và trộn điểm bằng `beta`.
- Quét số chunk và hai kiểu đáp án `cite`/`raw`.
- Tính METEOR, ROUGE-L trên dữ liệu train.
- Sinh đúng cấu trúc `submission.json` và file ZIP.
- Xuất retrieval contexts bằng `--candidates-out` để tái sử dụng cho generator.

Notebook [`COLAB_TASK2_QA.ipynb`](COLAB_TASK2_QA.ipynb) dựng và cache toàn bộ
artifact BM25/dense trên Google Drive. Dense encoder mặc định là
`AITeamVN/Vietnamese_Embedding_V2`.

### 2. Fine-tune reranker

[`Finetune-LegalIR/mine_qa_reranker.py`](Finetune-LegalIR/mine_qa_reranker.py)
mine positive và hard negative chỉ từ Task 2. Script
[`Finetune-LegalIR/train_ce.py`](Finetune-LegalIR/train_ce.py) hỗ trợ LoRA hoặc
full fine-tune cross-encoder.

Notebook [`COLAB_TASK2_RERANKER_FT.ipynb`](COLAB_TASK2_RERANKER_FT.ipynb) được
thiết kế cho GPU lớn: full fine-tune 568M tham số, BF16, sequence length 384,
tám hard negatives và effective batch 64.

Kết quả epoch 1:

| Beta | METEOR | ROUGE-L |
|---:|---:|---:|
| 0.4 | 0.5216 | 0.4319 |
| 0.5 | 0.5263 | 0.4327 |
| 0.6 | 0.5249 | 0.4316 |
| **0.7** | **0.5275** | 0.4272 |
| 0.8 | 0.5232 | 0.4237 |

Checkpoint fine-tune chưa vượt reranker zero-shot (`0.5347` METEOR), vì vậy
không dùng checkpoint này cho best submission hiện tại.

### 3. Sinh đáp án bằng Qwen3.5-2B và vLLM

[`Retrieval-LegalIR/qa_generate_vllm.py`](Retrieval-LegalIR/qa_generate_vllm.py)
nhận contexts được xuất từ `qa_predict.py`. Notebook
[`COLAB_TASK2_VLLM_QA.ipynb`](COLAB_TASK2_VLLM_QA.ipynb) chạy model
`Qwen/Qwen3.5-2B` bằng vLLM.

Thử nghiệm viết lại toàn bộ đáp án không hiệu quả:

| Cách sinh trên cùng retrieval contexts | METEOR | ROUGE-L |
|---|---:|---:|
| Rule-based | **0.5347** | **0.4688** |
| Qwen rewrite | 0.3632 | 0.4369 |

METEOR giảm `0.1715` do model diễn giải lại và làm mất token pháp lý trùng với
đáp án tham chiếu. Vì vậy không nên submit chế độ `rewrite`.

Chế độ `conclusion` hiện giữ nguyên rule answer và chỉ yêu cầu Qwen thêm một câu
chốt bắt đầu bằng `Theo đó,`. Validator kiểm tra độ dài, định dạng và không cho
model thay đổi số tiền, thời hạn hoặc số Điều/Khoản; output lỗi sẽ fallback về
rule answer. Chế độ này đã được implement nhưng chưa có kết quả validation cuối
cùng, nên chưa được xem là tốt hơn pipeline rule-based.

## Chuẩn bị Google Drive

Các notebook mặc định dùng cấu trúc:

```text
MyDrive/DSC2026/task2_qa/
├── input/
│   ├── train.json
│   ├── public-official.json
│   └── selected-contexts.zip
└── artifacts/
    ├── chunks_qa_full.jsonl
    ├── index_qa.zip
    ├── emb_qa_v2.zip
    └── qa_reranker_full_e1.zip       # chỉ cần nếu thử model fine-tune
```

Ba file trong `input/` là dữ liệu của ban tổ chức và không được commit vào repo.
Các artifact lớn cũng được lưu trên Drive thay vì GitHub.

### Kiểm tra quan trọng

- `selected-contexts` phải có 8.532 file context.
- `train.json` phải có 7.000 câu.
- `public-official.json` phải có 1.000 câu.
- Full chunk corpus, BM25 index và dense embedding phải cùng có **487.194 hàng**.
- Không dùng `chunks_qa_truncated`: bản này chỉ có 60.508 hàng và sẽ làm lệch
  chunk/index/embedding.
- Không dùng `emb_v2ft`, `v2-full-ft` hoặc reranker học từ Task 1.

## Thứ tự chạy đề xuất

1. Mở [`COLAB_TASK2_QA.ipynb`](COLAB_TASK2_QA.ipynb) để kiểm tra hoặc dựng
   `chunks_qa`, `index_qa` và `emb_qa_v2`.
2. Dùng `qa_predict.py` hoặc các cell cuối notebook để sweep validation.
3. Sinh submission bằng cấu hình tốt nhất: `alpha=0.7`, reranker zero-shot,
   `beta=0.6`, `nchunk=3`, `style=cite`.
4. Chỉ mở notebook fine-tune hoặc vLLM khi cần tiếp tục nghiên cứu; hai hướng này
   chưa vượt best rule-based pipeline.

Ví dụ chạy cấu hình tốt nhất sau khi đã có artifact:

```bash
python Retrieval-LegalIR/qa_predict.py \
  --qa-dir /content/task2_qa \
  --questions /content/task2_qa/public-official.json \
  --index /content/task2_work/index_qa \
  --emb /content/task2_work/emb_qa_v2 \
  --retriever hybrid \
  --alpha 0.7 \
  --reranker AITeamVN/Vietnamese_Reranker \
  --beta 0.6 \
  --batch 128 \
  --nchunk 3 \
  --style cite \
  --out /content/task2_work/submission_qa_best
```

File cần nộp là `/content/task2_work/submission_qa_best.zip`; bên trong ZIP phải
có đúng `submission.json` ở thư mục gốc.

## Hướng phát triển tiếp theo

Ưu tiên các thử nghiệm có thể đo trên cùng validation split:

1. Cải thiện cách mine label cho reranker thay vì train trực tiếp từ pseudo-label
   nhiễu hiện tại.
2. Thêm validator extractive chặt hơn cho câu kết luận của Qwen và chỉ append khi
   câu mới có độ phủ token cao so với rule answer.
3. Tối ưu số chunk theo từng loại câu hỏi thay vì cố định ba chunk.
4. Giữ rule-based answer làm mốc; chỉ nhận thay đổi nếu cả METEOR và ROUGE-L trên
   cùng tập validation không giảm đáng kể.
