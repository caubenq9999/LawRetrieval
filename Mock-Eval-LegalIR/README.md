# Mock Eval — DSC 2026 Task 1: LegalIR

Bộ công cụ chạy thử **chương trình chấm điểm gốc của BTC** ở máy local, dùng dữ liệu
trích từ `train.json` làm reference giả lập. Chỉ dùng thư viện chuẩn + `numpy`
(vì `scoring.py` cần `numpy`).

## Quy trình

```bash
# 1. Trích 50 câu từ train.json, dựng cây thư mục mock
python make_mock.py --train "../LegalIR - Public Test/train.json" --n 50

# 2. Chấm thử một bài nộp
python run_scoring.py --mock mock --submission mock/samples/perfect.json

# 3. Hoặc chấm hết các bài mẫu để so sánh
python run_scoring.py --mock mock --all
```

`run_scoring.py` nhận cả `.zip`, nên có thể test trọn chuỗi thật:

```bash
python ../Validator-Task-LegalIR/validate_submission.py my_sub.zip --ref mock/questions.json
python run_scoring.py --mock mock --submission my_sub.zip
```

## ⚠️ Định dạng file reference

`scoring.py` đọc truth **phẳng**, không qua key `answer`:

```python
recall = ... len(set(y_true[k]) & set(y_pred.get(k, ...)))/len(y_true[k]) ...
```

Nên file reference phải là:

```json
{ "86666": ["280282"], "145746": ["56081"] }
```

**không phải** định dạng của `train.json`:

```json
{ "86666": { "question": "...", "answer": ["280282"] } }
```

Nếu đưa nhầm `train.json` vào làm reference thì `set(y_true[k])` sẽ ra
`{"question", "answer"}`, và **mọi bài nộp đều được 0 điểm** mà không báo lỗi gì.
`make_mock.py` đã tự convert sang dạng phẳng.

## Cây thư mục sinh ra

```
mock/
  input/ref/metadata.json    {"files": {"reference": "truth.json", "input": "submission.json"}}
  input/ref/truth.json       đáp án dạng phẳng {qid: [doc_id]}
  input/res/                 nơi đặt submission.json cần chấm
  output/scores.json         kết quả scoring.py ghi ra
  questions.json             "đề bài" giả lập (answer = null) — dùng làm --ref cho validator
  answer_key.json            đáp án kèm câu hỏi, để tra cứu khi debug
  samples/                   các bài nộp mẫu
```

## Các bài nộp mẫu

| Tên | Mô phỏng | Recall | Precision |
|---|---|---|---|
| `perfect` | dự đoán hoàn hảo — mốc trên | 1.0000 | 1.0000 |
| `top1` | chỉ trả về 1 kết quả/câu | 0.9567 | 1.0000 |
| `padded` | đáp án đúng + nhiễu cho đủ 5 ID | 1.0000 | 0.2200 |
| `half` | đúng khoảng một nửa số câu | 0.4400 | 0.4400 |
| `random` | sai hoàn toàn — mốc dưới | 0.0000 | 0.0000 |
| `violate_limit` | ~10% số câu trả về 6 ID | 0.8600 | 0.8600 |
| `int_ids` | `document_id` để nguyên kiểu `int` | 0.0000 | 0.0000 |

*(số liệu với `--n 50 --seed 42`)*

Hai dòng đáng chú ý:

- **`padded`** — nhồi đủ 5 ID giữ Recall = 1.0 nhưng dìm Precision xuống 0.22. Vì Recall là
  độ đo chính, chiến thuật này **có lợi** trên bảng xếp hạng. Đây là điều BTC nên cân nhắc
  khi thiết kế tiebreak.
- **`int_ids`** — trùng khớp hoàn hảo với đáp án nhưng ra **0 điểm**, không một cảnh báo nào.
  Đây chính là lỗi mà validator chặn bằng E010.

## Cách hoạt động

`run_scoring.py` **không sửa** `scoring.py`. Nó đọc nguyên văn file đó, `exec` trong một
namespace riêng với `__name__ = 'btc_scoring'` (để khối `if __name__ == "__main__"` không
chạy), rồi ghi đè 3 biến `reference_dir` / `prediction_dir` / `score_dir` sang thư mục mock
trước khi gọi `main()`.

Nhờ vậy kết quả local phản ánh đúng hành vi thật trên hệ thống — **kể cả khi `scoring.py`
crash**. Bài nộp có `"answer": null` chẳng hạn:

```
THAT BAI: TypeError: object of type 'NoneType' has no len()
Bai nop nay se lam chuong trinh cham diem loi tren he thong that.
```

## Tuỳ chọn

```
make_mock.py
  --train, -t   đường dẫn train.json (bắt buộc)
  --n, -n       số câu trích ra, mặc định 50, dùng 0 để lấy hết 7000
  --seed, -s    seed ngẫu nhiên, mặc định 42
  --out, -o     thư mục đầu ra, mặc định mock/
  --keep        giữ thư mục cũ thay vì xoá tạo lại

run_scoring.py
  --mock, -m        thư mục mock, mặc định mock/
  --submission, -s  file bài nộp (.json hoặc .zip)
  --all, -a         chấm hết mock/samples/ và in bảng so sánh
  --scoring         đường dẫn scoring.py, mặc định ../Scoring-Program-Task-LegalIR/scoring.py
```
