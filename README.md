# Viper — HASTIKA @ ICON-2026

Repository fine-tune `google-bert/bert-base-uncased` độc lập cho hai subtask
HASTIKA. Mỗi script tạo đúng một stratified holdout 80/20, chỉ cập nhật trọng
số trên 80% train, chọn một checkpoint bằng Macro-F1 trên 20% validation và
dùng checkpoint đó dự đoán file input không nhãn của CodaBench.

- **Task A:** Binary Hate Speech Detection (`Hate`, `Non-Hate`).
- **Task B:** Fine-Grained Hate Speech Classification (`Gender`, `Political`,
  `Religion`, `Geo-political`, `Violence`, `Others`).
- **Loss:** CrossEntropyLoss thường cho cả hai task, không dùng class weights.
- **GPU:** tự động sử dụng Kaggle T4 x2 bằng PyTorch DDP.
- **Submission:** tạo `predictions.csv` và ZIP đúng định dạng CodaBench.

## 1. Cấu trúc repository

```text
.
├── Train_A.py                 # Chạy toàn bộ Task A
├── Train_B.py                 # Chạy toàn bộ Task B
├── config.py                  # Tất cả cấu hình và hyperparameter
├── requirements.txt
├── resources/
│   ├── kannada_stopwords.txt
│   └── romanized_kannada_stopwords.txt
├── src/
│   ├── data.py                # Đọc, kiểm tra và tokenize dữ liệu
│   ├── distributed.py         # Tự khởi chạy DDP trên hai GPU
│   ├── preprocessing.py       # Preprocessing gần nhất với bài báo
│   ├── submission.py          # Kiểm tra và đóng gói submission
│   └── training.py            # Holdout, training và chọn checkpoint
└── tests/                     # Unit test và CPU/DDP smoke test
```

Các thư mục sau chỉ tồn tại ở local/Kaggle và đã bị `.gitignore` chặn:

```text
data/          # Dataset cuộc thi; tuyệt đối không push lên GitHub
checkpoints/   # Một checkpoint tốt nhất cho mỗi task
outputs/       # Submission, validation diagnostics và run metadata
```

## 2. Chuẩn bị Private Dataset trên Kaggle

Không đưa CSV của ban tổ chức lên GitHub. Trên Kaggle:

1. Chọn **Create → New Dataset** và upload:
   - `binary_train.csv`
   - `binary_validation_inputs.csv`
   - `multiclass_train.csv`
   - `multiclass_validation_inputs.csv`
2. Giữ dataset ở chế độ **Private**.
3. Trong **Settings → Sharing**, thêm tài khoản Kaggle của các thành viên Viper.
4. Trong Notebook, chọn **Add Input → Your Datasets** và gắn dataset này.

Code tự tìm các file trong `/kaggle/input`. Nếu có nhiều bản trùng tên, chỉ
định thư mục chính xác:

```python
import os

os.environ["HASTIKA_DATA_DIR"] = "/kaggle/input/hastika-viper-private"
```

## 3. Cấu hình Kaggle Notebook

Trong **Notebook options**:

- Accelerator: **GPU T4 x2**.
- Internet: **On** để tải BERT, NLTK resources và clone GitHub.
- Giữ Notebook **Private** trong thời gian thi.

```bash
!git clone --branch Khangtest --single-branch https://github.com/POG42069/Hastika_Icon_2026_Viper.git
%cd Hastika_Icon_2026_Viper
!pip install -q -r requirements.txt
```

NLTK tự kiểm tra và tải `stopwords`, `wordnet`, `omw-1.4` và
`averaged_perceptron_tagger_eng` ở lần chạy đầu. Script dừng với thông báo rõ
ràng nếu resource thiếu; preprocessing không bao giờ bị âm thầm bỏ qua.

## 4. Pipeline huấn luyện

Task A và Task B chạy độc lập nhưng dùng cùng logic:

```text
Labeled train CSV
        ↓
Paper-style preprocessing
        ↓
Stratified split một lần, seed 42
        ↓
80% train ── backpropagation/cập nhật BERT
20% validation ── chỉ eval, chọn checkpoint theo Macro-F1
        ↓
Load checkpoint tốt nhất
        ↓
Dự đoán competition input không nhãn
        ↓
predictions.csv + submission ZIP
```

Kích thước holdout mặc định:

| Task | Tổng labeled | Train 80% | Validation 20% |
|---|---:|---:|---:|
| A | 6.446 | 5.156 | 1.290 |
| B | 3.159 | 2.527 | 632 |

Không có K-fold, ensemble, OOF hoặc retrain trên 100% labeled data.

## 5. Chạy Task A

```bash
!python Train_A.py
```

Đầu ra:

```text
checkpoints/task_a/best_model.pt
checkpoints/task_a/metrics.json
outputs/task_a/predictions.csv
outputs/task_a/task_a_submission.zip
outputs/task_a/validation_predictions.csv
outputs/task_a/split_manifest.csv
outputs/task_a/run_summary.json
```

Nộp `outputs/task_a/task_a_submission.zip` vào track Task A.

## 6. Chạy Task B

```bash
!python Train_B.py
```

Đầu ra tương tự trong `checkpoints/task_b/` và `outputs/task_b/`. Nộp
`outputs/task_b/task_b_submission.zip` vào track Task B. Task B không sử dụng
dự đoán của Task A làm đầu vào.

## 7. Định dạng submission

Mỗi ZIP chỉ chứa một file `predictions.csv` ở thư mục gốc:

```csv
id,label
4186,Non-Hate
5693,Hate
```

Task A chỉ xuất `Hate` hoặc `Non-Hate`. Task B chỉ xuất `Gender`, `Political`,
`Religion`, `Geo-political`, `Violence` hoặc `Others`. Code kiểm tra header,
thứ tự ID, số dòng, nhãn và nội dung ZIP trước khi hoàn tất.

## 8. Hyperparameter

Tất cả cấu hình nằm trong `config.py`:

| Cấu hình | Mặc định | Ý nghĩa |
|---|---:|---|
| `model_name` | `google-bert/bert-base-uncased` | Backbone cho cả hai task |
| `validation_size` | `0.20` | Tỷ lệ holdout có nhãn |
| `max_epochs` | 5 | Epoch tối đa |
| `learning_rate` | `2e-5` | Learning rate AdamW |
| `train_batch_size_per_gpu` | 8 | Batch riêng của từng GPU |
| `eval_batch_size_per_gpu` | 16 | Batch validation/prediction mỗi GPU |
| `gradient_accumulation_steps` | 1 | Số bước cộng dồn gradient |
| `max_length` | 128 | Độ dài token tối đa |
| `weight_decay` | 0.01 | AdamW weight decay |
| `warmup_ratio` | 0.10 | Tỷ lệ warm-up |
| `early_stopping_patience` | 2 | Epoch không cải thiện trước khi dừng |
| `seed` | 42 | Seed của split và training |

Với T4 x2, effective batch mặc định là:

```text
8 mẫu/GPU × 2 GPU × 1 gradient accumulation = 16 mẫu/update
```

## 9. Preprocessing

Các bước chạy theo thứ tự mô tả trong bài báo HASTIKA:

1. Sửa Unicode, giải mã HTML entity và lowercase.
2. Xóa HTML, URL, mention, toàn bộ hashtag, emoji, punctuation và symbol.
3. Tách Unicode word token, giữ chữ Kannada/English và số.
4. Loại English, Kannada và romanized Kannada stopwords.
5. Giữ từ ngữ cảnh tác giả nêu và các từ phủ định quan trọng.
6. POS tagging và WordNet lemmatization cho token English.
7. Chuẩn hóa ký tự và từ lặp.

Bài báo không công bố danh sách stopword Kannada/romanized Kannada hoặc công
cụ lemmatization Kannada. Vì vậy hai danh sách trong `resources/` là bản xấp xỉ
được version hóa của Viper; Kannada và romanized Kannada không bị lemmatize.
Đây là bản tái hiện gần nhất có thể kiểm chứng, không được tuyên bố là code gốc
của tác giả.

## 10. Chạy kiểm thử

Unit test và CPU smoke test:

```bash
python -m unittest discover -s tests -v
```

DDP smoke test hai tiến trình CPU:

```bash
python tests/run_ddp_smoke.py
```

Các test không tải hoặc fine-tune BERT; preprocessing test chỉ tải NLTK data
nhỏ nếu máy chưa có.

## 11. Xử lý lỗi thường gặp

### Không tìm thấy CSV

Kiểm tra private dataset đã được gắn bằng **Add Input**. Nếu nhiều file trùng
tên, đặt `HASTIKA_DATA_DIR` như mục 2.

### Không tải được BERT hoặc NLTK

Bật Internet ở lần chạy đầu. Script không tiếp tục với preprocessing thiếu
resource vì điều đó làm kết quả giữa các lần chạy không nhất quán.

### CUDA out of memory

Giảm `train_batch_size_per_gpu` từ 8 xuống 4 và có thể tăng
`gradient_accumulation_steps` lên 2 để giữ effective batch 16.

### Chỉ phát hiện một GPU

Kiểm tra Accelerator là **GPU T4 x2**. Khi thành công, log hiển thị
`world_size=2`; người dùng không phải tự gọi `torchrun`.
