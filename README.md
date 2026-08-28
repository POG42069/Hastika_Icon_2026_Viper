# Viper — HASTIKA @ ICON-2026

Repository huấn luyện hai subtask HASTIKA bằng `google/muril-base-cased`,
Stratified 5-fold cross-validation và ensemble xác suất của năm checkpoint.

- **Task A:** Binary Hate Speech Detection (`Hate`, `Non-Hate`).
- **Task B:** Fine-Grained Hate Speech Classification (`Gender`, `Political`,
  `Religion`, `Geo-political`, `Violence`, `Others`).
- **Metric chọn checkpoint:** validation Macro-F1.
- **GPU:** tự động sử dụng cả hai GPU của Kaggle T4 x2 bằng PyTorch DDP.
- **Submission:** tự tạo `predictions.csv` và file ZIP đúng định dạng CodaBench.

## 1. Cấu trúc repository

```text
.
├── Train_A.py                 # Chạy toàn bộ pipeline Task A
├── Train_B.py                 # Chạy toàn bộ pipeline Task B
├── config.py                  # Tất cả đường dẫn và hyperparameter
├── requirements.txt
├── src/
│   ├── data.py                # Đọc, kiểm tra và tiền xử lý dữ liệu
│   ├── distributed.py         # Tự khởi chạy DDP trên hai GPU
│   ├── preprocessing.py       # Chuẩn hóa văn bản code-mixed
│   ├── submission.py          # Kiểm tra và đóng gói submission
│   └── training.py            # 5-fold, early stopping, ensemble
└── tests/
    └── test_pipeline_utils.py
```

Các thư mục dưới đây chỉ được tạo ở local/Kaggle và đã bị `.gitignore` chặn:

```text
data/          # Dataset của ban tổ chức; tuyệt đối không push lên GitHub
checkpoints/   # Năm checkpoint tốt nhất của mỗi task
outputs/       # predictions.csv, ZIP, OOF và báo cáo metric
```

## 2. Chuẩn bị Private Dataset trên Kaggle

Không đưa các CSV của ban tổ chức lên GitHub. Thay vào đó:

1. Đăng nhập Kaggle, chọn **Create → New Dataset**.
2. Upload đúng bốn file:
   - `binary_train.csv`
   - `binary_validation_inputs.csv`
   - `multiclass_train.csv`
   - `multiclass_validation_inputs.csv`
3. Giữ dataset ở chế độ **Private**.
4. Vào **Settings → Sharing** và thêm các tài khoản Kaggle của team Viper.
5. Trong Kaggle Notebook, chọn **Add Input → Your Datasets** và gắn private
   dataset vừa tạo.

Code tự tìm bốn file trong `/kaggle/input`. Nếu notebook gắn nhiều dataset có
cùng tên file, chỉ định rõ thư mục trước khi chạy:

```python
import os

os.environ["HASTIKA_DATA_DIR"] = "/kaggle/input/hastika-viper-private"
```

## 3. Cấu hình Kaggle Notebook

Trong phần **Notebook options**:

- Accelerator: **GPU T4 x2**.
- Internet: **On** ở lần đầu để tải MuRIL từ Hugging Face và clone GitHub.
- Giữ Notebook ở chế độ **Private** trong thời gian thi.

Clone đúng nhánh làm việc:

```bash
!git clone --branch Khangtest --single-branch https://github.com/POG42069/Hastika_Icon_2026_Viper.git
%cd Hastika_Icon_2026_Viper
!pip install -q -r requirements.txt
```

Kaggle đã cài sẵn PyTorch CUDA. Nếu `pip` thông báo phiên bản PyTorch hiện có
đã thỏa mãn yêu cầu thì không cần cài lại kernel CUDA.

## 4. Chạy Task A

Chỉ cần một lệnh:

```bash
!python Train_A.py
```

Script tự thực hiện:

1. Đọc `binary_train.csv` và `binary_validation_inputs.csv`.
2. Tiền xử lý văn bản và tokenize bằng tokenizer chính thức của MuRIL.
3. Chia Stratified 5-fold.
4. Fine-tune tối đa năm epoch cho mỗi fold.
5. Lưu checkpoint có validation Macro-F1 tốt nhất của từng fold.
6. Lấy trung bình xác suất của năm checkpoint trên tập cần dự đoán.
7. Xuất:

```text
outputs/task_a/predictions.csv
outputs/task_a/task_a_submission.zip
outputs/task_a/oof_predictions.csv
outputs/task_a/run_summary.json
```

File cần nộp lên track Task A là `task_a_submission.zip`.

## 5. Chạy Task B

Chỉ cần một lệnh:

```bash
!python Train_B.py
```

Task B hoạt động độc lập với Task A. Mỗi fold tính class weight chỉ trên phần
train của fold theo công thức `N / (K × N_c)` rồi dùng weighted cross-entropy.
Task B không sử dụng dự đoán Hate/Non-Hate của Task A làm đầu vào.

Các file đầu ra:

```text
outputs/task_b/predictions.csv
outputs/task_b/task_b_submission.zip
outputs/task_b/oof_predictions.csv
outputs/task_b/run_summary.json
```

File cần nộp lên track Task B là `task_b_submission.zip`.

## 6. Định dạng submission

Mỗi task tạo một ZIP riêng. Trong ZIP chỉ có đúng một file mang tên
`predictions.csv` với hai cột:

```csv
id,label
4186,Non-Hate
5693,Hate
```

Task A chỉ xuất `Hate` hoặc `Non-Hate`. Task B chỉ xuất một trong sáu nhãn:
`Gender`, `Political`, `Religion`, `Geo-political`, `Violence`, `Others`.
Mã nguồn kiểm tra header, thứ tự ID, số dòng, nhãn hợp lệ và nội dung ZIP trước
khi hoàn tất.

## 7. Điều chỉnh hyperparameter

Toàn bộ cấu hình nằm trong [`config.py`](config.py), chủ yếu ở `TrainingConfig`:

| Cấu hình | Mặc định | Ý nghĩa |
|---|---:|---|
| `num_folds` | 5 | Số fold cross-validation |
| `max_epochs` | 5 | Số epoch tối đa trên mỗi fold |
| `learning_rate` | `2e-5` | Learning rate của AdamW |
| `train_batch_size_per_gpu` | 8 | Batch riêng của từng GPU |
| `eval_batch_size_per_gpu` | 16 | Batch validation của từng GPU |
| `gradient_accumulation_steps` | 1 | Số bước cộng dồn gradient |
| `max_length` | 128 | Độ dài token tối đa |
| `weight_decay` | 0.01 | AdamW weight decay |
| `warmup_ratio` | 0.10 | Tỷ lệ bước warm-up |
| `early_stopping_patience` | 2 | Dừng sau số epoch không cải thiện |

Với Kaggle T4 x2 và giá trị mặc định, effective training batch size là:

```text
8 mẫu/GPU × 2 GPU × 1 gradient accumulation = 16 mẫu/update
```

Khi chạy `python Train_A.py` hoặc `python Train_B.py`, script phát hiện hai GPU
và tự relaunch bằng `torchrun`; người dùng không phải tự viết lệnh DDP.

## 8. Tiền xử lý

Pipeline mặc định:

- sửa Unicode/encoding và giải mã HTML entity;
- thay `<br>`/HTML tag bằng khoảng trắng;
- thay URL bằng token `URL`, mention bằng `USER`;
- bỏ dấu `#` nhưng giữ nội dung hashtag;
- rút chuỗi ký tự lặp từ ba lần trở lên còn hai lần;
- chuẩn hóa khoảng trắng;
- giữ chữ hoa/thường vì đang dùng `muril-base-cased`;
- giữ emoji, dấu câu, stopword và dạng biến thể của từ để tránh mất ngữ cảnh.

Các lựa chọn nằm trong `PreprocessConfig` của `config.py`.

## 9. Chạy kiểm thử nhanh

Kiểm thử không tải model và không huấn luyện:

```bash
python -m unittest discover -s tests -v
```

## 10. Xử lý lỗi thường gặp

### Không tìm thấy CSV

Kiểm tra private dataset đã được gắn vào Notebook bằng **Add Input**. Nếu có
nhiều file trùng tên, đặt biến `HASTIKA_DATA_DIR` như mục 2.

### CUDA out of memory

Giảm `train_batch_size_per_gpu` từ 8 xuống 4 trong `config.py`. Có thể tăng
`gradient_accumulation_steps` lên 2 để giữ effective batch size bằng 16.

### Không tải được MuRIL

Bật Internet cho Notebook ở lần chạy đầu. Các lần sau Hugging Face cache model
trong phiên làm việc hiện tại.

### Chỉ phát hiện một GPU

Kiểm tra Accelerator của Kaggle thật sự đang là **GPU T4 x2**. Script in
`world_size=2` khi DDP đã khởi chạy thành công.
