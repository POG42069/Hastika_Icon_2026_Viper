# Task B trên Nhotest theo Khangtest

Đối chiếu nhánh `Khangtest` tại commit `e27514b`. Cấu hình đang hoạt động trên
nhánh đó bỏ OOF và train trực tiếp toàn bộ dữ liệu, 5 epoch cho mỗi seed.

- Qwen3-4B-Instruct-2507, 5 seed `42, 52, 62, 72, 82`, LoRA rank 32/alpha 64.
- Train đủ 3.155 câu original của 5 fold, không augmentation, không cap Gender.
- Learning rate `1e-4`. Loss trên nhãn được nhân class weight:
  `clip((N / (6 * n_class)) ** 0.5, 0.75, 1.50)`, rồi chuẩn hóa để trọng số
  trung bình của các mẫu bằng 1. Giới hạn clip là trước bước chuẩn hóa.
- Mỗi seed kết thúc đúng epoch 5; không early stopping và không chọn best epoch.
- Chấm điểm sáu nhãn bằng mean token log-likelihood, softmax trong mỗi seed,
  lấy trung bình xác suất của 5 seed rồi argmax. Số phiếu chỉ dùng để audit.
- Task A giữ nguyên cấu hình và luồng huấn luyện hiện tại của Nhotest.

Các điều chỉnh để dùng dataset đã giao: `Comment` chứa tiếng Anh,
`OriginalComment` chứa câu gốc. Không yêu cầu đổi cột sang `Comment_en` như
Khangtest. Prompt Task B giữ cách trình bày của Khangtest:

```text
Comment (raw code-mixed):
<OriginalComment>

English translation:
<Comment>
```

Giữ giới hạn 512 token và revision Qwen3 đã pin theo yêu cầu trước. Nếu quá
dài, cả hai phiên bản được chia ngân sách token và giữ phần đầu/cuối. Nhãn
trả lời được giữ đầy đủ. Dữ liệu và nhãn trong CSV không bị sửa.

## Chạy trên Kaggle

```python
%env HASTIKA_DATA_DIR=/kaggle/input/datasets/nnnho2605/hastika-final/hastika_bilingual_no_augments
!python Train_B_LLM.py
```

Loader đọc `original_1.csv` tới `original_5.csv` và
`multiclass_validation_inputs.csv` trực tiếp trong thư mục trên. Không cần
file train gộp. Code tự dùng hai GPU nếu có, như Task A.

## Nhãn validation để báo cáo

Nếu có `hastika_multiclass_validation.csv` trong thư mục dữ liệu, code ghép
nhãn theo ID với `multiclass_validation_inputs.csv`, báo loss, macro-F1,
precision/recall/F1 từng lớp và confusion matrix sau mỗi epoch và sau ensemble.
Các mẫu này không nằm trong train, không dùng để chọn epoch hay cập nhật model.
Gold có thể chứa câu code-mixed; prompt lấy đúng hai cột từ dataset song ngữ.
`--prediction-file` đổi sang test vẫn giữ nguồn input development để báo điểm.

ZIP song ngữ đã giao không có file gold này. Nếu gold nằm ở dataset riêng:

```python
!python Train_B_LLM.py --epoch-validation-file /kaggle/input/your-gold-dataset/hastika_multiclass_validation.csv
```

Khi không có gold mặc định, code báo rõ và vẫn train/xuất prediction; không
tính metric validation. Nếu chỉ định file bằng cờ trên thì file phải tồn tại
và nhãn/ID phải hợp lệ. Không tự tạo nhãn hay dùng nhãn dự đoán làm nhãn thật.

## Checkpoint và kết quả

Thư mục task mới: `task_b_qwen3_4b_2507_lora_bilingual_fixed_soft` dưới
`checkpoints/` và `outputs/`. Mỗi epoch hoàn tất lưu adapter, optimizer,
scheduler, GradScaler và RNG bằng cơ chế resume của Nhotest. Epoch bị timeout
giữa chừng được chạy lại; seed đã hoàn thành được tái sử dụng. Khi đổi input
prediction, adapter đã train đủ vẫn được dùng để dự đoán lại.

Giữ cả `outputs/` và `checkpoints/` trong Notebook Output/Dataset, rồi gắn
chúng bằng Add Input ở phiên sau. Code tự tìm dưới `/kaggle/input`, hoặc dùng
`--resume-from /kaggle/input/your-previous-output`. Cấu hình train và số GPU
phải khớp để resume. Checkpoint Task B theo pipeline cũ không được dùng lại.

Kết quả chính: `predictions.csv`, `task_b_submission.zip`,
`ensemble_predictions.csv` và `run_summary.json`. Mỗi seed lưu
`final_full_data_adapter`, `full_data_training_history.json`,
`prediction_scores.npz`; nếu có gold thì thêm `official_validation/`.

Kiểm tra CPU và tokenizer không xác nhận chất lượng macro-F1 hay VRAM thực
tế của Qwen3-4B trên GPU. Cần chạy thí nghiệm trên Kaggle để đánh giá.
