# Task B trên Nhotest theo Khangtest

Với thí nghiệm mới, `Khangtest2` bỏ khối bản dịch tiếng Anh khỏi prompt Task B;
`Khangtest3` giữ cả bản dịch và câu gốc. Phần huấn luyện bên dưới giữ nguyên.
Xem [TASK_B_TRANSLATION_EXPERIMENT.md](TASK_B_TRANSLATION_EXPERIMENT.md) để dùng
đúng preset và thư mục output/checkpoint của hai branch.

Task B giữ system prompt, class-weighted loss và soft ensemble đã đối chiếu với
`Khangtest` tại commit `e27514b`. Theo yêu cầu mới, Nhotest thêm lại bước chọn
best epoch trước khi retrain, giống hai giai đoạn của Task A:

1. Chạy từng seed `42, 52, 62, 72, 82` trên fold 1–4 (2.524 câu), dùng fold 5
   (631 câu original) để chọn epoch. Tối đa 5 epoch; early stopping khi 2 epoch
   liên tiếp không cải thiện. Macro-F1 cao hơn được ưu tiên; nếu bằng nhau thì
   lấy validation loss thấp hơn, nếu cả hai bằng nhau giữ epoch sớm hơn.
2. Hoàn tất chọn epoch của cả 5 seed, rồi khởi tạo lại backbone/LoRA và optimizer
   cho từng seed. Retrain đủ 3.155 câu của cả 5 fold theo best epoch riêng của seed
   đó. Không tiếp tục từ adapter holdout; không augmentation hay cap Gender.
3. Chấm điểm sáu nhãn bằng mean token log-likelihood, softmax trong mỗi seed,
   lấy trung bình xác suất của 5 model retrain rồi argmax. Số phiếu dùng để audit.

Qwen3-4B-Instruct-2507, LoRA rank 32/alpha 64, learning rate `1e-4` giữ nguyên.
Loss nhãn nhân class weight `clip((N / (6 * n_class)) ** 0.5, 0.75, 1.50)`,
rồi chuẩn hóa để trọng số trung bình của mẫu bằng 1. Class weight ở bước chọn
epoch chỉ tính trên fold 1–4; bước retrain tính lại trên cả 5 fold.
Task A giữ nguyên cấu hình và luồng huấn luyện hiện tại.

Các điều chỉnh để dùng dataset đã giao: `Comment` chứa tiếng Anh,
`OriginalComment` chứa câu gốc. Không yêu cầu đổi cột sang `Comment_en` như
Khangtest. Prompt Task B dùng các tiêu đề của Khangtest, cố định tiếng Anh trước câu gốc:

```text
English translation:
<Comment>

Comment (raw code-mixed):
<OriginalComment>
```

Giới hạn mặc định là 256 token, dùng revision Qwen3 đã pin. Nếu quá
dài, cả hai phiên bản được chia ngân sách token và giữ phần đầu/cuối. Nhãn
trả lời được giữ đầy đủ. Dữ liệu và nhãn trong CSV không bị sửa.

Không cần truyền cờ để đặt tiếng Anh trước. Cờ này vẫn được chấp nhận:

```python
!python Train_B_LLM.py --prompt-order english-first
```

`--prompt-order original-first` bị từ chối cho Task B để tránh đảo thứ tự ngoài ý muốn.
Thứ tự cố định áp dụng nhất quán khi train, validation và prediction.

## Chạy trên Kaggle

```python
%env HASTIKA_DATA_DIR=/kaggle/input/datasets/nnnho2605/hastika-final/hastika_bilingual_no_augments
!python Train_B_LLM.py
```

Loader đọc `original_1.csv` tới `original_5.csv` trong thư mục trên.
Inference và gold mặc định dùng
`data/open_validation/task_b_open_test_translated_labeled.csv` có sẵn trong repo.
Không cần file train gộp hay gắn thêm dataset. Code tự dùng hai GPU nếu có, như Task A.

## Nhãn validation để báo cáo

File open validation mặc định có đủ 395 câu gốc, bản dịch và nhãn. Code ghép
nhãn theo ID trong cùng file, báo loss, macro-F1,
precision/recall/F1 từng lớp và confusion matrix sau mỗi epoch full retrain và sau ensemble.
Các mẫu này không nằm trong train, không dùng để chọn epoch hay cập nhật model.
Gold có thể chứa câu code-mixed; prompt lấy đúng hai cột từ dataset song ngữ.
`--prediction-file` đổi sang test vẫn giữ nguồn input development để báo điểm.

File `multiclass_validation_inputs.csv` 396 câu raw-only trong ZIP bilingual cũ
không còn là đầu vào mặc định. Nếu cần dùng gold khác cùng các ID open validation:

```python
!python Train_B_LLM.py --epoch-validation-file /kaggle/input/your-gold-dataset/hastika_multiclass_validation.csv
```

Open validation đi kèm repo là bắt buộc ở cấu hình mặc định; nếu checkout thiếu
file này thì cần cập nhật hoặc clone lại branch. Nếu chỉ định gold bằng cờ trên,
file phải tồn tại và nhãn/ID phải hợp lệ. Không tự tạo nhãn hay dùng nhãn dự đoán
làm nhãn thật. Nhãn trong file inference không được đưa vào prompt.

## Checkpoint và kết quả

Thư mục task mới: `task_b_qwen3_4b_2507_lora_bilingual_selection_soft_english_first_256` dưới
`checkpoints/` và `outputs/`, tách biệt với run 512 hoặc câu gốc trước.
Mỗi epoch hoàn tất ở cả selection và full retrain lưu adapter, optimizer,
scheduler, GradScaler và RNG bằng cơ chế resume của Nhotest. Epoch bị timeout
giữa chừng được chạy lại; seed đã hoàn thành được tái sử dụng. Khi đổi input
prediction, adapter đã train đủ vẫn được dùng để dự đoán lại.

Giữ cả `outputs/` và `checkpoints/` trong Notebook Output/Dataset, rồi gắn
chúng bằng Add Input ở phiên sau. Code tự tìm dưới `/kaggle/input`, hoặc dùng
`--resume-from /kaggle/input/your-previous-output`. Cấu hình train và số GPU
phải khớp để resume. Checkpoint Task B train cố định trước đây không được dùng lại cho pipeline mới.

`split_manifest.csv` ghi rõ ID train/validation ở bước chọn epoch.
`run_summary.json` ghi best epoch và metrics holdout của từng seed.

Kết quả chính: `predictions.csv`, `task_b_submission.zip`,
`ensemble_predictions.csv` và `run_summary.json`. Mỗi seed lưu
`best_holdout_adapter`, `selection_summary.json`, `selection_training_history.json`,
`validation_predictions.csv`, `final_full_data_adapter`, `full_data_training_history.json`,
`task_b_summary.json`, `prediction_scores.npz`; nếu có gold thì thêm `official_validation/`.

Kiểm tra CPU và tokenizer không xác nhận chất lượng macro-F1 hay VRAM thực
tế của Qwen3-4B trên GPU. Cần chạy thí nghiệm trên Kaggle để đánh giá.
