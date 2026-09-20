# HASTIKA @ ICON-2026 — LLM LoRA

Pipeline fine-tune causal LLM cho hai task HASTIKA:

- Task A: phát hiện `Non-Hate` / `Hate`.
- Task B: phân loại `Gender`, `Political`, `Religion`, `Geo-political`,
  `Violence`, `Others`.

Pipeline dùng **LoRA chuẩn, không lượng tử hóa**. Backbone được nạp bằng
FP16 trên T4 hoặc BF16 trên GPU hỗ trợ native BF16; trọng số backbone đóng
băng và chỉ các ma trận LoRA được optimizer cập nhật.

## Quy trình huấn luyện

Mỗi entry point tự chạy hai giai đoạn. Cách tạo holdout phụ thuộc vào task:

1. Cả hai task dùng predefined folds để giữ câu nguồn và augment cùng một fold.
   Task A dùng `binary_original_1..4 + binary_augment1..4` để train và chỉ
   `binary_original_5` để validation. Task B dùng
   `original_1..4 + augment1..4` để train và `original_5` để validation.
2. Task A và Task B đều chạy 5 seed `42, 52, 62, 72, 82`. Mỗi seed train LoRA
   và tự chọn `best_epoch` trên cùng một validation cố định. Macro-F1 cao hơn
   thắng; validation loss thấp hơn dùng để phá hòa. Với Task B, cùng quy tắc
   này còn được dùng để chọn một seed chiến thắng. Task B lấy ngẫu nhiên một
   tập Gender khác nhau xuống tối đa 870 câu cho mỗi seed.
3. Với Task A, cả 5 seed đều khởi tạo lại backbone và LoRA từ đầu, rồi retrain
   trên `binary_original_1..5 + binary_augment1..5` theo `best_epoch` riêng.
   Năm adapter dự đoán development độc lập; nhãn cuối là hard majority vote,
   tức `Hate` khi có ít nhất 3 phiếu, ngược lại là `Non-Hate`.
4. Task B vẫn chọn seed tốt nhất, khởi tạo lại từ đầu và retrain trên
   `original_1..5 + augment1..5`, ngoại trừ các câu Gender bị seed chiến thắng
   loại khỏi fold 1-4.
5. Adapter full-retrain dùng để dự đoán development inputs và tạo
   `predictions.csv` cùng file ZIP nộp CodaBench. Full-retrain không đánh giá
   lại fold 5 vì fold này đã được đưa vào dữ liệu huấn luyện đầy đủ.

Việc khởi tạo lại ở bước full-retrain giúp tất cả mẫu được học số epoch như nhau. Nếu
tiếp tục trực tiếp từ adapter 80%, nhóm 80% ban đầu sẽ được học nhiều lần hơn
nhóm validation cũ.

## Light text normalization

Pipeline chỉ sửa nhiễu kỹ thuật trước khi đưa cột `Comment` vào chat tokenizer:

- Sửa lỗi encoding bằng `ftfy`, chuẩn hóa Unicode NFKC và giải mã HTML entity.
- Thu gọn khoảng trắng; ô `Comment` thiếu được chuyển thành chuỗi rỗng.
- Giữ nguyên URL, mention và thẻ HTML.
- Không lowercase; giữ hashtag, emoji và dấu câu.
- Không xóa stopword, không lemmatize và không chuẩn hóa ký tự/từ lặp.

Khi comment quá dài, `ChatEncoder` chỉ cắt phần cuối comment để toàn bộ system
prompt và assistant label vẫn nằm trong `max_length`.

## LoRA

Cấu hình mặc định trong `llm_config.py`:

| Tham số | Giá trị |
|---|---:|
| Backbone | `Qwen/Qwen2.5-3B-Instruct` |
| LoRA rank `r` | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Target modules | `all-linear` |
| Max epochs để chọn | 5 |
| Early-stopping patience | 2 |
| Max chat length | 256 |
| Learning rate | `1e-4` |
| Batch/GPU | 4 |
| Gradient accumulation | 2 |
| Eval batch/GPU | 2 candidate completions |
| Progress log | Một dòng cố định mỗi 10% |

LoRA áp dụng:

```text
W = W0 + (alpha / r) BA
```

Với `r=32`, `alpha=64`, hệ số scale là 2. Backbone `W0` không được optimizer
cập nhật.

Đây không còn là QLoRA: code không dùng NF4, double quantization hay
`BitsAndBytesConfig`. Vì vậy standard LoRA dùng VRAM nhiều hơn đáng kể.
Qwen2.5-3B phù hợp hơn với T4 16 GB; Qwen2.5-7B FP16 thường không còn đủ
khoảng trống để train LoRA trên một T4, và DDP không cộng VRAM của hai GPU.

## Dữ liệu

Mặc định pipeline tìm các file sau trong `HASTIKA_DATA_DIR`, thư mục `data/`,
hoặc một Kaggle Dataset dưới `/kaggle/input`.

Task A:

```text
binary_original_1.csv ... binary_original_5.csv
binary_augment1.csv ... binary_augment5.csv
binary_validation_inputs.csv
```

Task B:

```text
original_1.csv ... original_5.csv
augment1.csv ... augment5.csv
multiclass_validation_inputs.csv
```

Train CSV cần các cột `id`, `Comment` và cột nhãn tương ứng. Development CSV
cần `id`, `Comment`. Các file `augment*.csv` cần thêm `source_id`,
`source_group` và `fold`. Loader kiểm tra nguồn tồn tại, nguồn nằm cùng fold,
nhãn augment khớp nhãn nguồn và một `source_group` không vượt qua nhiều fold.

Loader còn tạo `augmentation_audit.csv` để đánh dấu câu quá ngắn, trùng lặp,
quá khác câu nguồn hoặc gần giống một original khác. Cấu hình hiện tại dùng
`augmentation_audit_action="report"`, nên audit chỉ báo cáo; các câu bị đánh
dấu vẫn được giữ trong dữ liệu huấn luyện.

Tạo lại dữ liệu Task A cân bằng từ bộ dữ liệu đã làm sạch:

```bash
python scripts/build_task_a_balanced_data.py \
  --input-dir ../hastika_regenerated_augments \
  --output-dir ../task_a_balanced
```

Script không sửa dữ liệu nguồn. Nó audit Hate augment, khôi phục các câu nguồn
Task B còn thiếu trong binary data, gán fold có seed cố định, sinh Non-Hate
variants bảo thủ, tạo hàng chờ human review và xuất các fold Task A. Đặt
`HASTIKA_DATA_DIR` trỏ tới thư mục output trước khi train.

Chạy sanity check nhẹ trước khi dùng GPU:

```bash
python scripts/evaluate_task_a_data_sanity.py --data-dir ../task_a_balanced
```

Đây chỉ là kiểm tra hồi quy dữ liệu bằng character n-gram Naive Bayes. Quyết
định cuối cùng vẫn dựa trên LoRA experiment với original-only fold 5.

## Chạy trên Kaggle

```bash
!pip install -q -r requirements.txt
!python Train_A_LLM.py
!python Train_B_LLM.py
```

Chạy tuần tự Task A và Task B. Khi Kaggle có T4 x2, entry point tự relaunch
bằng DDP với hai worker; mỗi GPU giữ một bản backbone đầy đủ.

Đổi backbone hoặc giới hạn tài nguyên:

```bash
!python Train_A_LLM.py --model Qwen/Qwen2.5-3B-Instruct
!python Train_A_LLM.py --max-epochs 3 --batch-size 1 --max-length 256
!python Train_B_LLM.py --single-gpu --gradient-accumulation 16
!python Train_A_LLM.py --prediction-file /path/to/official_inputs.csv
```

Có thể đặt thư mục bằng biến môi trường trước khi chạy:

```bash
%env HASTIKA_DATA_DIR=/kaggle/input/hastika-data
%env HASTIKA_OUTPUT_DIR=/kaggle/working/outputs
%env HASTIKA_CHECKPOINT_DIR=/kaggle/working/checkpoints
```

## Tokenization, loss và dự đoán

Task A dùng trực tiếp nhãn chính thức trong assistant response:

```text
System: Classify ... Allowed labels: Non-Hate; Hate.
User: Comment:\n<normalized Comment>
Assistant: Non-Hate hoặc Hate
```

Task B vẫn dùng các mã nội bộ `A`–`F` để giữ các candidate completion ngắn.

Code gọi `apply_chat_template` của tokenizer. Prompt và padding được mask bằng
`-100`; loss chỉ áp dụng cho các token của assistant label và phần kết thúc
chat. Khi dự đoán, pipeline tính mean token log-likelihood cho từng câu trả lời
hợp lệ và chọn nhãn có điểm lớn nhất; phép chuẩn hóa này tránh thiên vị câu trả
lời ngắn khi Task A dùng trực tiếp tên nhãn. Pipeline không sinh văn bản tự do
và không regex sửa kết quả.

Tiến trình train, validation và prediction được ghi thành dòng cố định mỗi 10%
với số batch/candidate, elapsed time, ETA, tốc độ và mean loss khi train. Cách
này tránh progress bar động bị Kaggle nối thành một khối log dài khó đọc.
Sau mỗi epoch, cả hai task đều ghi F1 của từng nhãn. Task A còn ghi precision,
recall, support và số lỗi theo cả hai chiều: các nhãn thật bị dự đoán nhầm vào
nhãn đang xét và các mẫu thật của nhãn đang xét bị dự đoán sang từng nhãn khác.
Task B hiện không in các thống kê chi tiết này vì
`report_per_class_metrics=False`. Ma trận nhầm lẫn của cả hai task vẫn được lưu
trong lịch sử huấn luyện, CSV và metrics JSON.

## Artifacts

Task A:

```text
checkpoints/task_a_lora/seed_42/best_holdout_adapter/
checkpoints/task_a_lora/seed_42/final_full_data_adapter/
checkpoints/task_a_lora/seed_52/...
checkpoints/task_a_lora/seed_62/...
checkpoints/task_a_lora/seed_72/...
checkpoints/task_a_lora/seed_82/...
checkpoints/task_a_lora/metrics.json
outputs/task_a_lora/seed_42/validation_predictions.csv
outputs/task_a_lora/seed_42/predictions.csv
outputs/task_a_lora/seed_52/...
outputs/task_a_lora/selection_training_history.json
outputs/task_a_lora/split_manifest.csv
outputs/task_a_lora/validation_predictions.csv
outputs/task_a_lora/seed_selection_results.csv
outputs/task_a_lora/voting_predictions.csv
outputs/task_a_lora/predictions.csv
outputs/task_a_lora/task_a_submission.zip
outputs/task_a_lora/run_summary.json
```

`validation_predictions.csv` ở thư mục gốc là kết quả của seed có holdout tốt
nhất và chỉ dùng để báo cáo. `voting_predictions.csv` lưu dự đoán của cả 5 seed,
số phiếu từng nhãn và nhãn cuối. `predictions.csv` cùng ZIP ở thư mục gốc là
kết quả majority vote; prediction của từng adapter nằm trong thư mục seed.

Task B còn tạo một thư mục riêng cho từng seed, ví dụ
`outputs/task_b_lora/seed_42/`, cùng `seed_selection_results.csv` và
`seed_selection_summary.json`. Các artifact ở trực tiếp
`outputs/task_b_lora/` luôn tương ứng với seed chiến thắng.

Với Task A, Macro-F1 holdout chỉ chọn `best_epoch` riêng cho từng seed; cả 5
seed đều được full-retrain và tham gia majority vote. Seed có holdout tốt nhất
chỉ đại diện cho báo cáo validation. Với Task B, Macro-F1 holdout vừa chọn
`best_epoch` trong từng seed, vừa chọn một winning seed duy nhất để full-retrain
và dự đoán development. File tổng hợp lưu cả điểm trung bình và độ lệch chuẩn;
test/development chính thức vẫn là đánh giá cuối cùng.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
python tests/run_llm_ddp_smoke.py
```

Smoke test CUDA riêng:

```bash
python tests/run_llm_cuda_smoke.py
```
