# HASTIKA @ ICON-2026 — LLM LoRA

Pipeline fine-tune causal LLM cho hai task HASTIKA:

- Task A: phát hiện `Non-Hate` / `Hate`.
- Task B: phân loại `Gender`, `Political`, `Religion`, `Geo-political`,
  `Violence`, `Others`.

Pipeline dùng **LoRA chuẩn, không lượng tử hóa**. Backbone được nạp bằng
FP16 trên T4 hoặc BF16 trên GPU hỗ trợ native BF16; trọng số backbone đóng
băng và chỉ các ma trận LoRA được optimizer cập nhật.

## Quy trình huấn luyện

Task A giữ pipeline hai giai đoạn: chia stratified 80/20 với seed 42, chọn
checkpoint có Macro-F1 validation tốt nhất, rồi khởi tạo lại Qwen + LoRA và
retrain trên toàn bộ dữ liệu gốc đúng số epoch đã chọn.

Task B tạm thời bỏ toàn bộ selection training và validation:

1. Nạp `original_1..5 + augment1..5` theo đúng thứ tự đã cấu hình.
2. Dùng seed `52` trên pool Gender của fold 1-4 để tái tạo đúng danh sách 219
   câu bị loại, giữ tối đa 870 câu Gender trong pool này.
3. Khởi tạo Qwen + LoRA đúng một lần và train trực tiếp **3 epoch** trên toàn bộ
   3.930 dòng được giữ lại, bao gồm fold 5 và augment 5.
4. Dùng adapter đó dự đoán development inputs và tạo `predictions.csv` cùng
   file ZIP nộp CodaBench.

Task B không tính validation F1, không early stopping và không tự chọn lại số
epoch. Cách này cố định đúng ba epoch từng được dùng cho submission tốt nhất.

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
| Max epochs để chọn Task A | 5 |
| Epoch cố định Task B | 3 |
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
binary_train.csv
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

Mỗi mẫu có dạng:

```text
System: Classify ... Reply with exactly one allowed label.
User: Comment:\n<normalized Comment>
Assistant: <official label>
```

Code gọi `apply_chat_template` của tokenizer. Prompt và padding được mask bằng
`-100`; loss chỉ áp dụng cho các token của assistant label và phần kết thúc
chat. Khi dự đoán, pipeline tính tổng log-likelihood cho từng nhãn hợp lệ và
chọn nhãn có điểm lớn nhất; không sinh văn bản tự do và không regex sửa kết quả.

Tiến trình train, validation và prediction được ghi thành dòng cố định mỗi 10%
với số batch/candidate, elapsed time, ETA, tốc độ và mean loss khi train. Cách
này tránh progress bar động bị Kaggle nối thành một khối log dài khó đọc.
Task B hiện không chạy validation nên log chỉ ghi train loss của ba epoch và
tiến trình dự đoán development.

## Artifacts

Ví dụ Task A:

```text
checkpoints/task_a_lora/best_holdout_adapter/
checkpoints/task_a_lora/final_full_data_adapter/
checkpoints/task_a_lora/metrics.json
outputs/task_a_lora/selection_training_history.json
outputs/task_a_lora/full_data_training_history.json
outputs/task_a_lora/split_manifest.csv
outputs/task_a_lora/validation_predictions.csv
outputs/task_a_lora/predictions.csv
outputs/task_a_lora/task_a_submission.zip
outputs/task_a_lora/run_summary.json
```

`validation_predictions.csv` được tạo bằng best holdout adapter để phân tích.
`predictions.csv` và ZIP luôn được tạo bằng `final_full_data_adapter`.

Task B tạo trực tiếp:

```text
checkpoints/task_b_lora/final_full_data_adapter/
checkpoints/task_b_lora/metrics.json
outputs/task_b_lora/full_data_training_history.json
outputs/task_b_lora/full_training_manifest.csv
outputs/task_b_lora/downsampled_rows.csv
outputs/task_b_lora/predictions.csv
outputs/task_b_lora/task_b_submission.zip
outputs/task_b_lora/run_summary.json
```

Task B không tạo checkpoint holdout, validation predictions hoặc artifact chọn
seed/epoch trong chế độ direct full-data hiện tại.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
python tests/run_llm_ddp_smoke.py
```

Smoke test CUDA riêng:

```bash
python tests/run_llm_cuda_smoke.py
```
