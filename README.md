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

1. Task A chia stratified 80% train và 20% validation với seed 42. Task B dùng
   `original_1..4 + augment1..4` để train và `original_5` để validation.
2. Task B chạy độc lập với 5 seed `42, 52, 62, 72, 82`. Mỗi seed lấy ngẫu
   nhiên một tập Gender khác nhau xuống tối đa 870 câu, train LoRA và chọn
   checkpoint tốt nhất trên cùng một validation cố định. Macro-F1 cao hơn
   thắng; validation loss thấp hơn dùng để phá hòa.
3. Chọn seed tốt nhất, lấy `best_epoch` của seed đó, khởi tạo lại backbone và
   LoRA từ đầu, rồi train
   đúng số epoch đó trên toàn bộ dữ liệu được giữ lại. Task A dùng toàn bộ dữ
   liệu gốc. Task B dùng cả `original_1..5 + augment1..5`, ngoại trừ các câu
   Gender bị seed chiến thắng loại khỏi fold 1-4. Không validation hoặc early
   stopping ở giai đoạn này.
4. Dùng adapter retrain 100% để dự đoán development inputs và tạo
   `predictions.csv` cùng file ZIP nộp CodaBench.

Việc khởi tạo lại ở bước 3 giúp tất cả mẫu được học số epoch như nhau. Nếu
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
| Task B model revision | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
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
binary_train.csv
binary_validation_inputs.csv
```

Task B:

```text
original_1.csv ... original_5.csv
augment1.csv ... augment5.csv
multiclass_validation_inputs.csv
```

`Train_B_LLM.py` tái hiện cấu hình của lượt Task B đạt `0.66`. Trước khi train,
script đối chiếu cấu hình, hai GPU worker, FP16, các phiên bản package đã ghi
nhận và SHA-256 của cả 11 file input với `reproduction_task_b_066.json`. Nếu
có bất kỳ sai khác nào, script dừng trước khi tốn thời gian GPU. Bộ input khớp
tham chiếu đang được lưu riêng trong `New_Hastika_1.zip`; dữ liệu vẫn bị
`.gitignore` loại khỏi Git.

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
Sau mỗi epoch của Task B, log còn ghi precision, recall, F1 và support của từng
nhãn, kèm số lỗi theo cả hai chiều: các nhãn thật bị dự đoán nhầm vào nhãn đang
xét và các mẫu thật của nhãn đang xét bị dự đoán sang từng nhãn khác. Ma trận
nhầm lẫn tương ứng cũng được lưu trong lịch sử huấn luyện, CSV và metrics JSON.

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

Task B còn tạo một thư mục riêng cho từng seed, ví dụ
`outputs/task_b_lora/seed_42/`, cùng `seed_selection_results.csv` và
`seed_selection_summary.json`. Các artifact ở trực tiếp
`outputs/task_b_lora/` luôn tương ứng với seed chiến thắng.

Lưu ý: chọn seed tốt nhất trên cùng một validation cũng là một dạng tối ưu theo
validation, nên Macro-F1 của seed thắng có thể lạc quan hơn khả năng tổng quát
thực tế. File tổng hợp vì vậy lưu cả điểm trung bình và độ lệch chuẩn của 5
seed; test/development chính thức vẫn là đánh giá cuối cùng.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
python tests/run_llm_ddp_smoke.py
```

Smoke test CUDA riêng:

```bash
python tests/run_llm_cuda_smoke.py
```
