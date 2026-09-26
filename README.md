# HASTIKA @ ICON-2026 — LLM LoRA

Nếu lần chạy Task A trên Kaggle bị dừng sau 12 giờ và bạn có
`results-7.zip`, xem [RESUME_KAGGLE.md](RESUME_KAGGLE.md) để chạy tiếp theo từng
seed bằng `Resume_A_LLM.py`.

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
   thắng; validation loss thấp hơn dùng để phá hòa trong cùng seed. Task B còn
   lấy ngẫu nhiên một tập Gender khác nhau xuống tối đa 870 câu cho mỗi seed.
3. Với cả hai task, từng seed khởi tạo lại backbone và LoRA từ đầu rồi retrain
   trên dữ liệu đầy đủ theo `best_epoch` riêng. Task B tiếp tục loại đúng tập
   Gender mà seed đó đã chọn khi chạy phase selection; vì vậy 5 seed có thể
   retrain trên 5 tập Gender khác nhau nhưng cùng số lượng mẫu.
4. Năm adapter dự đoán development độc lập và có trọng số ngang nhau. Task A
   dùng majority vote. Task B dùng plurality vote; nếu nhiều nhãn cùng số phiếu
   cao nhất thì lần lượt phá hòa bằng tổng điểm Borda, tổng `mean_token_log_probs`
   đã z-normalize riêng trong từng seed/mẫu, rồi thứ tự label cố định trong
   `llm_config.py` nếu mọi giá trị vẫn bằng nhau.
5. Kết quả ensemble dùng để tạo
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
| Backbone | `Qwen/Qwen3-4B-Instruct-2507` |
| Model revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| LoRA rank `r` | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Target modules | `all-linear` |
| Max epochs để chọn | 10 |
| Early-stopping patience | 6 |
| Max chat length | 256 |
| Learning rate | `2e-4` |
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
Qwen3-4B-Instruct-2507 chỉ hỗ trợ non-thinking mode theo model card chính thức:
https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507. Không cần truyền
`enable_thinking=False` cho chat template. DDP không cộng VRAM của hai GPU:
mỗi GPU giữ một bản backbone. Revision model được pin để tránh thay đổi
checkpoint giữa các lần chạy.
Mức `2e-4` là thiết lập thí nghiệm, chưa được xác nhận là LR tối ưu trên dev.
Nếu đặt `HASTIKA_LLM_MODEL` trong môi trường, giá trị đó sẽ ghi đè model mặc
định; khi đổi sang model khác, cần truyền `--revision` phù hợp với model đó.

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
!python Train_A_LLM.py --model Qwen/Qwen3-4B-Instruct-2507
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

Task B cũng dùng trực tiếp các nhãn chính thức `Gender`, `Political`, `Religion`,
`Geo-political`, `Violence` và `Others`; pipeline không dùng mã nội bộ `A`–`F`.

Code gọi `apply_chat_template` của tokenizer. Prompt và padding được mask bằng
`-100`; loss chỉ áp dụng cho các token của assistant label và phần kết thúc
chat. Khi dự đoán, pipeline tính mean token log-likelihood cho từng câu trả lời
hợp lệ và chọn nhãn có điểm lớn nhất. Phép chuẩn hóa theo số token tránh thiên
vị câu trả lời ngắn khi `Non-Hate` và `Hate` có độ dài token khác nhau. Pipeline
không sinh văn bản tự do và không regex sửa kết quả.

Tiến trình train, validation và prediction được ghi thành dòng cố định mỗi 10%
với số batch/candidate, elapsed time, ETA, tốc độ và mean loss khi train. Cách
này tránh progress bar động bị Kaggle nối thành một khối log dài khó đọc.
Sau mỗi epoch của Task B, log còn ghi precision, recall, F1 và support của từng
nhãn, kèm số lỗi theo cả hai chiều: các nhãn thật bị dự đoán nhầm vào nhãn đang
xét và các mẫu thật của nhãn đang xét bị dự đoán sang từng nhãn khác. Ma trận
nhầm lẫn tương ứng cũng được lưu trong lịch sử huấn luyện, CSV và metrics JSON.

## Artifacts

Task A:

```text
checkpoints/task_a_qwen3_4b_2507_lora/seed_42/best_holdout_adapter/
checkpoints/task_a_qwen3_4b_2507_lora/seed_42/final_full_data_adapter/
checkpoints/task_a_qwen3_4b_2507_lora/seed_52/...
checkpoints/task_a_qwen3_4b_2507_lora/seed_62/...
checkpoints/task_a_qwen3_4b_2507_lora/seed_72/...
checkpoints/task_a_qwen3_4b_2507_lora/seed_82/...
checkpoints/task_a_qwen3_4b_2507_lora/metrics.json
outputs/task_a_qwen3_4b_2507_lora/seed_42/validation_predictions.csv
outputs/task_a_qwen3_4b_2507_lora/seed_42/predictions.csv
outputs/task_a_qwen3_4b_2507_lora/seed_52/...
outputs/task_a_qwen3_4b_2507_lora/selection_training_history.json
outputs/task_a_qwen3_4b_2507_lora/split_manifest.csv
outputs/task_a_qwen3_4b_2507_lora/validation_predictions.csv
outputs/task_a_qwen3_4b_2507_lora/seed_selection_results.csv
outputs/task_a_qwen3_4b_2507_lora/voting_predictions.csv
outputs/task_a_qwen3_4b_2507_lora/predictions.csv
outputs/task_a_qwen3_4b_2507_lora/task_a_submission.zip
outputs/task_a_qwen3_4b_2507_lora/run_summary.json
```

`validation_predictions.csv` ở thư mục gốc là kết quả của seed có holdout tốt
nhất và chỉ dùng để báo cáo. `voting_predictions.csv` lưu dự đoán của cả 5 seed,
số phiếu từng nhãn và nhãn cuối. `predictions.csv` cùng ZIP ở thư mục gốc là
kết quả majority vote; prediction của từng adapter nằm trong thư mục seed.

Task B có cấu trúc artifact tương tự Task A: mỗi seed có
`best_holdout_adapter/`, `final_full_data_adapter/`, prediction và lịch sử
retrain riêng. `outputs/task_b_qwen3_4b_2507_lora/voting_predictions.csv` lưu nhãn của từng
seed, số phiếu, điểm Borda, tổng candidate score đã chuẩn hóa, phương thức phá
hòa và nhãn cuối. Các file `predictions.csv`, ZIP và `run_summary.json` ở thư
mục gốc là kết quả ensemble; seed có holdout tốt nhất chỉ được dùng để xuất báo
cáo validation đại diện, không còn là seed duy nhất được retrain.

Macro-F1 holdout của 5 seed chỉ dùng để chọn epoch riêng cho từng seed. File
tổng hợp lưu cả điểm trung bình và độ lệch chuẩn; test/development chính thức
vẫn là đánh giá cuối cùng.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
python tests/run_llm_ddp_smoke.py
```

Smoke test CUDA riêng:

```bash
python tests/run_llm_cuda_smoke.py
```
