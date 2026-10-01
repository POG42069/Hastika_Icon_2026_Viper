# HASTIKA @ ICON-2026 — LLM LoRA

Trên `Khangtest2`, Task B train **original-only** và chỉ đưa bản dịch tiếng Anh
`Comment_en` vào prompt plain text. Nhánh này lấy pipeline của `Khangtest`,
giữ `Qwen/Qwen2.5-3B-Instruct`, full retrain cố định 5 epoch cho năm seed,
class-weighted loss và soft ensemble. Task A giống `Khangtest`.
Khác biệt thí nghiệm của Task B là `Khangtest` dùng raw + English, còn
`Khangtest2` dùng English-only.

Pipeline fine-tune causal LLM cho hai task HASTIKA:

- Task A: phát hiện `Non-Hate` / `Hate`.
- Task B: phân loại `Gender`, `Political`, `Religion`, `Geo-political`,
  `Violence`, `Others`.

Pipeline dùng **LoRA chuẩn, không lượng tử hóa**. Backbone được nạp bằng FP16
trên T4 hoặc BF16 trên GPU hỗ trợ BF16 native. Trọng số backbone đóng băng;
optimizer chỉ cập nhật các ma trận LoRA.

## Quy trình huấn luyện

### Task A

1. Mỗi seed `42, 52, 62, 72, 82` train trên
   `binary_original_1..4 + binary_augment1..4` và validation trên
   `binary_original_5`.
2. Mỗi seed chọn epoch có Macro-F1 cao nhất; validation loss thấp hơn dùng để
   phá hòa. Early stopping có patience 2.
3. Mỗi seed khởi tạo lại backbone và LoRA, rồi retrain trên toàn bộ
   `binary_original_1..5 + binary_augment1..5` theo best epoch riêng.
4. Năm model dự đoán development độc lập. Nhãn cuối được chọn bằng hard
   majority vote: ít nhất 3 phiếu `Hate` thì trả `Hate`.

### Task B: thí nghiệm fixed epoch + năm full-data model

Task B không còn downsample Gender và không loại bất kỳ câu train nào.

1. Tạm bỏ toàn bộ giai đoạn five-fold OOF và không chọn global best epoch.
2. Đặt số epoch full-data cố định bằng `5` cho thí nghiệm phương sai.
3. Khởi tạo mới năm model với seed `42, 52, 62, 72, 82`. Mỗi model train trên
   toàn bộ `original_1..5` trong đúng năm epoch. Các file `augment1..5` không
   được đọc, không audit và không tham gia tính class weight.
4. Mỗi model trả xác suất cho sáu nhãn. Pipeline lấy trung bình xác suất của
   năm model với trọng số bằng nhau rồi lấy nhãn có xác suất trung bình cao
   nhất. Đây là soft ensemble, không phải hard voting.

Class-weighted loss vẫn được áp dụng cho cả năm model. Mỗi model được khởi tạo
mới để mọi hàng dữ liệu được học đúng năm epoch như nhau.
Prompt English-only được áp dụng nhất quán ở train, validation mỗi epoch
và development prediction. Nhãn gold chỉ dùng cho loss hoặc tính metric.

## Class-weighted loss của Task B

Với một câu thuộc class `c`, pipeline trước tiên tính mean negative
log-likelihood trên các token của câu trả lời assistant:

```text
L_i = mean NLL của các answer token trong sample i
```

Trọng số class được tính từ riêng tập train của lượt hiện tại:

```text
raw_weight_c = clip(sqrt(N / (K * n_c)), 0.75, 1.50)
weight_c = raw_weight_c / mean_sample(raw_weight)
weighted_loss = mean(weight_label_i * L_i)
```

Trong đó `N` là tổng số mẫu train, `K=6`, và `n_c` là số mẫu của class
`c`. Căn bậc hai làm mức bù nhẹ hơn inverse-frequency đầy đủ; khoảng clip
ngăn class hiếm chi phối gradient. Bước chuẩn hóa giữ mean sample weight bằng
1, nên độ lớn loss và learning rate gần với pipeline cũ.

Trong thí nghiệm fixed epoch hiện tại, weights được tính từ năm file original
trước khi train năm full-data model.

Với dữ liệu Task B hiện tại:

| Label | Số mẫu | Full-data weight |
|---|---:|---:|
| Gender | 1362 | 0.7623 |
| Political | 558 | 0.9866 |
| Religion | 382 | 1.1924 |
| Geo-political | 184 | 1.5245 |
| Violence | 221 | 1.5245 |
| Others | 448 | 1.1011 |

Tổng cộng có 3155 mẫu original. Gender vẫn xuất hiện đủ 1362 câu; không
downsample. Công thức weighting giữ nguyên, chỉ tính lại trên original-only.
Khoảng clip áp dụng trước chuẩn hóa nên weight cuối có thể lớn hơn 1.50.

## Light text normalization

Pipeline chỉ sửa nhiễu kỹ thuật của comment. Task A đưa `Comment` vào chat
tokenizer; Task B chỉ đưa `Comment_en`. Raw `Comment` vẫn được chuẩn hóa để
kiểm tra dữ liệu và ghép gold validation, không được đưa vào prompt Task B:

- Sửa lỗi encoding bằng `ftfy`, chuẩn hóa Unicode NFKC và giải mã HTML entity.
- Thu gọn khoảng trắng; ô `Comment` thiếu được chuyển thành chuỗi rỗng.
- Giữ nguyên URL, mention và thẻ HTML.
- Không lowercase; giữ hashtag, emoji và dấu câu.
- Không xóa stopword, không lemmatize và không chuẩn hóa ký tự/từ lặp.

Khi comment quá dài, `ChatEncoder` chỉ cắt phần comment để system prompt và
assistant label vẫn nằm trong `max_length`. Với Task B, toàn bộ token còn lại
dành cho bản dịch tiếng Anh; độ dài raw không ảnh hưởng đến prompt hoặc việc
truncate. Nếu phải cắt, giữ đầu bản dịch, tiêu đề và toàn bộ assistant label.
`max_length` vẫn là 256.

## Cấu hình LoRA mặc định

| Tham số | Giá trị |
|---|---:|
| Backbone | `Qwen/Qwen2.5-3B-Instruct` |
| LoRA rank `r` | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Target modules | `all-linear` |
| Max epochs | 5 cho Task A; Task B cố định 5 epoch |
| Early-stopping patience | 2, chỉ áp dụng quy trình Task A |
| Max chat length | 256 |
| Learning rate | `1e-4` |
| Weight decay | 0.01 |
| Warmup ratio | 0.10 |
| Batch/GPU | 4 |
| Gradient accumulation | 2 |
| Effective batch trên T4 x2 | 16 |
| Eval batch/GPU | 2 candidate completions |
| Max gradient norm | 1.0 |

LoRA áp dụng:

```text
W = W0 + (alpha / r) BA
```

Với `r=32`, `alpha=64`, hệ số scale là 2. Backbone `W0` không được
optimizer cập nhật.

Đây không phải QLoRA: code không dùng NF4, double quantization hay
`BitsAndBytesConfig`. Qwen2.5-3B phù hợp với T4 16 GB hơn bản 7B khi dùng
LoRA FP16. DDP dùng cả hai T4, nhưng mỗi GPU vẫn giữ một bản backbone đầy đủ.

## Dữ liệu

Mặc định pipeline tìm file trong `HASTIKA_DATA_DIR`, thư mục `data/`, hoặc
một Kaggle Dataset dưới `/kaggle/input`.

Task A:

```text
binary_original_1.csv ... binary_original_5.csv
binary_augment1.csv ... binary_augment5.csv
binary_validation_inputs.csv
```

Task B:

```text
original_1.csv ... original_5.csv
multiclass_validation_inputs.csv
hastika_multiclass_validation.csv
```

Task A: train CSV cần `id`, `Comment`, `Label`; development cần `id`, `Comment`.
Các file augment Task A cần thêm `source_id`,
`source_group` và `fold`. Loader kiểm tra nguồn tồn tại, nguồn cùng fold,
nhãn augment khớp nhãn nguồn và một `source_group` không vượt qua nhiều fold.

Task B: dùng năm original và development từ `enhanced_hastika_augment_translate.zip`.
Original cần `id`, `Comment`, `Comment_en`, `Hate Category`; development cần
`id`, `Comment`, `Comment_en`. Thiếu cột hoặc bản dịch trống sẽ báo lỗi,
không tự dịch, không dùng raw thay thế khi thiếu English. Các file augment có thể vẫn nằm trong
dataset nhưng Task B không đọc chúng.
Không cần sửa hay bỏ cột `Comment` trong dataset đã dịch. Cột này được giữ
để kiểm tra ID và đối chiếu gold validation; `english_only_prompt=True`
bảo đảm chỉ `Comment_en` đi vào prompt.

`hastika_multiclass_validation.csv` vẫn là file gold để báo cáo metric, không
tham gia train. File này không bắt buộc có `Comment_en`: loader ghép bản dịch
từ development theo ID và kiểm tra comment raw khớp sau chuẩn hóa nhẹ. ID
thiếu, raw không khớp hoặc bản dịch có sẵn khác development sẽ báo lỗi.
Thứ tự file gold được giữ nguyên khi ghép.

`augmentation_audit.csv` của Task A đánh dấu câu quá ngắn, trùng lặp, quá khác câu
nguồn hoặc gần giống original khác. Cấu hình
`augmentation_audit_action="report"` chỉ báo cáo; các câu bị đánh dấu vẫn
được train.

Dữ liệu cuộc thi nằm trong `data/` và được `.gitignore` chặn, không được
commit lên GitHub.

## Tokenization, loss và dự đoán

Task A dùng nhãn chính thức làm assistant response:

```text
System: Classify ... Allowed labels: Non-Hate; Hate.
User: Comment:\n<normalized Comment>
Assistant: Non-Hate hoặc Hate
```

Task B cũng dùng trực tiếp sáu tên nhãn chính thức làm assistant response:
`Gender`, `Political`, `Religion`, `Geo-political`, `Violence`, `Others`.
Prompt liệt kê các nhãn hợp lệ; file submission giữ nguyên tên nhãn chính thức.

User prompt Task B (plain text, không mã hóa A–F):

```text
English translation:
<normalized Comment_en>
```

System prompt Task B yêu cầu phân loại một hateful English comment; danh
sách sáu nhãn và cách chấm likelihood giữ nguyên. Không
đưa nhãn gold hoặc ghi chú trong `translation_audit.csv` vào user prompt.
`run_summary.json` ghi `prompt_input.english_only_prompt=true`,
`raw_column=null`, cột dịch, augmentation đã tắt và cách truncate.

Code gọi `apply_chat_template`. Prompt và padding được mask bằng `-100`;
loss chỉ áp dụng cho token của assistant label và token kết thúc chat. Khi dự
đoán, pipeline tính mean token log-likelihood cho từng câu trả lời hợp lệ,
softmax sáu candidate scores để lấy xác suất, rồi chọn nhãn theo quy tắc
ensemble của từng task. Pipeline không sinh văn bản tự do và không regex sửa
kết quả.

## Chạy trên Kaggle

```bash
!pip install -q -r requirements.txt
!python Train_A_LLM.py
!python Train_B_LLM.py
```

Khi Kaggle có T4 x2, entry point tự relaunch bằng DDP với hai worker.

Có thể đổi tham số qua CLI:

```bash
!python Train_B_LLM.py --model Qwen/Qwen2.5-3B-Instruct
!python Train_B_LLM.py --batch-size 4 --gradient-accumulation 2
!python Train_B_LLM.py --single-gpu
!python Train_B_LLM.py --prediction-file /path/to/official_inputs.csv
```

Số epoch của thí nghiệm Task B hiện được khóa bằng
`TASK_B.fixed_full_retrain_epochs=5`; tùy chọn `--max-epochs` không thay đổi
giá trị cố định này.

### Log validation sau mỗi epoch của Task B

Thêm file gold `hastika_multiclass_validation.csv` của BTC vào Kaggle Dataset
hoặc `data/`. File cần các cột `id`, `Comment`, `Hate Category` và có 395 câu.
Pipeline đọc file này độc lập với input dùng để tạo submission, nên vẫn có
thể dùng `--prediction-file` để dự đoán test không nhãn.

Sau mỗi epoch của từng seed, Task B đánh giá trên gold validation và in:

- Seed, epoch, validation loss, Macro-F1 và accuracy.
- Confusion matrix dạng text: hàng là nhãn thật, cột là nhãn dự đoán.
- Precision, recall, F1, support và số câu nhầm vào/ra từng class.

Các metric chỉ để theo dõi. Pipeline vẫn train đủ 5 epoch, không early stop
hoặc chọn checkpoint bằng gold validation. Validation không được ghép vào
train; ID trùng với train sẽ gây lỗi. Trạng thái RNG được giữ lại khi chấm
để bước đánh giá không làm thay đổi random của epoch train tiếp theo.

Khi cả năm model hoàn tất, code cũng in metric của soft ensemble trên cùng
gold validation. Metric này dùng để so sánh hai nhánh raw + English và
English-only; cả hai đều không dùng augment cho Task B.
Task A giữ nguyên pipeline và cách log hiện tại.

Metric mỗi epoch được lưu trong `seed_<seed>/full_data_training_history.json`.
Thư mục `seed_<seed>/official_validation/` chứa predictions, metric JSON và
confusion matrix CSV theo epoch. Kết quả ensemble cuối nằm trong
`official_validation_metrics.json` và `official_validation_predictions.csv`.

Có thể đặt thư mục bằng biến môi trường:

```bash
%env HASTIKA_DATA_DIR=/kaggle/input/hastika-data
%env HASTIKA_OUTPUT_DIR=/kaggle/working/outputs
%env HASTIKA_CHECKPOINT_DIR=/kaggle/working/checkpoints
```

Task B hiện bỏ qua OOF và chỉ train năm full-data model, mỗi model đúng 5 epoch.

## Artifacts

Task B tạo các artifact chính:

```text
outputs/task_b_lora/seed_42/full_data_training_history.json
outputs/task_b_lora/seed_42/predictions.csv
outputs/task_b_lora/seed_52/...
outputs/task_b_lora/ensemble_predictions.csv
outputs/task_b_lora/predictions.csv
outputs/task_b_lora/task_b_submission.zip
outputs/task_b_lora/run_summary.json
checkpoints/task_b_lora/seed_42/final_full_data_adapter/
checkpoints/task_b_lora/seed_52/...
checkpoints/task_b_lora/metrics.json
```

`ensemble_predictions.csv` chứa nhãn từng seed, xác suất trung bình, số phiếu
chẩn đoán và nhãn cuối. File để nộp là
`outputs/task_b_lora/task_b_submission.zip`.

Một lượt chạy mới không tạo artifact OOF. Pipeline không tự động xóa artifact
OOF cũ nếu cùng thư mục output từng được dùng cho phiên bản trước.

Task A tiếp tục tạo per-seed holdout history, full-data adapters,
`voting_predictions.csv`, `predictions.csv` và
`task_a_submission.zip`.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
python tests/run_llm_ddp_smoke.py
```

Smoke test CUDA riêng:

```bash
python tests/run_llm_cuda_smoke.py
```
