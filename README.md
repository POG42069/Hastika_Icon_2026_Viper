# HASTIKA @ ICON-2026 — LLM LoRA

Nhánh `Khangtest2` dùng dữ liệu original-only cho cả Task A và B
(`use_augmentation=False`). Task B đổi backbone từ `Qwen/Qwen2.5-3B-Instruct`
sang `sarvamai/sarvam-1`; nội dung prompt, seed, hyperparameter, loss và cách
ensemble giữ nguyên. Task A tiếp tục dùng `Qwen/Qwen2.5-3B-Instruct`.
Các file augment có thể vẫn nằm trong dataset nhưng không được đọc hoặc
dùng để train.

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
   `binary_original_1..4` và validation trên
   `binary_original_5`.
2. Mỗi seed chọn epoch có Macro-F1 cao nhất; validation loss thấp hơn dùng để
   phá hòa. Early stopping có patience 2.
3. Mỗi seed khởi tạo lại backbone và LoRA, rồi retrain trên toàn bộ
   `binary_original_1..5` theo best epoch riêng.
4. Năm model dự đoán development độc lập. Nhãn cuối được chọn bằng hard
   majority vote: ít nhất 3 phiếu `Hate` thì trả `Hate`.

### Task B: best epoch riêng cho từng seed + năm full-data model

Task B không còn downsample Gender và không loại bất kỳ câu train nào.

1. Với mỗi seed `42, 52, 62, 72, 82`, train trên `original_1..4` và validation
   trên `original_5`. Tập validation cố định giữa các seed, không dùng augment.
2. Selection chạy tối đa 5 epoch. Mỗi seed chọn epoch có Macro-F1 cao nhất;
   validation loss thấp hơn phá hòa, còn cả hai bằng nhau thì giữ epoch trước.
   Early stopping có patience 2. Macro-F1 là trung bình F1 của đủ sáu nhãn.
3. Khôi phục `best_holdout_adapter` để đánh giá lại và lưu kết quả của từng seed.
4. Khởi tạo lại backbone và LoRA cho cả năm seed. Mỗi model retrain trên toàn bộ
   `original_1..5` trong đúng số best epoch của chính seed đó. Ví dụ best epoch
   là `3, 3, 4, 2, 3` thì các full-data model cũng train lần lượt `3, 3, 4, 2, 3`
   epoch. Không chọn một seed thắng để loại các model còn lại.
5. Mỗi model trả xác suất cho sáu nhãn. Pipeline lấy trung bình xác suất của
   năm model với trọng số bằng nhau rồi lấy nhãn có xác suất trung bình cao
   nhất. Đây là soft ensemble, không phải hard voting.

Class-weighted loss áp dụng trong cả selection và full-data retrain. Task B
không chạy five-fold OOF và không dùng global best epoch hoặc số epoch cố định.
Full retrain không tiếp tục train từ adapter holdout: mỗi seed được khởi tạo mới.

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

Trong selection, weights được tính chỉ từ `original_1..4`, không dùng phân bố
nhãn của validation fold 5. Trong full retrain, weights được tính lại từ
`original_1..5` rồi áp dụng cho cả năm full-data model.

Với dữ liệu Task B hiện tại:

| Label | Số mẫu | Full-data weight |
|---|---:|---:|
| Gender | 1362 | 0.7623 |
| Political | 558 | 0.9866 |
| Religion | 382 | 1.1924 |
| Geo-political | 184 | 1.5245 |
| Violence | 221 | 1.5245 |
| Others | 448 | 1.1011 |

Tổng cộng có 3155 mẫu original. Công thức weighting không đổi so với
`Khangtest`, nhưng weights được tính lại từ phân bố original-only. Khoảng
clip áp dụng trước chuẩn hóa nên weight cuối có thể lớn hơn 1.50.

## Light text normalization

Pipeline chỉ sửa nhiễu kỹ thuật trước khi đưa cột `Comment` vào chat
tokenizer:

- Sửa lỗi encoding bằng `ftfy`, chuẩn hóa Unicode NFKC và giải mã HTML entity.
- Thu gọn khoảng trắng; ô `Comment` thiếu được chuyển thành chuỗi rỗng.
- Giữ nguyên URL, mention và thẻ HTML.
- Không lowercase; giữ hashtag, emoji và dấu câu.
- Không xóa stopword, không lemmatize và không chuẩn hóa ký tự/từ lặp.

Khi comment quá dài, `ChatEncoder` chỉ cắt phần comment để system prompt và
assistant label vẫn nằm trong `max_length`.

## Cấu hình LoRA mặc định

| Tham số | Giá trị |
|---|---:|
| Backbone Task A | `Qwen/Qwen2.5-3B-Instruct` |
| Backbone Task B | `sarvamai/sarvam-1` |
| LoRA rank `r` | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Target modules | `all-linear` |
| Max selection epochs | 5 cho cả Task A và Task B |
| Full-data epochs | Best epoch riêng của mỗi seed, cho cả hai task |
| Early-stopping patience | 2, chỉ áp dụng giai đoạn selection |
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
`BitsAndBytesConfig`. Cả hai backbone dùng LoRA FP16 trên T4. DDP dùng cả
hai T4, nhưng mỗi GPU vẫn giữ một bản backbone đầy đủ.

## Dữ liệu

Mặc định pipeline tìm file trong `HASTIKA_DATA_DIR`, thư mục `data/`, hoặc
một Kaggle Dataset dưới `/kaggle/input`.

Task A:

```text
binary_original_1.csv ... binary_original_5.csv
binary_validation_inputs.csv
```

Task B:

```text
original_1.csv ... original_5.csv
multiclass_validation_inputs.csv
hastika_multiclass_validation.csv
```

Train CSV cần các cột `id`, `Comment` và cột nhãn tương ứng. Development
CSV cần `id`, `Comment`. Nhánh này không yêu cầu các file augment và không
chạy augmentation audit. Dữ liệu original và development giữ nguyên;
không cần xóa hoặc sửa các file augment hiện có.

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

Code gọi `apply_chat_template`. Prompt và padding được mask bằng `-100`;
loss chỉ áp dụng cho token của assistant label và token kết thúc chat. Khi dự
đoán, pipeline tính mean token log-likelihood cho từng câu trả lời hợp lệ,
softmax sáu candidate scores để lấy xác suất, rồi chọn nhãn theo quy tắc
ensemble của từng task. Pipeline không sinh văn bản tự do và không regex sửa
kết quả.

Task B dùng chat template có sẵn trong tokenizer Sarvam: system instruction
nằm trong `<<SYS>> ... <</SYS>>`, user nằm trong `[INST] ... [/INST]`, và
assistant answer kết thúc bằng `</s>`. Nội dung system prompt và `Comment:`
vẫn là plain text như trước; các token bao quanh hội thoại theo checkpoint
mới. Tokenizer không có PAD nên code dùng EOS làm PAD và mask các vị trí
padding như hiện tại.

Theo [model card chính thức](https://huggingface.co/sarvamai/sarvam-1),
Sarvam-1 là base model hỗ trợ Kannada và tiếng Anh, chưa được instruction-tune.
Code vẫn fine-tune LoRA trên câu trả lời là nhãn và chấm sáu candidate trực
tiếp, không cần model sinh hoặc hiểu hội thoại tốt trước khi fine-tune.

## Chạy trên Kaggle

```bash
!pip install -q -r requirements.txt
!python Train_A_LLM.py
!python Train_B_LLM.py
```

Khi Kaggle có T4 x2, entry point tự relaunch bằng DDP với hai worker.

Có thể đổi tham số qua CLI:

```bash
!python Train_B_LLM.py --model sarvamai/sarvam-1
!python Train_B_LLM.py --batch-size 4 --gradient-accumulation 2
!python Train_B_LLM.py --single-gpu
!python Train_B_LLM.py --prediction-file /path/to/official_inputs.csv
```

`Train_B_LLM.py` mặc định dùng Sarvam-1. `--model` hoặc biến môi trường
`HASTIKA_LLM_MODEL` có thể ghi đè checkpoint; bỏ override Qwen cũ nếu muốn
chạy backbone mặc định mới. `HASTIKA_LLM_MODEL` vẫn áp dụng cho cả hai task.

`--max-epochs` điều chỉnh số epoch selection tối đa của Task B. Số epoch full
retrain được lấy từ best epoch riêng của từng seed, không khóa bằng 5.

### Log validation sau mỗi epoch của Task B

Thêm file gold `hastika_multiclass_validation.csv` của BTC vào Kaggle Dataset
hoặc `data/`. File cần các cột `id`, `Comment`, `Hate Category` và có 395 câu.
Pipeline đọc file này độc lập với input dùng để tạo submission, nên vẫn có
thể dùng `--prediction-file` để dự đoán test không nhãn.

Trong selection, Task B chấm `original_5` sau mỗi epoch và in Macro-F1, loss,
accuracy, confusion matrix dạng text cùng precision, recall, F1, support của
từng nhãn. Các metric này được dùng để chọn best epoch của chính seed đó.

Trong full retrain, sau mỗi epoch của từng seed, Task B đánh giá trên gold
validation 395 câu và in:

- Seed, epoch, validation loss, Macro-F1 và accuracy.
- Confusion matrix dạng text: hàng là nhãn thật, cột là nhãn dự đoán.
- Precision, recall, F1, support và số câu nhầm vào/ra từng class.

Các metric của gold validation 395 câu chỉ để theo dõi. Full retrain chạy đủ
best epoch đã chọn từ holdout, không early stop hoặc chọn checkpoint bằng gold
validation. Gold validation không được ghép vào train; ID trùng với train sẽ
gây lỗi. Trạng thái RNG được giữ lại khi chấm
để bước đánh giá không làm thay đổi random của epoch train tiếp theo.

Khi cả năm model hoàn tất, code cũng in metric của soft ensemble trên cùng
gold validation. Metric này dùng để so sánh hai nhánh có/không augment.
Task A giữ nguyên pipeline và cách log hiện tại.

Metric selection được lưu trong `seed_<seed>/selection_training_history.json`;
dự đoán holdout tốt nhất nằm trong `seed_<seed>/validation_predictions.csv`.
`seed_selection_results.csv` và `seed_selection_summary.json` ghi best epoch
riêng của tất cả seed. Metric full retrain mỗi epoch được lưu trong
`seed_<seed>/full_data_training_history.json`.
Thư mục `seed_<seed>/official_validation/` chứa predictions, metric JSON và
confusion matrix CSV theo epoch. Kết quả ensemble cuối nằm trong
`official_validation_metrics.json` và `official_validation_predictions.csv`.

Có thể đặt thư mục bằng biến môi trường:

```bash
%env HASTIKA_DATA_DIR=/kaggle/input/hastika-data
%env HASTIKA_OUTPUT_DIR=/kaggle/working/outputs
%env HASTIKA_CHECKPOINT_DIR=/kaggle/working/checkpoints
```

Task B train năm model selection trước, sau đó train năm full-data model theo
best epoch riêng; năm lượt selection đều giữ nguyên original fold 5 làm holdout.

## Artifacts

Task B tạo các artifact chính:

```text
outputs/task_b_lora/seed_42/full_data_training_history.json
outputs/task_b_lora/seed_42/selection_training_history.json
outputs/task_b_lora/seed_42/validation_predictions.csv
outputs/task_b_lora/seed_42/predictions.csv
outputs/task_b_lora/seed_52/...
outputs/task_b_lora/seed_selection_results.csv
outputs/task_b_lora/seed_selection_summary.json
outputs/task_b_lora/ensemble_predictions.csv
outputs/task_b_lora/predictions.csv
outputs/task_b_lora/task_b_submission.zip
outputs/task_b_lora/run_summary.json
checkpoints/task_b_lora/seed_42/final_full_data_adapter/
checkpoints/task_b_lora/seed_42/best_holdout_adapter/
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
