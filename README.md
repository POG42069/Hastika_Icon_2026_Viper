# HASTIKA @ ICON-2026 — LLM LoRA

Bản Nhotest này mặc định dùng **bản gốc code-mixed + bản dịch tiếng Anh** trong
cùng một prompt. Xem [BILINGUAL.md](BILINGUAL.md) để chọn đúng thư mục dữ liệu,
chạy huấn luyện và so sánh với English-only.

Pipeline fine-tune causal LLM cho hai task HASTIKA:

- Task A: phát hiện `Non-Hate` / `Hate`.
- Task B: phân loại `Gender`, `Political`, `Religion`, `Geo-political`,
  `Violence`, `Others`.

Pipeline dùng **LoRA chuẩn, không lượng tử hóa**. Backbone được nạp bằng
FP16 trên T4 hoặc BF16 trên GPU hỗ trợ native BF16; trọng số backbone đóng
băng và chỉ các ma trận LoRA được optimizer cập nhật.

## Quy trình huấn luyện

Task A giữ luồng holdout selection và full retrain của Nhotest:

1. Dùng original song ngữ, không augmentation; train fold 1–4, validation fold 5.
   Có thể đọc file compact `binary_original_1_4.csv` + `binary_original_5.csv`,
   hoặc đủ năm file `binary_original_1.csv` tới `binary_original_5.csv`.
2. Chạy 5 seed `42, 52, 62, 72, 82`, chọn best epoch bằng macro-F1 validation,
   rồi dùng validation loss để phá hòa.
3. Từng seed khởi tạo lại backbone/LoRA và retrain dữ liệu đầy đủ theo best epoch.
4. Dự đoán từng seed và majority vote với trọng số ngang nhau.
5. Xuất `predictions.csv` và ZIP nộp CodaBench.

Task B giữ prompt, class-weighted loss và soft ensemble từ Khangtest (`e27514b`).
Chạy 5 seed trên fold 1–4, chọn best epoch bằng macro-F1 fold 5 (loss phá hòa),
tối đa 5 epoch và patience 2. Sau đó khởi tạo lại từng seed để retrain cả 5 fold
theo best epoch riêng, không cap Gender; ensemble bằng trung bình xác suất. Loader dùng `OriginalComment` (gốc) và
`Comment` (tiếng Anh) từ dataset song ngữ. Xem [TASK_B_KHANGTEST.md](TASK_B_KHANGTEST.md)
để chạy, cung cấp gold validation và resume. Task B vẫn dùng 512 token như yêu cầu.

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

Khi prompt song ngữ quá dài, `ChatEncoder` chia ngân sách token cho cả hai
phiên bản, giữ phần đầu/cuối và đánh dấu `[...]` ở phần bị lược. System prompt
và assistant label vẫn nằm đầy đủ trong `max_length`. Chế độ English-only
giữ quy tắc cắt phần cuối comment.

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
| Max epochs để chọn | 5 |
| Early-stopping patience | 4 |
| Max chat length | Task A `256`; Task B `512` |
| Learning rate | Task A `2e-4`; Task B `1e-4` |
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
Pipeline dùng chat template chính thức của
https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507. Đây là bản Instruct
non-thinking; template không thêm phần suy luận `<think>`. DDP không cộng VRAM
của hai GPU: mỗi GPU giữ một bản backbone. Revision model được pin để tránh
thay đổi checkpoint giữa các lần chạy.
Mức `2e-4` là thiết lập thí nghiệm, chưa được xác nhận là LR tối ưu trên dev.
Nếu đặt `HASTIKA_LLM_MODEL` trong môi trường, giá trị đó sẽ ghi đè model mặc
định; khi đổi sang model khác, cần truyền `--revision` phù hợp với model đó.

## Dữ liệu

Mặc định pipeline tìm các file sau trong `HASTIKA_DATA_DIR`, thư mục `data/`,
hoặc một Kaggle Dataset dưới `/kaggle/input`.

Task A (mặc định):

```text
binary_original_1_4.csv
binary_original_5.csv
binary_validation_inputs.csv
```

Task B (mặc định):

```text
original_1.csv
original_2.csv
original_3.csv
original_4.csv
original_5.csv
multiclass_validation_inputs.csv
```

Train CSV cần `id`, `Comment`, `OriginalComment` và cột nhãn tương ứng.
Development CSV cần `id`, `Comment`, `OriginalComment`. Giải nén
`hastika_bilingual_no_augments.zip` và trỏ `HASTIKA_DATA_DIR` vào đúng thư mục
chứa CSV. Cả hai task giữ cột tiếng Anh và câu gốc trong cùng một mẫu.

Script thử nghiệm augmentation Task A dưới đây chỉ dành cho bộ dữ liệu cũ;
pipeline tiếng Anh mặc định không dùng output của script này:

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

### Tự tiếp tục sau timeout 12 giờ

Cơ chế resume nằm trực tiếp trong cả hai entry point. Sau một version bị dừng:

1. Lưu output của version đó thành Kaggle Dataset/Notebook Output.
2. Gắn **tất cả** output của các version trước vào notebook tiếp theo bằng
   `Add Input`.
3. Chạy lại đúng các lệnh cũ:

```bash
python Train_A_LLM.py
python Train_B_LLM.py
```

Pipeline tự quét `/kaggle/input`, kể cả các cây thư mục đã giải nén và file
`.zip` như `result-7.zip`. Nó chỉ đọc artifact thuộc đúng task đang chạy. Có
thể chỉ định nguồn rõ ràng (lặp cờ nhiều lần) hoặc tắt auto-scan:

```bash
python Train_A_LLM.py \
  --resume-from /kaggle/input/run-01 \
  --resume-from /kaggle/input/run-02/result-7.zip

python Train_B_LLM.py --no-auto-resume
```

Thứ tự tận dụng artifact cho từng seed:

- Nếu phase 1 có `seed_summary.json`, split manifest và validation prediction
  hợp lệ, pipeline bỏ qua toàn bộ holdout selection của seed đó.
- Nếu phase 2 đã có `prediction_scores.npz`, pipeline dùng ngay prediction và
  không nạp model. Nếu chỉ có final adapter, pipeline chỉ chạy inference lại,
  không train lại.
- Trong các run tạo bởi phiên bản code mới này, cuối **mỗi epoch** pipeline lưu
  adapter hiện tại, optimizer, scheduler, GradScaler, RNG của từng DDP rank và
  DataLoader generator. Lần sau tiếp tục chính xác từ epoch kế tiếp. Nếu timeout
  giữa một epoch thì chỉ epoch đang dở phải chạy lại.
- Với output cũ chưa có training state, các seed đã hoàn tất vẫn được bỏ qua.
  Nếu phase 1 còn dở nhưng có history và `best_holdout_adapter`, pipeline tiếp
  tục từ adapter đó với optimizer/scheduler mới và ghi cảnh báo trong log. Đây
  là phần tốt nhất có thể khôi phục vì optimizer state cũ không tồn tại.
- Trường hợp output rất cũ chỉ còn `predictions.csv`, vote của seed vẫn được
  dùng. Riêng Task B, nếu không còn adapter hoặc candidate scores để phá hòa,
  log sẽ báo rõ rằng tie-break phải dùng one-hot vote khôi phục.

Artifact mới được kiểm tra bằng hash dữ liệu train, model revision, cấu hình
LoRA/training, số GPU DDP, ID/thứ tự development và hash file prediction trước
khi được dùng. Artifact kiểu cũ không có hash được kiểm tra bằng toàn bộ ID,
split assignment, số dòng, seed và số epoch có sẵn. Để exact-resume, lần chạy
sau phải giữ cùng tham số CLI và cùng số GPU; nếu không khớp, checkpoint state
sẽ bị từ chối an toàn và log nêu lý do.

Nên giữ `outputs/` và `checkpoints/` trong `/kaggle/working` (hoặc bên dưới thư
mục repo đã clone vào đó), vì cả báo cáo nhỏ lẫn adapter/training state đều cần
có trong Notebook Output. Checkpoint tạm của một phase được dọn sau khi phase
đã xuất artifact hoàn chỉnh, nên dung lượng tăng thêm chủ yếu chỉ tồn tại cho
seed đang chạy dở.

Đổi backbone hoặc giới hạn tài nguyên:

```bash
!python Train_A_LLM.py --model Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554
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
checkpoints/task_a_qwen3_4b_2507_lora_bilingual/seed_42/best_holdout_adapter/
checkpoints/task_a_qwen3_4b_2507_lora_bilingual/seed_42/final_full_data_adapter/
checkpoints/task_a_qwen3_4b_2507_lora_bilingual/seed_52/...
checkpoints/task_a_qwen3_4b_2507_lora_bilingual/seed_62/...
checkpoints/task_a_qwen3_4b_2507_lora_bilingual/metrics.json
outputs/task_a_qwen3_4b_2507_lora_bilingual/seed_42/validation_predictions.csv
outputs/task_a_qwen3_4b_2507_lora_bilingual/seed_42/predictions.csv
outputs/task_a_qwen3_4b_2507_lora_bilingual/seed_42/prediction_scores.npz
outputs/task_a_qwen3_4b_2507_lora_bilingual/seed_52/...
outputs/task_a_qwen3_4b_2507_lora_bilingual/selection_training_history.json
outputs/task_a_qwen3_4b_2507_lora_bilingual/split_manifest.csv
outputs/task_a_qwen3_4b_2507_lora_bilingual/validation_predictions.csv
outputs/task_a_qwen3_4b_2507_lora_bilingual/seed_selection_results.csv
outputs/task_a_qwen3_4b_2507_lora_bilingual/voting_predictions.csv
outputs/task_a_qwen3_4b_2507_lora_bilingual/predictions.csv
outputs/task_a_qwen3_4b_2507_lora_bilingual/task_a_submission.zip
outputs/task_a_qwen3_4b_2507_lora_bilingual/run_summary.json
```

Trong lúc một epoch vừa hoàn tất nhưng phase chưa xong, checkpoint có thêm:

```text
checkpoints/<task>/seed_<seed>/resume_state/selection/epoch_<N>/
checkpoints/<task>/seed_<seed>/resume_state/full_retrain/epoch_<N>/
```

Mỗi phase chỉ giữ checkpoint epoch hoàn chỉnh mới nhất. `latest.json` chỉ được
cập nhật sau khi adapter và `training_state.pt` đã ghi xong, tránh nhận nhầm
checkpoint bị cắt giữa lúc Kaggle timeout.

`validation_predictions.csv` ở thư mục gốc là kết quả của seed có holdout tốt
nhất và chỉ dùng để báo cáo. `voting_predictions.csv` lưu dự đoán của cả 5 seed Task A,
số phiếu từng nhãn và nhãn cuối. `predictions.csv` cùng ZIP ở thư mục gốc là
kết quả majority vote; prediction của từng adapter nằm trong thư mục seed.

Task B dùng thư mục `task_b_qwen3_4b_2507_lora_bilingual_selection_soft`.
`ensemble_predictions.csv` lưu nhãn từng seed và xác suất trung bình dùng cho
nhãn cuối. Mỗi seed có best holdout adapter và summary best epoch, final adapter sau retrain,
lịch sử cả hai giai đoạn và cache prediction. Chi tiết trong [TASK_B_KHANGTEST.md](TASK_B_KHANGTEST.md).

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
