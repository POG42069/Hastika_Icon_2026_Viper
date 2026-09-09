# Viper — HASTIKA @ ICON-2026

Repository có **bốn entry point** để so sánh BERT với Qwen QLoRA trên hai
subtask HASTIKA. `Train_A.py` và `Train_B.py` giữ nguyên pipeline BERT;
`Train_A_LLM.py` và `Train_B_LLM.py` dùng `Qwen/Qwen2.5-3B-Instruct` với
QLoRA NF4 4-bit. Mỗi script tạo đúng một stratified holdout 80/20, chỉ cập nhật trọng
số trên 80% train, chọn một checkpoint bằng Macro-F1 trên 20% validation và
dùng checkpoint đó dự đoán file input không nhãn của CodaBench.

- **Task A:** Binary Hate Speech Detection (`Hate`, `Non-Hate`).
- **Task B:** Fine-Grained Hate Speech Classification (`Gender`, `Political`,
  `Religion`, `Geo-political`, `Violence`, `Others`).
- **BERT loss:** CrossEntropyLoss thường, không dùng class weights.
- **LLM loss:** causal cross-entropy chỉ trên token đáp án và phần kết thúc
  assistant; lấy trung bình theo token trong từng mẫu, rồi trung bình các mẫu.
- **GPU:** tự động sử dụng Kaggle T4 x2 bằng PyTorch DDP.
- **Submission:** tạo `predictions.csv` và ZIP đúng định dạng CodaBench.

## 1. Cấu trúc repository

```text
.
├── Train_A.py                 # Chạy toàn bộ Task A
├── Train_B.py                 # Chạy toàn bộ Task B
├── Train_A_LLM.py             # Task A: Qwen + QLoRA
├── Train_B_LLM.py             # Task B: Qwen + QLoRA
├── config.py                  # BERT, preprocessing và cấu hình task chung
├── llm_config.py              # Model và hyperparameter QLoRA
├── requirements.txt
├── resources/
│   ├── kannada_stopwords.txt
│   └── romanized_kannada_stopwords.txt
├── src/
│   ├── data.py                # Đọc, kiểm tra và tokenize dữ liệu
│   ├── distributed.py         # Tự khởi chạy DDP trên hai GPU
│   ├── llm_data.py            # Chat template, cắt comment, mask loss đáp án
│   ├── llm_training.py        # QLoRA, chấm các nhãn, chọn và lưu adapter
│   ├── preprocessing.py       # Preprocessing gần nhất với bài báo
│   ├── submission.py          # Kiểm tra và đóng gói submission
│   └── training.py            # Holdout, training và chọn checkpoint
└── tests/                     # Unit test và CPU/DDP smoke test
```

Các thư mục sau chỉ tồn tại ở local/Kaggle và đã bị `.gitignore` chặn:

```text
data/          # Dataset cuộc thi; tuyệt đối không push lên GitHub
checkpoints/   # Một checkpoint tốt nhất cho mỗi task
outputs/       # Submission, validation diagnostics và run metadata
```

## 2. Chuẩn bị Private Dataset trên Kaggle

Không đưa CSV của ban tổ chức lên GitHub. Trên Kaggle:

1. Chọn **Create → New Dataset** và upload:
   - `binary_train.csv`
   - `binary_validation_inputs.csv`
   - `multiclass_train.csv`
   - `multiclass_validation_inputs.csv`
2. Giữ dataset ở chế độ **Private**.
3. Trong **Settings → Sharing**, thêm tài khoản Kaggle của các thành viên Viper.
4. Trong Notebook, chọn **Add Input → Your Datasets** và gắn dataset này.

Code tự tìm các file trong `/kaggle/input`. Nếu có nhiều bản trùng tên, chỉ
định thư mục chính xác:

```python
import os

os.environ["HASTIKA_DATA_DIR"] = "/kaggle/input/hastika-viper-private"
```

## 3. Cấu hình Kaggle Notebook

Trong **Notebook options**:

- Accelerator: **GPU T4 x2**.
- Internet: **On** để tải BERT, NLTK resources và clone GitHub.
- Giữ Notebook **Private** trong thời gian thi.

```bash
!git clone --branch Khangtest --single-branch https://github.com/POG42069/Hastika_Icon_2026_Viper.git
%cd Hastika_Icon_2026_Viper
!pip install -q -r requirements.txt
```

NLTK tự kiểm tra và tải `stopwords`, `wordnet`, `omw-1.4` và
`averaged_perceptron_tagger_eng` ở lần chạy đầu. Script dừng với thông báo rõ
ràng nếu resource thiếu; preprocessing không bao giờ bị âm thầm bỏ qua.

## 4. Pipeline BERT

Task A và Task B chạy độc lập nhưng dùng cùng logic:

```text
Labeled train CSV
        ↓
Paper-style preprocessing
        ↓
Stratified split một lần, seed 42
        ↓
80% train ── backpropagation/cập nhật BERT
20% validation ── chỉ eval, chọn checkpoint theo Macro-F1
        ↓
Load checkpoint tốt nhất
        ↓
Dự đoán competition input không nhãn
        ↓
predictions.csv + submission ZIP
```

Kích thước holdout mặc định:

| Task | Tổng labeled | Train 80% | Validation 20% |
|---|---:|---:|---:|
| A | 6.446 | 5.156 | 1.290 |
| B | 3.159 | 2.527 | 632 |

Không có K-fold, ensemble, OOF hoặc retrain trên 100% labeled data.

## 5. Chạy Task A

```bash
!python Train_A.py
```

Đầu ra:

```text
checkpoints/task_a/best_model.pt
checkpoints/task_a/metrics.json
outputs/task_a/predictions.csv
outputs/task_a/task_a_submission.zip
outputs/task_a/validation_predictions.csv
outputs/task_a/split_manifest.csv
outputs/task_a/run_summary.json
```

Nộp `outputs/task_a/task_a_submission.zip` vào track Task A.

## 6. Chạy Task B

```bash
!python Train_B.py
```

Đầu ra tương tự trong `checkpoints/task_b/` và `outputs/task_b/`. Nộp
`outputs/task_b/task_b_submission.zip` vào track Task B. Task B không sử dụng
dự đoán của Task A làm đầu vào.

## 7. Định dạng submission

Mỗi ZIP chỉ chứa một file `predictions.csv` ở thư mục gốc:

```csv
id,label
4186,Non-Hate
5693,Hate
```

Task A chỉ xuất `Hate` hoặc `Non-Hate`. Task B chỉ xuất `Gender`, `Political`,
`Religion`, `Geo-political`, `Violence` hoặc `Others`. Code kiểm tra header,
thứ tự ID, số dòng, nhãn và nội dung ZIP trước khi hoàn tất.

## 8. Hyperparameter

Tất cả cấu hình nằm trong `config.py`:

| Cấu hình | Mặc định | Ý nghĩa |
|---|---:|---|
| `model_name` | `google-bert/bert-base-uncased` | Backbone cho cả hai task |
| `validation_size` | `0.20` | Tỷ lệ holdout có nhãn |
| `max_epochs` | 5 | Epoch tối đa |
| `learning_rate` | `2e-5` | Learning rate AdamW |
| `train_batch_size_per_gpu` | 8 | Batch riêng của từng GPU |
| `eval_batch_size_per_gpu` | 16 | Batch validation/prediction mỗi GPU |
| `gradient_accumulation_steps` | 1 | Số bước cộng dồn gradient |
| `max_length` | 128 | Độ dài token tối đa |
| `weight_decay` | 0.01 | AdamW weight decay |
| `warmup_ratio` | 0.10 | Tỷ lệ warm-up |
| `early_stopping_patience` | 2 | Epoch không cải thiện trước khi dừng |
| `seed` | 42 | Seed của split và training |

Với T4 x2, effective batch mặc định là:

```text
8 mẫu/GPU × 2 GPU × 1 gradient accumulation = 16 mẫu/update
```

## 9. Preprocessing

Các bước chạy theo thứ tự mô tả trong bài báo HASTIKA:

1. Sửa Unicode, giải mã HTML entity và lowercase.
2. Xóa HTML, URL, mention, toàn bộ hashtag, emoji, punctuation và symbol.
3. Tách Unicode word token, giữ chữ Kannada/English và số.
4. Loại English, Kannada và romanized Kannada stopwords.
5. Giữ từ ngữ cảnh tác giả nêu và các từ phủ định quan trọng.
6. POS tagging và WordNet lemmatization cho token English.
7. Chuẩn hóa ký tự và từ lặp.

Bài báo không công bố danh sách stopword Kannada/romanized Kannada hoặc công
cụ lemmatization Kannada. Vì vậy hai danh sách trong `resources/` là bản xấp xỉ
được version hóa của Viper; Kannada và romanized Kannada không bị lemmatize.
Đây là bản tái hiện gần nhất có thể kiểm chứng, không được tuyên bố là code gốc
của tác giả.

## 10. Chạy kiểm thử

Unit test và CPU smoke test:

```bash
python -m unittest discover -s tests -v
```

DDP smoke test hai tiến trình CPU:

```bash
python tests/run_ddp_smoke.py
```

Các unit test không tải pretrained weights; preprocessing test chỉ tải NLTK
data nhỏ nếu máy chưa có. Test LLM dùng Qwen2 khởi tạo ngẫu nhiên rất nhỏ và
adapter PEFT thật; đây là kiểm tra logic, không phải benchmark chất lượng.

## 11. Xử lý lỗi thường gặp

### Không tìm thấy CSV

Kiểm tra private dataset đã được gắn bằng **Add Input**. Nếu nhiều file trùng
tên, đặt `HASTIKA_DATA_DIR` như mục 2.

### Không tải được BERT hoặc NLTK

Bật Internet ở lần chạy đầu. Script không tiếp tục với preprocessing thiếu
resource vì điều đó làm kết quả giữa các lần chạy không nhất quán.

### CUDA out of memory

Giảm `train_batch_size_per_gpu` từ 8 xuống 4 và có thể tăng
`gradient_accumulation_steps` lên 2 để giữ effective batch 16.

### Chỉ phát hiện một GPU

Kiểm tra Accelerator là **GPU T4 x2**. Khi thành công, log hiển thị
`world_size=2`; người dùng không phải tự gọi `torchrun`.

## 12. Chạy LLM QLoRA

Yêu cầu Python 3.10+, PyTorch có CUDA và GPU NVIDIA. Kaggle T4 x2 là môi
trường mục tiêu. Cài `requirements.txt` trong notebook mới; nếu đã import các
thư viện trước khi cài, restart session trước khi train.

```bash
!pip install -q -r requirements.txt
!python Train_A_LLM.py
!python Train_B_LLM.py
```

Chạy từng lệnh sau khi lệnh trước đã kết thúc. Hai script LLM mặc định tự dùng
tối đa 2 GPU qua DDP, mỗi worker giữ một bản model 4-bit trên GPU riêng.
VRAM hai T4 không tự cộng thành bộ nhớ của một GPU. Một GPU CUDA cũng chạy
được; máy chỉ có CPU sẽ dừng với thông báo rõ ràng.

QLoRA dùng NF4, double quantization, LoRA trên các linear layer và gradient
checkpointing. Compute dtype tự chọn BF16 nếu GPU hỗ trợ native (Ampere trở lên),
FP16 trên T4; không tính hỗ trợ BF16 giả lập của PyTorch.
Không cần FlashAttention; dùng SDPA có sẵn trong PyTorch. Trọng số gốc đóng
băng, chỉ adapter được cập nhật. Optimizer là AdamW cho adapter; không dùng
paged optimizer. Code dùng vòng lặp PyTorch trực tiếp để kiểm soát loss và
DDP, không phụ thuộc TRL.

Mặc định chọn Qwen2.5-3B-Instruct để bắt đầu thử trên GPU giới hạn bộ nhớ;
đây không phải cam kết model tốt nhất cho Kannada-English. Xem
[model card và giấy phép](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
và khai báo model/tài nguyên trong system paper của đội.

## 13. Dữ liệu và cách LLM học

LLM dùng đúng label order, hàm stratified split, seed 42 và tỷ lệ holdout của
BERT. Mặc định `text_mode="paper"` dùng cùng preprocessing với BERT.
File `split_manifest.csv` ghi các ID train/validation để đối chiếu giữa hai
phương pháp. Tokenizer và giới hạn độ dài khác nhau nên số token thực tế khác.

Mỗi mẫu được chuyển thành system instruction, user comment và assistant label.
Ví dụ minh họa:

```text
System: Detect hate speech ... Reply with exactly one label ...
User: Comment: <clean_text>
Assistant: Hate
```

Chat được tokenize bằng `apply_chat_template` của chính model. Loss của prompt
và padding được mask bằng `-100`; attention vẫn đọc prompt. Các token nhãn và
phần kết thúc assistant do template tạo ra được học theo causal next-token
prediction, shift đúng một vị trí. `Hate`, `Non-Hate`, `Religion` và các nhãn
khác không bị giả định là một token.

`max_length` áp dụng cho **toàn bộ chat**. Nếu quá dài, chỉ cắt phần cuối
comment; giữ instructions, dấu bắt đầu assistant và toàn bộ response.
Mọi nhãn ứng viên của cùng comment dùng chính xác cùng prompt đã cắt.
Số comment bị cắt được ghi vào log và `run_summary.json`.

Mỗi epoch chỉ học trên 80% train. Validation không cập nhật tham số. Adapter
tốt nhất được chọn theo Macro-F1 trên tập nhãn cố định; loss thấp hơn phá hòa;
hòa cả hai thì giữ epoch trước. Nạp lại adapter này trước khi xuất kết quả.
Không dùng test labels, không retrain 100% và không chọn theo leaderboard.

## 14. Suy luận và file nộp LLM

Thay vì sinh văn bản tự do, pipeline tính **log-likelihood của từng đáp án hợp
lệ đầy đủ**, gồm tên nhãn và phần kết thúc chat:

```text
score(label) = sum log P(response_token | prompt, preceding response tokens)
prediction = argmax score(label)
```

Đây là cách phân loại bằng causal LM, không phải gắn classification head.
Không lấy riêng token đầu của nhãn, không sampling, không regex sửa nhãn sai
và không dùng nhãn thật để tạo prompt đánh giá. Việc thử từng candidate là
tính xác suất của một đáp án giả định, không phải cung cấp gold label.
Task B phải chấm 6 candidate/comment nên validation có thể tốn thời gian.

`prob_*` trong diagnostics là softmax của các điểm candidate, chuẩn hóa trên
các nhãn cho phép; chúng không phải confidence đã được hiệu chuẩn. Loss LLM
là token NLL nên không so trực tiếp giá trị loss với BERT; so Macro-F1 và
accuracy trên cùng split. Dùng tổng log-probability, không chuẩn hóa theo độ
dài khi chọn nhãn; độ dài nhãn là một yếu tố của xác suất chuỗi.

Đầu ra Task A:

```text
checkpoints/task_a_llm/best_adapter/adapter_config.json
checkpoints/task_a_llm/best_adapter/adapter_model.safetensors
checkpoints/task_a_llm/best_adapter/<tokenizer files>
checkpoints/task_a_llm/metrics.json
outputs/task_a_llm/predictions.csv
outputs/task_a_llm/task_a_submission.zip
outputs/task_a_llm/validation_predictions.csv
outputs/task_a_llm/split_manifest.csv
outputs/task_a_llm/training_history.json
outputs/task_a_llm/run_summary.json
```

Task B tương tự trong `task_b_llm/`, với `task_b_submission.zip`. ZIP của mỗi
task chứa duy nhất `predictions.csv`, header `id,label`, đúng ID và các nhãn
như pipeline BERT. File BERT trong `task_a/`, `task_b/` không bị ghi đè.

Adapter không phải model độc lập: khi sử dụng lại cần backbone tương ứng và
tokenizer đã lưu. Metadata ghi model revision đã resolve (nếu có), cấu hình,
package versions, input SHA-256, prompt, label order và token đáp án. Không
lưu optimizer state; chạy lại entry point sẽ bắt đầu một lần train mới và
ghi đè kết quả của cùng task LLM. Sao lưu thư mục output/checkpoint hoặc đổi
`HASTIKA_OUTPUT_DIR` và `HASTIKA_CHECKPOINT_DIR` trước khi thử cấu hình khác.

## 15. Hyperparameter LLM và tùy chọn dòng lệnh

Các mặc định trong `llm_config.py`: `LLM_TRAINING` cho Task A và
`LLM_TRAINING_B` cho Task B. Task B kế thừa cấu hình chung, đổi riêng train
batch/GPU thành 4 và gradient accumulation thành 2.

| Tham số | Mặc định |
|---|---|
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Max epochs / patience | 5 / 2 |
| Learning rate / weight decay | `1e-4` / `0.01` |
| Warmup ratio | `0.10` |
| Train batch/GPU / accumulation — Task A | 1 / 8 |
| Train batch/GPU / accumulation — Task B | 4 / 2 |
| Effective batch trên 2 GPU | 16 |
| Eval batch/GPU | 2 candidate completions |
| Max chat length | 256 token |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA target | `all-linear` |
| FP16 initial loss scale | 1024, tự điều chỉnh khi train |
| Preprocessing | `paper` (như BERT) |

Ví dụ đổi model, hoặc dùng một GPU:

```bash
python Train_A_LLM.py --model Qwen/Qwen2.5-7B-Instruct
python Train_B_LLM.py --single-gpu --gradient-accumulation 4
python Train_A_LLM.py --max-epochs 3 --max-length 256 --batch-size 1
python Train_B_LLM.py --text-mode raw
python Train_A_LLM.py --help
```

`HASTIKA_LLM_MODEL` cũng có thể đặt tên model mặc định. `--revision` nhận
commit hash/tag để cố định phiên bản backbone. Model thay thế phải hỗ trợ
causal LM, chat template có prefix response tương thích và bitsandbytes/PEFT;
không tự suy ra cấu hình thinking cho các họ model khác.

`--text-mode raw` giữ nguyên chữ hoa, dấu câu, hashtag và nội dung comment
(chỉ chuyển ô thiếu thành chuỗi rỗng). Đây là thí nghiệm khác về preprocessing;
nên ghi rõ khi so sánh với BERT. Pipeline mặc định vẫn dùng `paper`.

Sau khi ban tổ chức phát hành test, truyền đúng file đầu vào mới:

```bash
python Train_A_LLM.py --prediction-file /path/to/official_task_a_test_inputs.csv
python Train_B_LLM.py --prediction-file /path/to/official_task_b_test_inputs.csv
```

Các lệnh này vẫn train/chọn adapter rồi dự đoán; không phải chế độ chỉ inference.
Mặc định hai script chỉ nộp **validation inputs** của vòng Development. Holdout
20% nội bộ không phải validation inputs chính thức.

## 16. Kiểm thử LLM

```bash
python -m unittest discover -s tests -v
python tests/run_ddp_smoke.py
python tests/run_llm_ddp_smoke.py
```

Unit test kiểm tra causal mask, dịch nhãn đúng một vị trí, loss mask, nhãn
nhiều token, cắt comment, gradient accumulation ở nhóm cuối, adapter-only
updates, checkpoint roundtrip và toàn bộ luồng xuất CSV/ZIP cho cả hai task.
CPU DDP smoke test LLM dùng 2 worker Gloo với các shard lệch nhau và shard
rỗng để kiểm tra đủ dòng, đúng thứ tự, đồng bộ adapter và không bị treo.

Smoke test GPU riêng, cần PyTorch CUDA và bitsandbytes:

```bash
python tests/run_llm_cuda_smoke.py
```

Lệnh này tạo Qwen2 ngẫu nhiên rất nhỏ ngay trên máy, thực sự nạp NF4 4-bit,
gắn adapter, bật gradient checkpointing, train FP16 và kiểm tra save/reload.
Không tải weights Qwen3B và không cho biết chất lượng của mô hình cuộc thi.
Kết quả trên model nhỏ/CPU cũng không thay thế một lần train đầy đủ trên T4 x2.

## 17. Xử lý lỗi LLM

- **CUDA unavailable:** dùng notebook GPU, kiểm tra `torch.cuda.is_available()`;
  không cài wheel CPU-only vào Kaggle.
- **Out of memory:** dùng `--batch-size 1`, giảm `--max-length` nhưng chừa đủ
  chỗ cho prompt và nhãn; giảm eval batch trong `llm_config.py`; cân nhắc model
  nhỏ hơn. Gradient checkpointing đã bật mặc định.
- **BF16 không được hỗ trợ trên T4:** giữ `compute_dtype="auto"` để dùng FP16.
- **FP16 overflow:** loss scale tự giảm và LR không tăng ở bước optimizer bị
  bỏ qua. Nếu cả epoch không có update thành công, code dừng thay vì lưu một
  adapter chưa học; giảm `initial_loss_scale` để thử lại.
- **Lỗi import/quantization:** cài đúng các phiên bản trong `requirements.txt`
  rồi restart notebook; ghi lại `pip freeze` và thông báo lỗi.
- **Chat template không tương thích:** thử checkpoint Qwen2.5-Instruct mặc
  định; code dừng rõ ràng thay vì tạo supervision sai.

Tài liệu kỹ thuật: [QLoRA](https://arxiv.org/abs/2305.14314),
[PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization),
[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
