# Khangtest4: Sarvam-1 trên cả Task A và B

Nhánh được tạo từ **Khangtest3 `5308fee2027c07ee7b5b6e0ee1f8ab06ba0bc070`**.
Thay backbone/tokenizer bằng `sarvamai/sarvam-1`, pin revision
`e9607337286ddf496d4a2562b194e489dcf3feea` cho cả hai task.

Theo [model card](https://huggingface.co/sarvamai/sarvam-1), đây là model
text-completion, không phải checkpoint instruction-tuned. Thí nghiệm dùng chat
template đi kèm tokenizer để serialize system/user/assistant và fine-tune LoRA
trên label completion. Không giả định model có sẵn năng lực chat instruction.
Khi tokenizer không có padding token, code tiếp tục dùng EOS làm padding và mask
padding khỏi loss như pipeline Khangtest3.

## Settings giữ từ Khangtest3

| Setting | Task A | Task B |
|---|---|---|
| Input | English trước, original sau | English trước, original sau |
| Backbone | Sarvam-1 | Sarvam-1 |
| Max toàn bộ chat | 256 | 256 |
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 | 32 / 64 / 0.05 |
| Target modules | all-linear | all-linear |
| Learning rate | 2e-4 | 1e-4 |
| Chọn epoch | Fold 1–4 train, fold 5 holdout | Fold 1–4 train, fold 5 holdout |
| Epoch tối đa / patience | 5 / 4 | 5 / 2 |
| Seeds | 42, 52, 62, 72, 82 | 42, 52, 62, 72, 82 |
| Loss | Mean answer-token NLL | Class-weighted mean answer-token NLL |
| Full retrain | Adapter mới, đủ 5 fold, epoch riêng từng seed | Tương tự A |
| Ensemble | Majority vote 5 seed | Mean candidate probabilities 5 seed |

AdamW, weight decay 0.01, warmup 10%, linear decay, batch 4/GPU, accumulation 2,
gradient clipping 1.0, standard LoRA không quantization và logic resume giữ nguyên.
Prompt, nhãn hợp lệ, cách cắt đầu/cuối hai view và source dữ liệu đánh giá cũng
giữ nguyên; tokenization và số token giữ được phụ thuộc tokenizer Sarvam.

Kiểm tra bằng tokenizer thật phát hiện việc decode một lát token có thể cắt
giữa byte UTF-8 của emoji, làm xuất hiện ký tự thay thế rồi vượt ngân sách.
Khangtest4 sửa bước serialize phần đầu/cuối: fast tokenizer dùng character
offsets để cắt trực tiếp câu nguồn tại ranh giới Unicode, rồi đo lại toàn bộ chat.
Giới hạn 256 token và mục tiêu giữ đầu/cuối hai view vẫn giữ nguyên. Resume signature
ghi `truncation_serialization=unicode-safe-offsets-v1` để không dùng lại state của
cách serialize cũ. Khi mô tả so sánh backbone, ghi nhận khác biệt tương thích tokenizer này.

Không có thay đổi tự động nào với CSV train. Hai file open validation được dùng
trực tiếp từ `data/open_validation/` trong repo: A 806 và B 395 câu có gold.
Task A vẫn loại các ID trùng open validation trước selection và retrain, giống
Khangtest3. Với bộ cũ A 6.588 dòng thì chỉ 71 ID open bị loại, còn 6.517 dòng;
điều này không đồng nghĩa đã loại đủ 142 câu trùng cả open/private. Muốn chạy trên
pool A 6.446 sạch theo ID chính thức, dùng bundle `official_split` đã chuẩn bị,
và dùng cùng bundle trên Khangtest3 để so sánh backbone công bằng.

## Chạy Kaggle

Clone vào notebook/môi trường mới:

```python
!git clone --single-branch --branch Khangtest4 https://github.com/POG42069/Hastika_Icon_2026_Viper.git
%cd Hastika_Icon_2026_Viper
!pip install -r requirements.txt

%env HASTIKA_DATA_DIR=/kaggle/input/YOUR_DATASET/official_split
%env HASTIKA_OUTPUT_DIR=/kaggle/working/khangtest4_sarvam/outputs
%env HASTIKA_CHECKPOINT_DIR=/kaggle/working/khangtest4_sarvam/checkpoints

!python -c "from llm_config import LLM_TRAINING, LLM_TRAINING_B; print(LLM_TRAINING.model_name, LLM_TRAINING.model_revision); print('LR A/B:', LLM_TRAINING.learning_rate, LLM_TRAINING_B.learning_rate)"
!python scripts/check_sarvam_inputs.py --data-dir "$HASTIKA_DATA_DIR"
!python Train_A_LLM.py --no-auto-resume
!python Train_B_LLM.py --no-auto-resume
```

Thay đường dẫn dataset thực tế. Nếu dùng lại notebook đã đặt `HASTIKA_LLM_MODEL`
thành Qwen thì bỏ biến đó; biến môi trường vẫn có thể ghi đè model mặc định.
Không cần truyền `--model` hoặc `--revision` khi chạy mặc định Khangtest4.

Checkpoint/output của nhánh này:

```text
task_a_sarvam_1_lora_khangtest4_bilingual_english_first_256
task_b_sarvam_1_lora_khangtest4_bilingual_english_first_256
```

Model/revision nằm trong resume signature; adapter Qwen không tương thích với
run Sarvam. Bắt đầu bằng thư mục output/checkpoint mới. `--no-auto-resume` chỉ
tắt quét Kaggle inputs, không vô hiệu hóa artifacts sẵn trong thư mục hiện tại.

## Kiểm tra

```bash
python -m unittest discover -s tests -v
python scripts/check_sarvam_inputs.py --data-dir /path/to/official_split --report /path/to/sarvam_check.json
```

Unit tests gồm cả loss/backward trên Llama nhỏ với LoRA (kiến trúc Sarvam) và
các test selection/retrain/ensemble/resume của pipeline. Script preflight dùng
tokenizer Sarvam thật trên mọi câu train/open, kiểm tra thứ tự view, giữ label
và EOS, mask loss, giới hạn 256 token và gold B theo ID. Nó chỉ tải config và
tokenizer, không tải trọng số Sarvam và không train trên GPU.

Đã kiểm tra ngày 2026-10-29: 96 unit tests pass. Preflight trên bundle
`official_split` pass với A 6.446 train / 806 open validation và B 3.155 train /
395 open validation; không trùng ID train/open và mọi full chat tối đa 256 token.

Kết quả kiểm tra này không phải kết quả macro-F1 và chưa xác nhận VRAM/thời gian
của full GPU run. Lưu `run_summary.json`, prediction và metrics từng seed để
điền bảng backbone comparison trong paper sau khi thực nghiệm hoàn tất.
