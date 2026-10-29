# Khangtest5: Qwen2.5-3B-Instruct trên Task A và B

Nhánh tạo từ **Khangtest4 `177a126a1d434cc1ebc909340135e911cb55f32a`**,
đổi model và tokenizer sang [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
Revision được ghim: `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
Model config dùng kiến trúc `Qwen2ForCausalLM`. Pipeline dùng chat template đi
kèm checkpoint để serialize system/user/assistant, rồi train và chấm điểm
completion của từng nhãn hợp lệ.

## Cấu hình thực nghiệm

| Setting | Task A | Task B |
|---|---|---|
| Input | English trước, original sau | English trước, original sau |
| Max toàn bộ chat, gồm answer và EOS | 256 token | 256 token |
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 | 32 / 64 / 0.05 |
| Target modules | all-linear | all-linear |
| Learning rate | 2e-4 | 1e-4 |
| Epoch tối đa / patience | 5 / 4 | 5 / 2 |
| Seeds | 42, 52, 62, 72, 82 | 42, 52, 62, 72, 82 |
| Chọn epoch từng seed | Fold 1–4 train, fold 5 holdout | Tương tự A |
| Full retrain | Adapter mới, đủ 5 fold, theo best epoch từng seed | Tương tự A |
| Loss | Mean answer-token NLL | Class-weighted mean answer-token NLL |
| Ensemble | Majority vote 5 seed | Mean candidate probabilities 5 seed |

Giữ các setting từ Khangtest4: AdamW, weight decay 0.01, warmup 10%, linear
decay, batch 4/GPU, accumulation 2, gradient clipping 1.0, LoRA không
quantization. Giữ system prompt, nhãn, cách chia ngân sách giữa hai view và
cách cắt đầu/cuối theo Unicode offsets đã sửa ở Khangtest4. Tokenizer và chat
template khác nhau nên lượng nội dung được giữ trong 256 token có thể khác nhau.

## Dữ liệu

Không sửa CSV train hoặc logic chọn dữ liệu. Đặt `HASTIKA_DATA_DIR` tới cùng bộ
dữ liệu bạn dùng cho các backbone khác. Hai open validation có nhãn được đọc từ
`data/open_validation/` trong repo: A 806 và B 395 câu. Task A vẫn loại ID trùng
open validation trước selection và full retrain. Các file private test trong
dataset không được dùng làm open validation mặc định.

Để chạy trên bộ đã loại 142 ID overlap, dùng bundle `official_split` đã chuẩn bị:
A 6.446 câu train và B 3.155 câu train. Nhánh này không tự động làm sạch bộ cũ
A 6.588 dòng; bộ cũ vẫn chỉ bị loại 71 ID open theo logic Khangtest3/4. Dùng
cùng bundle dữ liệu khi so sánh các model.

## Chạy Kaggle

Dùng notebook/môi trường mới và sửa đường dẫn dataset cho đúng:

```python
!git clone --single-branch --branch Khangtest5 https://github.com/POG42069/Hastika_Icon_2026_Viper.git
%cd Hastika_Icon_2026_Viper
!pip install -r requirements.txt

import os
os.environ.pop("HASTIKA_LLM_MODEL", None)
os.environ["HASTIKA_DATA_DIR"] = "/kaggle/input/YOUR_DATASET/official_split"
os.environ["HASTIKA_OUTPUT_DIR"] = "/kaggle/working/khangtest5_qwen25/outputs"
os.environ["HASTIKA_CHECKPOINT_DIR"] = "/kaggle/working/khangtest5_qwen25/checkpoints"

!python -c "from llm_config import LLM_TRAINING, LLM_TRAINING_B; print(LLM_TRAINING.model_name, LLM_TRAINING.model_revision); print('LR A/B:', LLM_TRAINING.learning_rate, LLM_TRAINING_B.learning_rate)"
!python scripts/check_qwen25_inputs.py --data-dir "$HASTIKA_DATA_DIR"
!python Train_A_LLM.py --no-auto-resume
!python Train_B_LLM.py --no-auto-resume
```

Không cần truyền `--model` hay `--revision`. Biến `HASTIKA_LLM_MODEL`, nếu đã đặt
từ notebook cũ, có thể ghi đè model mặc định; cell trên xóa override đó.

Checkpoint/output riêng:

```text
task_a_qwen2_5_3b_instruct_lora_khangtest5_bilingual_english_first_256
task_b_qwen2_5_3b_instruct_lora_khangtest5_bilingual_english_first_256
```

Model và revision nằm trong resume signature. Bắt đầu bằng output/checkpoint
mới; adapter của Sarvam-1 hay Qwen3 không dùng cho Qwen2.5. `--no-auto-resume`
tắt quét Kaggle inputs, nhưng không tắt resume từ thư mục hiện tại nếu đã có
artifacts tương thích.

## Kiểm tra trước khi train

```bash
python -m unittest discover -s tests -v
python scripts/check_qwen25_inputs.py --data-dir /path/to/official_split --report /path/to/qwen25_check.json
```

Unit tests kiểm tra loss/backward với Qwen2 nhỏ và LoRA, masking, selection,
retrain, ensemble và resume. Preflight dùng tokenizer Qwen2.5 thật trên mọi câu
train/open để kiểm tra giới hạn 256 token, thứ tự view, answer/EOS, mask loss,
ID train/open và gold Task B theo ID. Script chỉ tải config/tokenizer, không
tải trọng số model hoặc chạy training GPU.

Đã kiểm tra ngày 2026-10-29: 96 unit tests pass. Preflight trên bundle
`official_split` pass với A 6.446 train / 806 open validation và B 3.155 train /
395 open validation; không trùng ID train/open và mọi full chat tối đa 256 token.
Đối chiếu cấu hình với Khangtest4 xác nhận chỉ đổi model/revision và tên thư mục
checkpoint/output; các setting train và dữ liệu giữ nguyên.

Các kiểm tra này không xác nhận điểm macro-F1, thời gian chạy hay VRAM của full
training. Điểm thực nghiệm cần lấy từ metrics và prediction sau khi train xong.
