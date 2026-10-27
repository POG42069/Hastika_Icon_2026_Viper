# Task B: có và không có bản dịch tiếng Anh

Hai branch được đồng bộ source từ `Nhotest` commit
`db8d017961886ab32ed3e2bd177a1977224c1e06`, rồi đặt hai mặc định cho prompt Task B:

| Branch | Nội dung prompt Task B |
|---|---|
| `Khangtest2` | Chỉ `OriginalComment`: câu gốc code-mixed |
| `Khangtest3` | `Comment`: bản dịch tiếng Anh, rồi `OriginalComment`: câu gốc |

Hai branch dùng chung code; chỉ giá trị mặc định
`TASK_B_INCLUDE_ENGLISH_TRANSLATION` trong `llm_config.py` khác nhau.
Task A giữ nguyên cấu hình và prompt của Nhotest trên cả hai branch.

## Cấu hình chung

- Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- LoRA rank 32, alpha 64, dropout 0.05, target `all-linear`.
- Learning rate `1e-4`, tối đa 5 epoch, patience 2, tổng chat tối đa 256 token.
- 5 seed `42, 52, 62, 72, 82`.
- Không augmentation, không cap Gender; giữ class-weighted loss.
- Chọn best epoch riêng từng seed: train fold 1–4, validation fold 5;
  ưu tiên macro-F1 cao hơn, rồi validation loss thấp hơn.
- Khởi tạo lại model để retrain cả 5 fold theo best epoch của seed đó.
- Ensemble bằng trung bình xác suất 5 model rồi argmax.
- System prompt, nhãn trả lời và cách chấm điểm nhãn giống nhau giữa hai phiên bản.

## Dữ liệu và prompt

Cả hai branch tiếp tục đọc **cùng bộ CSV song ngữ**, cùng ID, nhãn và fold.
`Comment` là tiếng Anh, `OriginalComment` là câu gốc; cả hai cột vẫn cần có trong CSV.
`input_mode=bilingual` chọn định dạng dữ liệu này. Cờ cấu hình
`include_english_translation` quyết định có đưa cột `Comment` vào prompt hay không.
Không cần xóa cột hay tạo lại dataset để chạy Khangtest2.

Khangtest2:

```text
Comment (raw code-mixed):
<OriginalComment>
```

Khangtest3:

```text
English translation:
<Comment>

Comment (raw code-mixed):
<OriginalComment>
```

Quy tắc này áp dụng cho train chọn epoch, validation fold 5, full retrain,
development có gold và prediction. Khangtest2 không dùng bản dịch để tính
ngân sách token; câu gốc được dùng toàn bộ phần token dành cho nội dung.
Nếu dài quá, code giữ đầu/cuối câu gốc và dấu `[...]`. Khangtest3 giữ cách chia
ngân sách cho hai phiên bản của Nhotest. Nhãn và phần kết thúc chat luôn đầy đủ.

## Chạy trên Kaggle

Clone branch tương ứng vào một notebook riêng, cài `requirements.txt`, rồi chạy:

```python
%env HASTIKA_DATA_DIR=/kaggle/input/your-dataset/hastika_bilingual_no_augments
!python Train_B_LLM.py
```

Không truyền `--input-mode english`: đó là chế độ chỉ tiếng Anh cũ, không phải
thí nghiệm câu gốc. Hai branch đã có mặc định phù hợp, không cần cờ bổ sung.
Dùng cùng dataset và cùng các tùy chọn CLI cho hai lần chạy.

Checkpoint và output Task B được tách riêng:

- Khangtest2: `task_b_qwen3_4b_2507_lora_khangtest2_original_only_256`.
- Khangtest3: `task_b_qwen3_4b_2507_lora_khangtest3_bilingual_english_first_256`.

Chữ ký resume phân biệt hai loại prompt, ngăn dùng nhầm adapter của thí nghiệm kia.
Giữ cả `outputs/` và `checkpoints/` để resume lần chạy cùng phiên bản.
`run_summary.json` ghi `prompt_input.include_english_translation` để kiểm tra preset.
Kết quả chính vẫn là `predictions.csv`, `task_b_submission.zip` và
`ensemble_predictions.csv`. Cách cung cấp gold và đọc metrics xem
`TASK_B_KHANGTEST.md`.

## Kiểm tra

```bash
python -m unittest discover -s tests -v
```

Các test kiểm tra đúng nguồn câu gốc, không lẫn bản dịch, cắt câu dài, mask loss,
resume và cả hai giai đoạn của pipeline. Chất lượng macro-F1 cần được đo bằng
hai lần huấn luyện trên GPU với cùng điều kiện.
