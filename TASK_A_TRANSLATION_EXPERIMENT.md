# Task A: câu gốc và câu gốc kèm bản dịch tiếng Anh

Trên `Khangtest5`, Task A dùng Qwen2.5-3B-Instruct và giữ chế độ song ngữ của Khangtest3.
Xem [QWEN25_EXPERIMENT.md](QWEN25_EXPERIMENT.md) cho preset/thư mục kết quả hiện tại.
Phần dưới mô tả các đối chứng Qwen3 trên Khangtest2/Khangtest3 trước đó.

| Branch | Prompt Task A |
|---|---|
| `Khangtest2` | Chỉ `OriginalComment`: câu gốc code-mixed |
| `Khangtest3` | `Comment`: tiếng Anh, rồi `OriginalComment`: câu gốc |

Hai branch dùng chung code, system prompt và nhãn `Non-Hate` / `Hate`.
`TASK_A_INCLUDE_ENGLISH_TRANSLATION` trong `llm_config.py` chọn phiên bản.
Thứ tự song ngữ cố định là tiếng Anh trước. Train, validation chọn epoch,
full retrain và inference đều dùng cùng chính sách của branch.
Task B giữ nguyên hai phiên bản đã triển khai trước đó.

## Huấn luyện

- Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- LoRA rank 32, alpha 64, dropout 0.05, target `all-linear`.
- Learning rate `2e-4`, tối đa 5 epoch, patience 4, chat tối đa 256 token.
- 5 seed `42, 52, 62, 72, 82`; không augmentation, không class weight.
- Chọn best epoch của từng seed bằng macro-F1 fold 5, loss phá hòa;
  các fold 1–4 dùng để train chọn epoch.
- Khởi tạo model mới cho từng seed và retrain đủ dữ liệu theo best epoch riêng.
- Mỗi model bỏ một phiếu; nhãn cuối là majority vote của 5 model.

Khangtest2 không dành token cho tiếng Anh. Khi câu gốc dài quá, code giữ đầu/cuối
và dấu `[...]`. Khangtest3 chia ngân sách cho hai phiên bản theo cơ chế hiện có.
Nhãn assistant được giữ đầy đủ, loss không tính trên prompt.

## Dữ liệu và loại ID đánh giá khỏi train

`HASTIKA_DATA_DIR` tiếp tục chỉ tới thư mục chứa `binary_original_1.csv` tới
`binary_original_5.csv` (hoặc cặp compact 1–4/5). Các CSV train vẫn có
`Comment` tiếng Anh, `OriginalComment` câu gốc và `Label`.

Đầu vào inference và gold mặc định là file có sẵn trong repo:
`data/open_validation/task_a_open_test_translated_labeled.csv`.
Đây là **806 câu open validation** lấy từ `open_test_translated_labeled.zip`
bạn cung cấp. Không cần upload thêm dataset hoặc đổi tên CSV trong notebook.
File raw-only `binary_validation_inputs.csv` cũ không được chọn mặc định.

Trong ZIP bilingual đã kiểm tra, 71 ID của open validation cũng có trong các
fold train. Code loại các ID này khỏi **toàn bộ các fold**, trước cả chọn epoch
và full retrain, ở cả hai branch. Với ZIP đó, 6.588 dòng được nạp và còn 6.517
dòng sau khi loại trùng. Chỉ dùng ID để loại; không dùng nhãn open để cân bằng lớp.
Log và `training_exclusions.json` ghi số dòng, ID, fold và nhãn của các dòng bị loại.
Nếu dataset khác, số dòng loại được tính theo ID thực tế.

Chữ ký resume chứa chính sách prompt và fingerprint các ID loại khỏi train.
Adapter cũ chưa có chính sách này không được dùng nhầm cho thí nghiệm mới.
Đổi riêng file inference không làm đổi tập ID dùng để loại khỏi train.

## Chạy trên Kaggle

Trong notebook đã clone đúng branch, giữ `HASTIKA_DATA_DIR` trỏ tới dữ liệu train:

```python
!git pull --ff-only
!python Train_A_LLM.py
```

Không cần truyền cờ để chọn câu gốc/song ngữ. Không dùng `--input-mode english`
cho thí nghiệm này, vì cờ đó là chế độ chỉ tiếng Anh riêng.

## Kết quả

Output và checkpoint Task A tách riêng theo branch:

- Khangtest2: `task_a_qwen3_4b_2507_lora_khangtest2_original_only_256`.
- Khangtest3: `task_a_qwen3_4b_2507_lora_khangtest3_bilingual_english_first_256`.

Sau khi bỏ phiếu đủ 5 seed, log `Official validation | FINAL MAJORITY VOTE`
báo macro-F1, accuracy, precision/recall/F1 từng lớp trên **806 câu**. Điểm này
dùng nhãn vote cuối, không dùng argmax của xác suất trung bình.
Open validation không dùng để chọn best epoch hay cập nhật trọng số.

Các file chính: `predictions.csv`, `task_a_submission.zip`,
`voting_predictions.csv`, `official_validation_metrics.json`,
`official_validation_predictions.csv`, `training_exclusions.json`, `run_summary.json`.

Nếu chủ động dùng `--prediction-file /path/to/another.csv`, inference đọc file
đó. Nếu CSV có cột `Label` hợp lệ, code chấm chính tập đó; nếu không có nhãn,
code chỉ xuất prediction và báo rõ không có metric gold. Không lấy nhãn của
tập 806 câu để chấm cho các ID khác.
