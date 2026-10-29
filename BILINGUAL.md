# Khangtest5 — Qwen2.5-3B-Instruct, original code-mixed + English translation

Nhánh này bắt đầu từ Khangtest4 `177a126` và đổi cả hai task sang Qwen2.5-3B-Instruct.
Xem [QWEN25_EXPERIMENT.md](QWEN25_EXPERIMENT.md) cho lệnh chạy và kiểm tra tương thích.

Trên hai branch thí nghiệm, Task B dùng câu gốc trên `Khangtest2`, hoặc tiếng Anh
cùng câu gốc trên `Khangtest3`. Cả hai vẫn đọc cùng CSV song ngữ. Xem
[TASK_B_TRANSLATION_EXPERIMENT.md](TASK_B_TRANSLATION_EXPERIMENT.md) cho mặc định,
prompt và thư mục kết quả của từng branch. Nội dung dưới đây mô tả nền Nhotest.

Code giữ pipeline Nhotest/Khangtest3. Backbone dùng
`Qwen/Qwen2.5-3B-Instruct`, revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. Đây là checkpoint Instruct;
chat template đi kèm tokenizer được dùng làm format supervised và chỉ chấm
điểm các nhãn hợp lệ. Giới hạn toàn bộ
chat mặc định là 256 token cho cả Task A và Task B. Task A giữ optimization
và hai giai đoạn huấn luyện của Nhotest, 5 seed
`42, 52, 62, 72, 82`. Task B giữ learning rate `1e-4`, class-weighted loss và soft ensemble từ Khangtest;
bỏ cap Gender. Chọn best epoch cho từng seed trên fold 1–4/validation fold 5
(tối đa 5 epoch, patience 2), rồi khởi tạo model mới để retrain cả 5 fold
theo best epoch riêng. Xem [TASK_B_KHANGTEST.md](TASK_B_KHANGTEST.md).

Task A hiện chọn chỉ câu gốc trên Khangtest2 hoặc song ngữ trên Khangtest3;
inference trên 806 câu open đi kèm repo và loại ID open khỏi train. Xem
[TASK_A_TRANSLATION_EXPERIMENT.md](TASK_A_TRANSLATION_EXPERIMENT.md).

## Dữ liệu

Dùng `hastika_bilingual_no_augments.zip`, được tạo trực tiếp từ ZIP English
original-only bạn chọn. Các CSV train giữ nguyên ID, nhãn và bản dịch, có thêm
`OriginalComment` từ câu code-mixed nguyên văn của cùng ID. Hai file đầu vào
dự đoán trong ZIP vẫn là raw-only; Task B dùng bộ open validation đi kèm repo
như mô tả bên dưới. Không áp dụng cách ly hay augmentation của lần sửa Task A trước.

- `Comment`: bản dịch tiếng Anh đã có.
- `OriginalComment`: câu code-mixed ban đầu, giữ nguyên trong CSV.

Các CSV train có hai cột này. Hai file `*_validation_inputs.csv` trong ZIP được
kiểm tra lại có `id,Comment` chứa câu gốc, chưa có cột song ngữ. Trên hai branch
thí nghiệm, riêng Task B dùng file open validation 395 câu có sẵn trong repo:
`data/open_validation/task_b_open_test_translated_labeled.csv`, đủ cả hai cột và gold.
Khi nạp dữ liệu song ngữ, cả hai
cột được chuẩn hóa nhẹ bằng quy tắc hiện có: ftfy, NFKC, HTML unescape và khoảng
trắng. Không chuẩn hóa lại các ô trong CSV. Bản dịch và câu gốc là một mẫu với
một nhãn, không phải hai mẫu riêng.

Loader yêu cầu đủ hai cột ở chế độ bilingual; không âm thầm bỏ qua bản gốc.
Một `HASTIKA_DATA_DIR` được chỉ định sẽ giới hạn việc tìm file train trong đúng thư
mục đó, tránh ghép nhầm file của bộ dữ liệu cũ. File open validation Task B được
đọc riêng từ repo. ZIP này không có file compact
1–4, nên code đọc đúng năm file original của mỗi task. Augmentation đã tắt.

## Chạy

Giải nén ZIP dữ liệu. Trỏ tới thư mục trực tiếp chứa các CSV:

```bash
export HASTIKA_DATA_DIR=/absolute/path/hastika_bilingual_no_augments
python Train_A_LLM.py --input-mode bilingual --no-auto-resume
python Train_B_LLM.py --input-mode bilingual --no-auto-resume
```

Trong Kaggle, thay đường dẫn bằng thư mục của Dataset mới. Cài dependencies
bằng requirements.txt như trước. Các tùy chọn model/epoch/batch đã dùng ở lần
chạy tốt nhất cần tiếp tục truyền đúng giá trị nếu khác cấu hình hiện có.

Prompt Task A có dạng:

```text
English translation:
<Comment>
Original code-mixed comment:
<OriginalComment>
```

Task B cố định `English translation:` trước, rồi tới câu gốc dưới tiêu đề
`Comment (raw code-mixed):`. Cờ `--prompt-order original-first` bị từ chối cho Task B.
System prompt yêu cầu phân loại cùng một comment từ cả
hai phiên bản và chỉ
trả một nhãn hợp lệ. Loss vẫn chỉ tính trên nhãn trả lời; không huấn luyện thêm
một nhiệm vụ sinh bản dịch. Cả selection, full retrain và inference dùng cùng
cách tạo prompt.

Nếu prompt vượt giới hạn của task, code chia chỗ cho cả hai phiên bản, giữ phần đầu/cuối
và đánh dấu `[...]` nơi bị lược. Bản ngắn được giữ nguyên nếu đủ chỗ. Nhãn và
phần kết thúc câu trả lời luôn được giữ đầy đủ; không bỏ hoàn toàn một phiên
bản. Việc lược vẫn có thể mất chi tiết ở giữa câu. Mặc định cả hai task là 256;
có thể truyền `--max-length` để đổi độ dài. Checkpoint huấn luyện ở 256
không khớp chữ ký resume của cấu hình 512, nên cần huấn luyện lại ở độ dài mới.

Output/checkpoint của cả hai task tách theo branch như hai tài liệu thí nghiệm
Task A và Task B; kết quả cũ được giữ lại. Resume kiểm tra
model, revision, input mode, cột nguồn, phiên bản prompt và dữ liệu; checkpoint English-only
hoặc legacy thiếu thông tin tương thích không được dùng lại cho bilingual.

## So sánh

Để thử English-only trên cùng các ID/nhãn và validation, dùng
`--input-mode english`. Chọn thư mục output/checkpoint riêng cho hai lần:

```bash
HASTIKA_OUTPUT_DIR=/absolute/path/results/english \
HASTIKA_CHECKPOINT_DIR=/absolute/path/checkpoints/english \
python Train_A_LLM.py --input-mode english --no-auto-resume

HASTIKA_OUTPUT_DIR=/absolute/path/results/bilingual \
HASTIKA_CHECKPOINT_DIR=/absolute/path/checkpoints/bilingual \
python Train_A_LLM.py --input-mode bilingual --no-auto-resume
```

Task A dùng fold 5 sau khi loại các ID open validation khỏi các fold. Task B chọn epoch trên 2.524 câu
fold 1–4 và 631 câu validation fold 5, rồi retrain cả 3.155 câu của 5 fold;
395 câu development đi kèm repo được báo điểm riêng. Không dùng quy tắc
lọc xuống 1.302 câu của bản Task A repair trước. Các kiểm tra CPU và tokenizer
xác nhận pipeline đúng dữ liệu, không xác nhận macro-F1 tốt hơn; cần chạy lại
backbone trên GPU để đánh giá chất lượng.
