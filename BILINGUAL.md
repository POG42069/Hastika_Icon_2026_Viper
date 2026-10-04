# Nhotest — original code-mixed + English translation

Code được sửa từ checkout Nhotest, không dùng code Sarvam. Backbone được chuyển sang
`Qwen/Qwen3-4B-Instruct-2507`, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. Đây là bản Instruct non-thinking,
dùng chat template chính thức và chỉ chấm điểm các nhãn hợp lệ. Giới hạn toàn bộ
chat mặc định là 256 token cho cả Task A và Task B. Task A giữ cấu hình
và luồng huấn luyện của Nhotest, 5 seed
`42, 52, 62, 72, 82`. Task B được chuyển sang luồng của Khangtest: train đủ
5 fold trong 5 epoch/seed, learning rate `1e-4`, class-weighted loss và soft
ensemble; bỏ cap Gender. Xem [TASK_B_KHANGTEST.md](TASK_B_KHANGTEST.md).

## Dữ liệu

Dùng `hastika_bilingual_no_augments.zip`, được tạo trực tiếp từ ZIP English
original-only bạn chọn. Cả 16 CSV giữ nguyên thứ tự dòng, ID, nhãn, bản dịch
và mọi ô dữ liệu cũ. Chỉ thêm `OriginalComment` từ câu code-mixed nguyên văn
của cùng ID; không áp dụng cách ly, chỉnh bản dịch hay augmentation của lần sửa
Task A trước.

- `Comment`: bản dịch tiếng Anh đã có.
- `OriginalComment`: câu code-mixed ban đầu, giữ nguyên trong CSV.

Cả training, validation và đầu vào dự đoán đều có hai cột này. Khi nạp, cả hai
cột được chuẩn hóa nhẹ bằng quy tắc hiện có: ftfy, NFKC, HTML unescape và khoảng
trắng. Không chuẩn hóa lại các ô trong CSV. Bản dịch và câu gốc là một mẫu với
một nhãn, không phải hai mẫu riêng.

Loader yêu cầu đủ hai cột ở chế độ bilingual; không âm thầm bỏ qua bản gốc.
Một `HASTIKA_DATA_DIR` được chỉ định sẽ giới hạn việc tìm file trong đúng thư
mục đó, tránh ghép nhầm file của bộ dữ liệu cũ. ZIP này không có file compact
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

Task B đặt câu gốc trước, dưới tiêu đề `Comment (raw code-mixed):`, rồi tới
`English translation:`. System prompt yêu cầu phân loại cùng một comment từ cả
hai phiên bản và chỉ
trả một nhãn hợp lệ. Loss vẫn chỉ tính trên nhãn trả lời; không huấn luyện thêm
một nhiệm vụ sinh bản dịch. Cả selection, full retrain và inference dùng cùng
cách tạo prompt.

Nếu prompt vượt 256 token, code chia chỗ cho cả hai phiên bản và giữ phần đầu
của mỗi phiên bản. Bản ngắn được giữ nguyên nếu đủ chỗ. Nhãn và
phần kết thúc câu trả lời luôn được giữ đầy đủ; không bỏ hoàn toàn một phiên
bản. Việc lược vẫn có thể mất chi tiết ở cuối câu. Có thể truyền
`--max-length 512` để đổi độ dài. Checkpoint dùng độ dài khác hoặc cách giữ
đầu/cuối cũ không khớp chữ ký resume, nên cần huấn luyện lại với cấu hình mới.

Output/checkpoint dùng thư mục `task_a_qwen3_4b_2507_lora_bilingual` và
`task_b_qwen3_4b_2507_lora_bilingual_fixed_soft` để giữ lại kết quả cũ. Resume kiểm tra
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

Task A dùng đầy đủ 1.317 câu fold 5. Task B train cả 3.155 câu của 5 fold;
395 câu development được báo điểm riêng nếu có file nhãn gold. Không dùng quy tắc
lọc xuống 1.302 câu của bản Task A repair trước. Các kiểm tra CPU và tokenizer
xác nhận pipeline đúng dữ liệu, không xác nhận macro-F1 tốt hơn; cần chạy lại
backbone trên GPU để đánh giá chất lượng.
