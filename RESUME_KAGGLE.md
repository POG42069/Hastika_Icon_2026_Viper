# Tiếp tục Task A sau khi Kaggle dừng ở 12 giờ

## Trạng thái của `results-7.zip`

ZIP được tạo từ `Nhotest` commit `911bef44cebdf31c500f503648a56b151148e69b`.
Seed 42, 52, 62, 72 đã xong selection; `best_epoch` lần lượt là **8, 9, 3,
10**. Seed 82 mới xong epoch 2/10. Cả năm seed đều có adapter tốt nhất tính
đến lúc dừng, nhưng ZIP **không có optimizer/scheduler state**. Vì vậy không
thể tiếp tục chính xác từ epoch 3 của seed 82.

Pipeline gốc khởi tạo model mới ở giai đoạn full-data retrain. Do đó các seed
42/52/62/72 chỉ cần `seed_summary.json` cùng dữ liệu đúng phiên bản để chạy
giai đoạn này; không cần tải lại 1.4 GB adapter selection. Script sẽ so sánh
ID và thứ tự của toàn bộ 8.592 dòng train với `split_manifest.csv` cũ trước
khi dùng `best_epoch`.

File nhỏ `task_a_resume_metadata.zip` đã được tạo từ `results-7.zip`, chứa
selection summary, history và manifest cần thiết. Nó không chứa model weights.

## Chuẩn bị trên Kaggle

Tạo một Dataset riêng cho mỗi file sau rồi gắn vào notebook:

1. `Hastika_Icon_2026_Viper_resume_code.zip`: mã nguồn từ nhánh làm việc
   `codex/kaggle-resume`, dựa trên `Nhotest`.
2. `task_a_resume_metadata.zip`: metadata rút từ `results-7.zip`.
3. Dataset train gốc có `binary_original_1..5.csv`, `binary_augment1..5.csv`,
   `binary_validation_inputs.csv` giống lần chạy cũ.

Trong notebook, giải nén mã vào `/kaggle/working`, cài dependencies rồi đặt
đường dẫn dataset. Thay các tên Dataset trong ví dụ bằng tên Kaggle thực tế:

```python
%cd /kaggle/working
!unzip -q /kaggle/input/hastika-resume-code/Hastika_Icon_2026_Viper_resume_code.zip
%cd /kaggle/working/Hastika_Icon_2026_Viper_resume
!pip install -q -r requirements.txt
%env HASTIKA_DATA_DIR=/kaggle/input/hastika-training-data
```

Nếu dữ liệu nằm trong một thư mục con, đặt `HASTIKA_DATA_DIR` đến đúng thư
mục chứa các CSV. Giữ model/revision mặc định như lần chạy trước.

## Chạy từng Save Version

Mỗi version chỉ xử lý **một seed**. Với seed 42, dùng:

```python
!python Resume_A_LLM.py train-seed \
  --archive /kaggle/input/hastika-resume-metadata/task_a_resume_metadata.zip \
  --seed 42
```

Lặp lại bằng `--seed 52`, `--seed 62`, `--seed 72`, mỗi seed trong một Save &
Run All riêng. Chúng bỏ qua selection đã hoàn tất và retrain trên toàn bộ dữ
liệu theo `best_epoch` của seed đó. Output nằm ở
`/kaggle/working/hastika_resume/seed_<N>/predictions.csv`; adapter mới ở
`/kaggle/working/hastika_resume_checkpoints/seed_<N>/final_full_data_adapter/`.

Seed 82 thiếu kết quả selection hoàn chỉnh nên chạy lại riêng:

```python
!python Resume_A_LLM.py redo-seed-82
```

Lệnh này dùng pipeline gốc để chọn epoch từ đầu rồi full-data retrain seed 82;
output nằm ở `/kaggle/working/hastika_resume/seed_82/`. Đây là cách giữ nguyên
quy trình hai giai đoạn. Adapter epoch 2 trong ZIP vẫn hữu ích cho phân tích
hoặc inference dự phòng, nhưng không thể thay thế optimizer state đã mất.

Sau mỗi lần **Save & Run All thành công**, trong notebook tiếp theo dùng
**Add Input → Notebook Output Files** để gắn output version đó. Kaggle lưu
file trong `/kaggle/working` của một version hoàn tất để tái sử dụng làm input.

## Gộp năm seed

Tạo một notebook cuối, gắn output của cả năm version. Tìm các thư mục seed
trong `/kaggle/input` và truyền từng thư mục vào lệnh sau:

```bash
python Resume_A_LLM.py ensemble \
  --seed-dir /kaggle/input/<output-42>/hastika_resume/seed_42 \
  --seed-dir /kaggle/input/<output-52>/hastika_resume/seed_52 \
  --seed-dir /kaggle/input/<output-62>/hastika_resume/seed_62 \
  --seed-dir /kaggle/input/<output-72>/hastika_resume/seed_72 \
  --seed-dir /kaggle/input/<output-82>/hastika_resume/seed_82 \
  --output-dir /kaggle/working/hastika_final
```

Kết quả là `hastika_final/predictions.csv` và `task_a_submission.zip`. Lệnh
kiểm tra đủ năm seed, cùng model/revision, ID khớp và nhãn hợp lệ trước khi
majority vote. Bước này chỉ dùng Python standard library, có thể chạy CPU.

## Kiểm tra archive ở máy cá nhân

```bash
python3 Resume_A_LLM.py inspect --archive /duong/dan/results-7.zip
python3 Resume_A_LLM.py metadata \
  --archive /duong/dan/results-7.zip \
  --output /duong/dan/task_a_resume_metadata.zip
```
