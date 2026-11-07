# Đồ án video-only RVQ-VAE

Pipeline sử dụng duy nhất dữ liệu ViTPose của bài báo:

`250 x 30 pose -> position + velocity -> RVQ-VAE -> motion tokens -> TF-IDF -> Logistic Regression/MLP`

Mỗi fold giữ nguyên test set theo trẻ của tác giả. RVQ-VAE, TF-IDF và classifier đều chỉ học từ training set của fold tương ứng.

## Chạy trên Kaggle

1. Thêm dataset `xndung/fidgety` vào notebook.
2. Bật GPU trong `Settings > Accelerator`.
3. Đưa thư mục `đồ án` lên repository được cấu hình trong notebook, hoặc upload thư mục thành một Kaggle Dataset và đặt `UPLOADED_PROJECT_DIR`.
4. Upload và mở `train_kaggle_video_rvq.ipynb`.
5. Giữ `RUN_MODE = "smoke"` để kiểm tra một fold.
6. Sau khi smoke test thành công, đổi thành `RUN_MODE = "full"` và chạy lại toàn bộ notebook.
7. Full mode dùng bộ lập lịch động với tối đa hai GPU; GPU hoàn thành trước sẽ nhận fold tiếp theo.

Kết quả nằm trong `/kaggle/working/video_rvq_full` và file nén `/kaggle/working/video_rvq_full.zip`.

## Chạy bằng dòng lệnh

```powershell
python run.py --data-path D:\src\data.zip --output-dir outputs --folds 1 --epochs 5
```

Với dữ liệu Kaggle:

```bash
python run.py --data-path /kaggle/input/datasets/xndung/fidgety/data --output-dir /kaggle/working/video_rvq_full
```

Chạy song song trên hai GPU:

```bash
python run_multi_gpu.py --data-path /kaggle/input/datasets/xndung/fidgety/data --output-dir /kaggle/working/video_rvq_full --gpu-workers 2
```

## Kết quả

- `fold_results.csv`: sensitivity, specificity và balanced accuracy của từng fold.
- `summary.csv`: trung bình và CI 95% qua các fold hoàn thành.
- `fold_N/rvqvae.pt`: RVQ-VAE của fold.
- `fold_N/tfidf.pkl`: TF-IDF chỉ fit trên training set.
- `fold_N/*_predictions.csv`: nhãn thật và dự đoán.
- `fold_N/codebook_statistics.json`: số code được sử dụng và perplexity.

Baseline video công bố của bài báo là balanced accuracy `90.6606%`. Chỉ so sánh chính thức sau khi hoàn thành đủ 9 fold; kết quả smoke test không dùng để kết luận.
