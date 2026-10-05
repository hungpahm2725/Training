# HƯỚNG DẪN CHẠY CODE THEO THỨ TỰ
### Đề tài: Hệ Thống Nhận Diện Sâu Bệnh Trên Cây Đậu (DATT)

Mô hình sử dụng: **YOLO26-cls** (phân loại 5 lớp bệnh lá đậu)

| # | Lớp bệnh | Số ảnh |
|---|---|---|
| 0 | Angular Leaf Spot (đốm lá góc cạnh) | 635 |
| 1 | Anthracnose (thán thư) | 671 |
| 2 | Fresh Leaf (lá khỏe) | 620 |
| 3 | Mosaic Virus (virus khảm) | 629 |
| 4 | Rust (gỉ sắt) | 625 |
| | **Tổng** | **3.180** |

---

## CHUẨN BỊ (làm một lần duy nhất)

Mở PowerShell và trỏ vào thư mục `training`:

```powershell
cd d:\Hoc_Tap\DATT\training
```

Kiểm tra thư viện đã có chưa:

```powershell
python -c "import ultralytics, torch; print(ultralytics.__version__); print(torch.cuda.is_available())"
```

Kết quả đúng phải in ra số phiên bản ultralytics và chữ `True` (nghĩa là đã nhận GPU).
Nếu thiếu thư viện thì cài:

```powershell
pip install -r ..\backend\requirements.txt
```

---

## CÁCH 1: CHẠY MỘT LỆNH DUY NHẤT (khuyên dùng khi demo)

```powershell
python run_pipeline.py
```

Lệnh này tự động chạy lần lượt **cả 3 bước** bên dưới rồi copy mô hình sang
backend. Chạy thử nhanh cho quen (khoảng vài phút):

```powershell
python run_pipeline.py --quick
```

---

## CÁCH 2: CHẠY TỪNG BƯỚC ĐỂ KIỂM SOÁT KẾT QUẢ

### BƯỚC 1 — Chia dữ liệu

```powershell
python prepare_dataset.py
```

**Làm gì:** đọc 3.180 ảnh trong `dataset/Bean Leaf Disease Original Image/`,
chia thành 3 tập theo đúng tỉ lệ 70/15/15, mỗi lớp đều được chia đủ.

**Kết quả:** tạo thư mục `training/datasets/bean_disease/` gồm
`train/`, `val/`, `test/`.

**Lưu ý:** mặc định dùng *hardlink* nên không tốn thêm dung lượng đĩa
(không phải chờ copy 9GB). Nếu muốn bản sao độc lập hoàn toàn thì thêm `--copy`.

Các tùy chọn khác:

```powershell
python prepare_dataset.py --verify    # chỉ kiểm tra lại, không chia
python prepare_dataset.py --force     # xóa và chia lại từ đầu
```

**Kiểm tra đạt:** bảng in ra có đủ 5 lớp, mỗi lớp có số ở cả 3 cột train/val/test,
dòng cuối ghi `DỮ LIỆU HỢP LỆ`.

---

### BƯỚC 2 — Huấn luyện mô hình

```powershell
python train_classification.py
```

**Làm gì:** tải trọng số `yolo26n-cls.pt` đã học sẵn trên ImageNet, rồi huấn luyện
tiếp trên dữ liệu lá đậu của bạn. Chạy 80 epoch, ảnh đầu vào 320x320.

**Kết quả:**
- Thư mục `training/runs/bean_disease_yolo26_cls/` chứa trọng số và đồ thị
- **Tự động copy** `best.pt` sang `backend/weights/best.pt`

**Thời gian dự kiến:** khoảng 30 đến 60 phút trên RTX 5070.

Chạy thử trước cho chắc (10 epoch, model nhỏ nhất, vài phút):

```powershell
python train_classification.py --quick
```

Tùy chỉnh khi cần:

```powershell
python train_classification.py --batch 32            # nếu GPU còn dư RAM
python train_classification.py --epochs 100 --model yolo26s-cls.pt
python train_classification.py --resume              # chạy tiếp khi bị ngắt
python train_classification.py --device cpu          # nếu không có GPU (rất chậm)
```

**Kiểm tra đạt:** dòng cuối in ra `[OK] Đã copy best.pt sang ...`.
Nếu báo `Out Of Memory` thì giảm `--batch` xuống 8.

---

### BƯỚC 3 — Đánh giá trên tập test

```powershell
python evaluate_model.py --md
```

**Làm gì:** chạy mô hình trên tập `test` (khoảng 477 ảnh mô hình **chưa từng thấy**
lúc train) và tính các chỉ số khoa học.

**Kết quả:**
- In ra màn hình: Top-1 Accuracy, Precision/Recall/F1 từng lớp
- `training/reports/<tên model>/per_class_metrics.csv`
- `training/reports/<tên model>/confusion_matrix.csv`
- `training/reports/<tên model>/BAO_CAO_DANH_GIA.md` ← **dán thẳng vào đồ án**

**Kiểm tra đạt:** Top-1 Accuracy từ 90% trở lên là tốt.

**QUAN TRỌNG — đọc trước khi ghi số vào báo cáo:**

Nếu kết quả ra xấp xỉ 100%, **đừng vội mừng và đừng dán ngay vào đồ án**. Con số
đó gần như chắc chắn là sai, do lỗi chia dữ liệu chứ không phải mô hình giỏi.

Bằng chứng nằm ngay ở tên file. Ảnh trong dataset được chụp theo từng đợt liên
tiếp, tên file dạng `IMG_YYYYMMDD_HHMMSS`. Ví dụ với lớp Rust:

```
train/Rust/IMG_20251230_152003.jpg
train/Rust/IMG_20251230_152005.jpg     <- train
 test/Rust/IMG_20251230_152002.jpg     <- test
```

Hai ảnh cách nhau **3 giây**, cùng một buổi chụp, cùng một cây, cùng nền và
ánh sáng — nhưng một ảnh nằm ở tập huấn luyện, một ảnh nằm ở tập kiểm thử. Mô
hình chỉ cần nhớ "đặc trưng buổi chụp" là đoán đúng, chứ không học dấu hiệu
bệnh. Đây gọi là **rò rỉ dữ liệu** (data leakage).

**Cách lấy con số trung thực:** chia dữ liệu theo NHÓM ẢNH CÙNG ĐỢT CHỤP, để
toàn bộ ảnh của một buổi chụp chỉ nằm trong một tập.

```powershell
python group_split_evaluate.py
```

Script gộp các ảnh cách nhau không quá 120 giây thành một nhóm, chia nhóm
(70/15/15) thay vì chia từng ảnh, rồi chấm điểm trên bản chia sạch đó. Kết quả
lưu ở `training/reports/grouped_eval/per_class_metrics_grouped.csv`.

**Dùng con số nào khi bảo vệ đồ án?** Dùng con số của
`group_split_evaluate.py`. Nếu hội đồng hỏi vì sao thấp hơn, đó là câu trả lời
tốt: bạn hiểu vấn đề rò rỉ dữ liệu và đã xử lý đúng cách — điều này ghi điểm
cao hơn một con số 100% không giải thích được.

---

### BƯỚC 4 — Kiểm tra độ nhận diện ảnh (Giao diện đồ họa hoặc Dòng lệnh)

**Cách 1: Nhấp đúp chuột để mở giao diện kiểm tra trực quan (Khuyên dùng khi demo):**
- Nhấp đúp file **`TEST_NHAN_DIEN.bat`** trong thư mục `training/` hoặc **`TEST_NHAN_DIEN_LA_DAU.bat`** ngoài màn hình Desktop.
- Giao diện đồ họa sẽ mở lên:
  + Bấm **"📂 Chọn ảnh từ máy tính"** để chọn bất kỳ ảnh lá đậu nào.
  + Bấm **"🌿 Chọn ảnh mẫu có sẵn"** để thử nghiệm ngay 5 lớp bệnh mẫu trong bộ dữ liệu.
  + Bấm **"📁 Test cả thư mục"** để test hàng loạt và xem % độ nhận diện chuẩn xác.
  + Bấm **"💾 Lưu ảnh kết quả"** để lưu ảnh chẩn đoán kèm watermark kết quả vào thư mục `ket_qua_test/`.
  + Hiển thị đầy đủ: Tên bệnh tiếng Việt, Tên khoa học, Mức độ nguy hiểm, Độ nhận diện (>90%, chuẩn thực địa), Biểu đồ thanh xác suất Top-5 lớp, Triệu chứng lâm sàng và Phác đồ điều trị khuyến nông.

**Cách 2: Chạy kiểm tra bằng dòng lệnh:**
```powershell
python test_nhan_dien.py "duong\dan\anh-la-dau.jpg"
```
Test hàng loạt cả thư mục:
```powershell
python test_nhan_dien.py --dir "datasets\bean_disease\test\Rust"
```
Hoặc dùng script cũ:
```powershell
python predict_image.py "dataset\Bean Leaf Disease Original Image\Rust"
```

---

### BƯỚC 5 — Chạy web để nhận diện trên giao diện

Mở **hai** cửa sổ PowerShell riêng biệt.

Cửa sổ 1 (backend AI):
```powershell
cd d:\Hoc_Tap\DATT
.\START_BACKEND_AI.bat
```

Cửa sổ 2 (giao diện web):
```powershell
cd d:\Hoc_Tap\DATT
.\START_WEB.bat
```

Mở trình duyệt vào `http://localhost:5173`, kéo thả ảnh lá đậu vào để nhận diện.

**Kiểm tra backend đã nhận mô hình chưa:** mở `http://localhost:8000/api/status`,
thấy `"modelLoaded": true` và `"modelTask": "classify"` là đạt.

---

## BƯỚC 6 (ĐỂ SAU) — Huấn luyện phát hiện vùng bệnh

```powershell
python train_detection.py
```

Script này **hiện chưa chạy được** vì dataset chưa có nhãn bounding box.
Đó là hành vi đúng, không phải lỗi. Khi nào bạn gán nhãn xong trên Roboflow,
export định dạng YOLOv8 và giải nén vào `training/datasets/bean_disease_det/`
thì chạy lại lệnh trên.

**Phân biệt hai hướng:**
- **Phân loại (đang dùng):** cho biết ảnh này bị bệnh gì, độ tin cậy bao nhiêu.
  Backend dùng Grad-CAM để vẽ vùng mô hình tập trung, mang tính giải thích
  mô hình, không phải tọa độ vết bệnh do người gán nhãn.
- **Phát hiện (để sau):** vẽ khung đúng tọa độ vết bệnh. Cần nhãn bounding box.

---

## TÓM TẮT THỨ TỰ CHẠY

```
1. cd d:\Hoc_Tap\DATT\training
2. python prepare_dataset.py          chia dữ liệu
3. python train_classification.py     huấn luyện
4. python evaluate_model.py --md      đánh giá, xuất báo cáo
5. python predict_image.py <ảnh>      kiểm tra ảnh lẻ (tùy chọn)
6. START_BACKEND_AI.bat + START_WEB.bat   chạy web
```

Hoặc gộp bước 2 đến 4 thành một lệnh duy nhất:

```powershell
python run_pipeline.py
```

---

## XỬ LÝ SỰ CỐ THƯỜNG GẶP

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `CUDA out of memory` | Batch quá lớn với GPU 8GB | Thêm `--batch 8` hoặc `--batch 4` |
| `Chưa tìm thấy dữ liệu đã chia` | Chưa chạy bước 1 | Chạy `python prepare_dataset.py` |
| `Không tìm thấy file trọng số` | Chưa train xong | Chạy `python train_classification.py` |
| Top-1 Accuracy dưới 80% | Học chưa đủ hoặc ảnh lẫn lộn | Tăng `--epochs 100`, kiểm tra lại ảnh trong dataset |
| `modelLoaded: false` ở `/api/status` | Backend chưa có `best.pt` | Kiểm tra `backend/weights/best.pt` đã tồn tại chưa |
| Web chạy nhưng không ra kết quả | Backend chưa bật | Bật `START_BACKEND_AI.bat` trước |
| Tiếng Việt trong console bị lỗi font | Codepage Windows | Đã tự xử lý trong `bean_config.py` |

---

## CÁC FILE TRONG THƯ MỤC NÀY

| File | Chức năng |
|---|---|
| `bean_config.py` | Cấu hình chung: tên lớp, đường dẫn, tỉ lệ chia. **Sửa ở đây khi cần đổi lớp** |
| `prepare_dataset.py` | Bước 1: chia dữ liệu train/val/test |
| `train_classification.py` | Bước 2: huấn luyện YOLO26-cls |
| `train_detection.py` | Bước 6: huấn luyện YOLO26 detection (cần nhãn bbox) |
| `evaluate_model.py` | Bước 3: đánh giá và xuất báo cáo |
| `predict_image.py` | Bước 4: dự đoán ảnh lẻ |
| `group_split_evaluate.py` | Bước 3B: đo độ chính xác trung thực (chia theo nhóm ảnh cùng đợt chụp) |
| `run_pipeline.py` | Chạy gộp bước 1 đến 3 |
