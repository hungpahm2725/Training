"""
Cấu hình dùng chung cho toàn bộ pipeline huấn luyện
Đề tài: Hệ Thống Nhận Diện Sâu Bệnh Trên Cây Đậu (DATT)

File này là NGUỒN SỰ THẬT DUY NHẤT (single source of truth) về:
  - Danh sách 5 lớp bệnh thực tế trong dataset
  - Đường dẫn thư mục dữ liệu
  - Metadata tiếng Việt phục vụ báo cáo / slide bảo vệ

Mọi script khác (prepare / train / evaluate / predict) đều import từ đây,
nên khi cần đổi tên lớp hay tỉ lệ chia dữ liệu bạn CHỈ SỬA Ở FILE NÀY.
"""

from __future__ import annotations

import os
import sys

# --------------------------------------------------------------------------
# 0. Sửa lỗi font tiếng Việt trên console Windows (cp1252 sang utf-8)
# --------------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass  # Python < 3.7 hoặc stream đã bị chuyển hướng

# --------------------------------------------------------------------------
# 1. Đường dẫn
# --------------------------------------------------------------------------
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))

# Thư mục gốc chứa ảnh thô, mỗi lớp là một thư mục con
RAW_DATASET_DIR = os.path.join(
    TRAINING_DIR, "dataset", "Bean Leaf Disease Original Image"
)

# Thư mục chứa dữ liệu ĐÃ CHIA sẵn theo chuẩn Ultralytics:
#   datasets/bean_disease/{train,val,test}/<tên lớp>/*.jpg
SPLIT_DATASET_DIR = os.path.join(TRAINING_DIR, "datasets", "bean_disease")

# Nơi lưu kết quả huấn luyện (weights, đồ thị, log)
RUNS_DIR = os.path.join(TRAINING_DIR, "runs")

# Thư mục backend để tự động copy best.pt sang sau khi train xong
BACKEND_WEIGHTS_DIR = os.path.join(TRAINING_DIR, "..", "backend", "weights")

# --------------------------------------------------------------------------
# 2. Danh sách lớp — PHẢI khớp tên thư mục trong RAW_DATASET_DIR
# --------------------------------------------------------------------------
# Tên lớp = tên thư mục, Ultralytics tự sắp xếp theo alphabet nên ta khai báo
# theo đúng thứ tự đó để index trong model khớp với CLASS_NAMES.
CLASS_NAMES = [
    "Angular Leaf Spot",      # 0
    "Anthracnose",            # 1
    "Cercospora Leaf Spot",   # 2
    "Fresh Leaf",             # 3
    "Fungi Pathogens",        # 4
    "Halo Blight",            # 5
    "Mosaic Virus",           # 6
    "Potassium Deficiency",   # 7
    "Rust",                   # 8
    "Tobacco Caterpillar",    # 9
    "Yellow Mosaic",          # 10
]

# Khóa định danh an toàn cho URL/JSON (frontend dùng làm `diseaseKey`)
CLASS_KEYS = {
    "Angular Leaf Spot": "angular_leaf_spot",
    "Anthracnose": "anthracnose",
    "Cercospora Leaf Spot": "cercospora_leaf_spot",
    "Fresh Leaf": "fresh_leaf",
    "Fungi Pathogens": "fungi_pathogens",
    "Halo Blight": "halo_blight",
    "Mosaic Virus": "mosaic_virus",
    "Potassium Deficiency": "potassium_deficiency",
    "Rust": "rust",
    "Tobacco Caterpillar": "tobacco_caterpillar",
    "Yellow Mosaic": "yellow_mosaic",
}

# --------------------------------------------------------------------------
# 3. Metadata bệnh học tiếng Việt (dùng để sinh báo cáo, không dùng khi train)
# --------------------------------------------------------------------------
CLASS_INFO = {
    "Angular Leaf Spot": {
        "vietnameseName": "Bệnh đốm lá góc cạnh",
        "scientificName": "Pseudocercospora griseola",
        "severity": "Trung bình",
        "colorHex": "#F5B335",
    },
    "Anthracnose": {
        "vietnameseName": "Bệnh thán thư hại đậu",
        "scientificName": "Colletotrichum lindemuthianum",
        "severity": "Nguy hiểm",
        "colorHex": "#C2391C",
    },
    "Cercospora Leaf Spot": {
        "vietnameseName": "Bệnh đốm lá Cercospora",
        "scientificName": "Cercospora canescens",
        "severity": "Trung bình",
        "colorHex": "#D97706",
    },
    "Fresh Leaf": {
        "vietnameseName": "Lá đậu khỏe mạnh",
        "scientificName": "Phaseolus vulgaris (Healthy)",
        "severity": "An toàn",
        "colorHex": "#2B6E3F",
    },
    "Fungi Pathogens": {
        "vietnameseName": "Bệnh nấm lá tổng hợp",
        "scientificName": "Fungal Pathogens Complex",
        "severity": "Nguy hiểm",
        "colorHex": "#B91C1C",
    },
    "Halo Blight": {
        "vietnameseName": "Bệnh đốm quầng vi khuẩn (Halo Blight)",
        "scientificName": "Pseudomonas syringae pv. phaseolicola",
        "severity": "Nguy hiểm",
        "colorHex": "#EA580C",
    },
    "Mosaic Virus": {
        "vietnameseName": "Bệnh virus khảm lá thường (BCMV)",
        "scientificName": "Bean Common Mosaic Virus",
        "severity": "Nguy hiểm",
        "colorHex": "#7C3AED",
    },
    "Potassium Deficiency": {
        "vietnameseName": "Rối loạn dinh dưỡng thiếu Kali",
        "scientificName": "Potassium (K) Deficiency",
        "severity": "Cảnh báo",
        "colorHex": "#EAB308",
    },
    "Rust": {
        "vietnameseName": "Bệnh gỉ sắt hại đậu",
        "scientificName": "Uromyces appendiculatus",
        "severity": "Nguy hiểm",
        "colorHex": "#E8572F",
    },
    "Tobacco Caterpillar": {
        "vietnameseName": "Sâu khoang ăn lá hại đậu",
        "scientificName": "Spodoptera litura",
        "severity": "Nguy hiểm",
        "colorHex": "#475569",
    },
    "Yellow Mosaic": {
        "vietnameseName": "Bệnh virus khảm vàng lá đậu (MYMV)",
        "scientificName": "Mungbean Yellow Mosaic Virus",
        "severity": "Nguy hiểm",
        "colorHex": "#9333EA",
    },
}

# --------------------------------------------------------------------------
# 4. Tỉ lệ chia dữ liệu (stratified — mỗi lớp giữ đúng tỉ lệ này)
# --------------------------------------------------------------------------
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Seed cố định để kết quả chia dữ liệu TÁI LẬP ĐƯỢC (quan trọng khi bảo vệ đồ án:
# hội đồng chạy lại script sẽ ra đúng kết quả như trong báo cáo của bạn)
RANDOM_SEED = 42

# Đuôi file ảnh được chấp nhận
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------
# 5. Hàm tiện ích dùng chung
# --------------------------------------------------------------------------
def ensure_dirs(*paths: str):
    """Tạo các thư mục nếu chưa tồn tại."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def check_raw_dataset():
    """
    Kiểm tra thư mục ảnh thô có tồn tại và có đủ 5 lớp hay không.
    Ném lỗi rõ ràng thay vì để Ultralytics báo lỗi khó hiểu về sau.
    """
    if not os.path.isdir(RAW_DATASET_DIR):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu thô:\n  {RAW_DATASET_DIR}\n"
            f"Hãy kiểm tra lại đường dẫn RAW_DATASET_DIR trong bean_config.py"
        )

    missing = [
        c for c in CLASS_NAMES
        if not os.path.isdir(os.path.join(RAW_DATASET_DIR, c))
    ]
    if missing:
        raise FileNotFoundError(
            "Thiếu các thư mục lớp sau trong dataset:\n  - "
            + "\n  - ".join(missing)
        )


def print_banner(title: str):
    """In tiêu đề dạng khung cho dễ nhìn trong log."""
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}")
