from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

from bean_config import (
    BACKEND_WEIGHTS_DIR,
    CLASS_NAMES,
    RUNS_DIR,
    SPLIT_DATASET_DIR,
    print_banner,
)

# Các mô hình YOLO26-cls hợp lệ, xếp từ nhẹ đến nặng
# n: nhanh nhất để thử   |   s: cân bằng   |   m/l/x: chính xác cao (nặng)
MODEL_CHOICES = [
    "yolo26n-cls.pt",
    "yolo26s-cls.pt",
    "yolo26m-cls.pt",
    "yolo26l-cls.pt",
    "yolo26x-cls.pt",
]

DEFAULT_MODEL = "yolo26n-cls.pt"
DEFAULT_RUN_NAME = "bean_disease_yolo26_cls"


def parse_args():
    p = argparse.ArgumentParser(
        description="Huấn luyện YOLO26-cls nhận diện 5 lớp bệnh lá đậu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL, choices=MODEL_CHOICES,
                   help="Trọng số YOLO26-cls khởi tạo (pretrained ImageNet)")
    p.add_argument("--epochs", type=int, default=80,
                   help="Số epoch huấn luyện")
    p.add_argument("--imgsz", type=int, default=320,
                   help="Kích thước ảnh đầu vào (320 đủ tốt cho lá đơn, train nhanh gấp 4 lần 640)")
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size (giảm xuống 8 nếu GPU báo Out Of Memory)")
    p.add_argument("--lr0", type=float, default=0.001,
                   help="Tốc độ học ban đầu")
    p.add_argument("--patience", type=int, default=0,
                   help="Early stopping: 0 = tắt dừng sớm để chạy đủ 100% số epoch (80 epoch)")
    p.add_argument("--device", default="0",
                   help="'0' = GPU đầu tiên, 'cpu' = chạy CPU (rất chậm)")
    p.add_argument("--workers", type=int, default=4,
                   help="Số luồng nạp dữ liệu")
    p.add_argument("--name", default=DEFAULT_RUN_NAME,
                   help="Tên thư mục lưu kết quả trong training/runs/")
    p.add_argument("--quick", action="store_true",
                   help="Chế độ chạy thử: 10 epoch, model nhỏ nhất")
    p.add_argument("--resume", action="store_true",
                   help="Train tiếp từ lần chạy dang dở")
    p.add_argument("--no-copy-to-backend", action="store_true",
                   help="Không tự copy best.pt sang backend/weights/")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.epochs = 10
        args.model = "yolo26n-cls.pt"
        args.name = "quick_test"

    # ---- Kiểm tra dữ liệu đã chia chưa -----------------------------------
    train_dir = os.path.join(SPLIT_DATASET_DIR, "train")
    val_dir = os.path.join(SPLIT_DATASET_DIR, "val")
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        print("[LỖI] Chưa tìm thấy dữ liệu đã chia.\n"
              "      Hãy chạy trước:  python prepare_dataset.py")
        return 1

    print_banner("BƯỚC 2: HUẤN LUYỆN YOLO26-CLS — NHẬN DIỆN BỆNH LÁ ĐẬU")
    print(f"  Dataset   : {SPLIT_DATASET_DIR}")
    print(f"  Model gốc : {args.model}")
    print(f"  Số lớp    : {len(CLASS_NAMES)} lớp: {', '.join(CLASS_NAMES)}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Ảnh vào   : {args.imgsz}x{args.imgsz}")
    print(f"  Batch     : {args.batch}")
    print(f"  Device    : {args.device}")

    # Import ở đây (không phải đầu file) để `--help` vẫn chạy được
    # ngay cả khi chưa cài ultralytics.
    from ultralytics import YOLO

    print(f"\n  Đang tải trọng số pretrained {args.model} (lần đầu sẽ tự tải về)...")
    model = YOLO(args.model)

    run_dir = os.path.join(RUNS_DIR, args.name)
    t0 = time.time()

    print("\n  Bắt đầu huấn luyện — theo dõi tiến độ ở bảng bên dưới.\n")
    model.train(
        data=SPLIT_DATASET_DIR,   # với bài toán cls, trỏ thẳng vào thư mục gốc
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        label_smoothing=0.08,  # Tránh overfit và hiện tượng dự đoán cực đoan 100%
        # --- Augmentation phù hợp với ảnh chụp lá ngoài đồng ---
        fliplr=0.5,     # lật ngang
        flipud=0.3,     # lật dọc (ảnh chụp lá từ trên xuống rất hợp)
        scale=0.5,      # thay đổi tỉ lệ, giúp mô hình chịu được ảnh xa gần khác nhau
        erasing=0.2,    # xóa ngẫu nhiên một vùng, chống học vẹt theo nền ảnh
        hsv_h=0.015,    # lệch màu nhẹ, giúp chịu được điều kiện ánh sáng khác nhau
        hsv_s=0.5,
        hsv_v=0.4,
        # --- Lưu kết quả ---
        project=RUNS_DIR,
        name=args.name,
        exist_ok=True,
        save=True,
        plots=True,     # sinh confusion_matrix.png, results.png cho báo cáo
        resume=args.resume,
        verbose=True,
    )

    elapsed_min = (time.time() - t0) / 60
    best_pt = os.path.join(run_dir, "weights", "best.pt")

    print_banner("HUẤN LUYỆN HOÀN TẤT")
    print(f"  Thời gian chạy : {elapsed_min:.1f} phút")
    print(f"  Thư mục kết quả: {run_dir}")
    print(f"  Trọng số tốt nhất: {best_pt}")

    if not os.path.exists(best_pt):
        print("\n  [CẢNH BÁO] Không tìm thấy best.pt. Hãy kiểm tra log phía trên.")
        return 1

    # ---- Tự động đưa mô hình vào backend để web chạy được ngay ----------
    if not args.no_copy_to_backend:
        dest_dir = os.path.abspath(BACKEND_WEIGHTS_DIR)
        os.makedirs(dest_dir, exist_ok=True)
        dest_pt = os.path.join(dest_dir, "best.pt")
        shutil.copy2(best_pt, dest_pt)
        # Lưu tên model để backend biết đây là YOLO26-cls
        with open(os.path.join(dest_dir, "model_info.txt"), "w", encoding="utf-8") as f:
            f.write(f"task=classify\nmodel={args.model}\nclasses={','.join(CLASS_NAMES)}\n")
        print(f"\n  [OK] Đã copy best.pt sang {dest_pt}")
        print("       Backend sẽ tự nhận mô hình mới khi bạn khởi động lại.")

    print("\n  Bước tiếp theo:")
    print("    python evaluate_model.py --weights "
          f"\"{best_pt}\"     # đánh giá chi tiết trên tập test")
    print("    python train_detection.py                       # nếu đã có nhãn bounding box")
    return 0


if __name__ == "__main__":
    sys.exit(main())
