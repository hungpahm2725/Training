from __future__ import annotations

import argparse
import os
import sys

from bean_config import CLASS_NAMES, RUNS_DIR, TRAINING_DIR, print_banner

# Thư mục dataset detection (khác với thư mục classification đã chia)
DET_DATASET_DIR = os.path.join(TRAINING_DIR, "datasets", "bean_disease_det")
DET_DATA_YAML = os.path.join(DET_DATASET_DIR, "data.yaml")

MODEL_CHOICES = ["yolo26n.pt"]


def write_data_yaml():
    """
    Sinh file data.yaml trỏ đúng đường dẫn tuyệt đối trên máy này.
    Dùng đường dẫn tuyệt đối để tránh lỗi 'dataset not found' của
    Ultralytics khi chạy script từ thư mục khác.
    """
    import yaml

    cfg = {
        "path": DET_DATASET_DIR.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(DET_DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"  Đã sinh file cấu hình: {DET_DATA_YAML}")


def preflight_check():
    """
    Kiểm tra dataset detection đã sẵn sàng chưa.
    Trả về danh sách các vấn đề tìm được (rỗng = OK).
    """
    problems = []
    if not os.path.isdir(DET_DATASET_DIR):
        problems.append(f"Chưa có thư mục dataset detection: {DET_DATASET_DIR}")
        return problems

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        p = os.path.join(DET_DATASET_DIR, sub)
        if not os.path.isdir(p):
            problems.append(f"Thiếu thư mục: {p}")

    # Đếm số ảnh và số nhãn để phát hiện tình trạng chia sai
    img_train = os.path.join(DET_DATASET_DIR, "images", "train")
    lbl_train = os.path.join(DET_DATASET_DIR, "labels", "train")
    if os.path.isdir(img_train) and os.path.isdir(lbl_train):
        n_img = len([f for f in os.listdir(img_train)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        n_lbl = len([f for f in os.listdir(lbl_train) if f.lower().endswith(".txt")])
        if n_img == 0:
            problems.append("Thư mục images/train rỗng.")
        elif n_lbl == 0:
            problems.append(
                f"Có {n_img} ảnh nhưng KHÔNG có file nhãn .txt nào trong labels/train.\n"
                "      Bạn chưa gán nhãn bounding box. Xem hướng dẫn ở đầu file này."
            )
        elif n_lbl < n_img:
            problems.append(
                f"Số file nhãn ({n_lbl}) ít hơn số ảnh ({n_img}). "
                "Một số ảnh sẽ bị Ultralytics bỏ qua."
            )
    return problems


def parse_args():
    p = argparse.ArgumentParser(
        description="Huấn luyện YOLO26 phát hiện vùng bệnh lá đậu (cần nhãn bounding box)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="yolo26n.pt", choices=MODEL_CHOICES,
                   help="Trọng số YOLO26 detect khởi tạo")
    p.add_argument("--epochs", type=int, default=100, help="Số epoch")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Kích thước ảnh (detection cần 640 để thấy vết bệnh nhỏ)")
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size — 8 an toàn cho GPU 8GB khi train detection 640px")
    p.add_argument("--patience", type=int, default=20, help="Early stopping")
    p.add_argument("--device", default="0", help="'0' = GPU, 'cpu' = CPU")
    p.add_argument("--workers", type=int, default=4, help="Số luồng nạp dữ liệu")
    p.add_argument("--name", default="bean_disease_yolo26_det",
                   help="Tên thư mục lưu kết quả")
    return p.parse_args()


def main():
    args = parse_args()

    print_banner("BƯỚC 2B: HUẤN LUYỆN YOLO26 DETECT — PHÁT HIỆN VÙNG BỆNH")

    problems = preflight_check()
    if problems:
        print("\n  KHÔNG THỂ BẮT ĐẦU. Các vấn đề sau cần xử lý trước:\n")
        for i, prob in enumerate(problems, 1):
            print(f"    {i}. {prob}")
        print("\n  Đây là bước TÙY CHỌN, không thuộc chuỗi chạy chính.")
        print("  Mô hình phân loại đã train xong và web đã chạy được rồi.")
        print("  Chỉ chạy bước này khi muốn nâng cấp lên phát hiện vùng bệnh,")
        print("  và phải gán nhãn bounding box cho ảnh trước đã.\n")
        return 1

    write_data_yaml()

    print(f"  Dataset : {DET_DATASET_DIR}")
    print(f"  Model   : {args.model}")
    print(f"  Epochs  : {args.epochs} | imgsz={args.imgsz} | batch={args.batch}\n")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=DET_DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=0.001,
        # Augmentation cho ảnh đồng ruộng
        mosaic=1.0,     # ghép 4 ảnh — rất hiệu quả để học vết bệnh nhỏ
        mixup=0.1,
        fliplr=0.5,
        flipud=0.2,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        project=RUNS_DIR,
        name=args.name,
        exist_ok=True,
        plots=True,
    )

    best_pt = os.path.join(RUNS_DIR, args.name, "weights", "best.pt")
    print_banner("HUẤN LUYỆN HOÀN TẤT")
    print(f"  Trọng số tốt nhất: {best_pt}")
    print("\n  Đánh giá:")
    print(f"    python evaluate_model.py --weights \"{best_pt}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
