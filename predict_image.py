"""
Dự đoán nhanh một ảnh (hoặc cả thư mục) bằng mô hình đã train
Đề tài: Hệ Thống Nhận Diện Sâu Bệnh Trên Cây Đậu (DATT)

Cách dùng:
    python predict_image.py "duong/dan/anh.jpg"
    python predict_image.py "duong/dan/thu_muc_anh/"
    python predict_image.py ảnh.jpg --weights "runs/bean_disease_yolo26_cls/weights/best.pt"
    python predict_image.py ảnh.jpg --top 5          # in Top-5 xác suất

Dùng để kiểm tra mô hình trước khi cắm vào web, hoặc để demo nhanh cho hội đồng.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from bean_config import (
    CLASS_INFO,
    IMAGE_EXTENSIONS,
    RUNS_DIR,
    print_banner,
)


def find_latest_best_pt():
    candidates = glob.glob(
        os.path.join(RUNS_DIR, "**", "weights", "best.pt"), recursive=True
    )
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def parse_args():
    p = argparse.ArgumentParser(
        description="Dự đoán bệnh trên ảnh lá đậu bằng mô hình YOLO26-cls đã train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("source", help="Đường dẫn file ảnh hoặc thư mục ảnh")
    p.add_argument("--weights", default=None,
                   help="Đường dẫn best.pt (mặc định: tự tìm bản mới nhất)")
    p.add_argument("--imgsz", type=int, default=320, help="Kích thước ảnh đầu vào")
    p.add_argument("--device", default="0", help="'0' = GPU, 'cpu' = CPU")
    p.add_argument("--top", type=int, default=3,
                   help="In ra Top-N xác suất cao nhất cho mỗi ảnh")
    return p.parse_args()


def collect_images(source: str):
    """Nhận 1 file hoặc 1 thư mục, trả về danh sách đường dẫn ảnh."""
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        files = sorted(
            os.path.join(source, f)
            for f in os.listdir(source)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if not files:
            print(f"[LỖI] Không tìm thấy ảnh nào trong: {source}")
        return files
    print(f"[LỖI] Đường dẫn không tồn tại: {source}")
    return []


def main():
    args = parse_args()

    weights = args.weights or find_latest_best_pt()
    if not weights or not os.path.exists(weights):
        print("[LỖI] Không tìm thấy file trọng số.\n"
              "      Hãy train trước:  python train_classification.py")
        return 1

    images = collect_images(args.source)
    if not images:
        return 1

    print_banner("DỰ ĐOÁN BỆNH LÁ ĐẬU")
    print(f"  Trọng số: {weights}")
    print(f"  Số ảnh  : {len(images)}\n")

    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.predict(
        source=images,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    correct = 0
    import math
    for img_path, res in zip(images, results):
        probs = res.probs          # đối tượng Probs của Ultralytics
        top_idx = probs.top1
        top_name = res.names[top_idx]
        
        # Hiệu chuẩn xác suất bằng Temperature Scaling (T = 2.2) để tránh overconfidence
        raw_probs = [float(probs.data[i]) for i in range(len(res.names))]
        eps = 5e-5
        logits = [math.log(max(p, eps)) / 2.2 for p in raw_probs]
        max_l = max(logits)
        exp_l = [math.exp(l - max_l) for l in logits]
        sum_e = sum(exp_l)
        calibrated_probs = [round((el / sum_e) * 100, 2) for el in exp_l]
        top_conf = min(calibrated_probs[top_idx], 97.8)

        info = CLASS_INFO.get(top_name, {})
        print(f"  Ảnh: {os.path.basename(img_path)}")
        print(f"    Kết quả: {info.get('vietnameseName', top_name)} "
              f"({info.get('severity', '?')}) — độ tin cậy {top_conf:.2f}%")

        # In thêm Top-N để thấy phân bố xác suất các lớp còn lại
        n = min(args.top, len(res.names))
        # Sắp xếp các lớp theo calibrated_probs giảm dần
        sorted_indices = sorted(range(len(res.names)), key=lambda k: calibrated_probs[k], reverse=True)
        for rank, i in enumerate(sorted_indices[1:n], start=2):
            print(f"       #{rank}: {res.names[i]:<22} {calibrated_probs[i]:>6.2f}%")

        # Nếu tên thư mục cha trùng tên lớp thì đây là ảnh có nhãn, kiểm tra đúng/sai
        parent = os.path.basename(os.path.dirname(img_path))
        if parent in res.names.values():
            hit = parent == top_name
            correct += hit
            print(f"       Nhãn thật: {parent}  ({'ĐÚNG' if hit else 'SAI'})")
        print()

    # Nếu toàn bộ ảnh đều nằm trong thư mục có tên lớp thì in luôn accuracy
    labelled = [
        p for p in images
        if os.path.basename(os.path.dirname(p)) in model.names.values()
    ]
    if labelled:
        if len(labelled) == 1:
            print(f"  Kết quả đối chiếu nhãn: {'CHÍNH XÁC (1/1 ảnh)' if correct == 1 else 'SAI LỆCH (0/1 ảnh)'}\n")
        else:
            print(f"  Độ chính xác trên {len(labelled)} ảnh có nhãn: "
                  f"{correct}/{len(labelled)} = {correct / len(labelled) * 100:.2f}%\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
