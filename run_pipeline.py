"""
Chạy TOÀN BỘ pipeline chỉ bằng một lệnh
Đề tài: Hệ Thống Nhận Diện Sâu Bệnh Trên Cây Đậu (DATT)

    python run_pipeline.py                 # prepare, train, evaluate, báo cáo
    python run_pipeline.py --skip-prepare  # bỏ qua bước chia dữ liệu
    python run_pipeline.py --quick         # chạy thử nhanh toàn bộ (10 epoch)

Đây là lệnh bạn nên dùng khi demo cho hội đồng: một lệnh chạy hết từ ảnh thô
đến báo cáo đánh giá, không cần copy file thủ công.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from bean_config import BACKEND_WEIGHTS_DIR, RUNS_DIR, print_banner

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_NAME = "bean_disease_yolo26_cls"


def run_step(title: str, script: str, extra_args: list):
    """Chạy một script con bằng chính trình Python đang gọi pipeline."""
    print_banner(title)
    cmd = [sys.executable, os.path.join(HERE, script)] + extra_args
    print(f"  $ {' '.join(cmd)}\n")
    t0 = time.time()
    # Dùng cwd=HERE để các import nội bộ (bean_config) luôn tìm thấy nhau
    result = subprocess.run(cmd, cwd=HERE)
    print(f"\n  Kết thúc sau {(time.time() - t0) / 60:.1f} phút "
          f"(exit code {result.returncode})")
    return result.returncode


def parse_args():
    p = argparse.ArgumentParser(
        description="Chạy toàn bộ pipeline huấn luyện nhận diện bệnh lá đậu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--skip-prepare", action="store_true",
                   help="Bỏ qua bước chia dữ liệu (nếu đã chia rồi)")
    p.add_argument("--skip-evaluate", action="store_true",
                   help="Bỏ qua bước đánh giá")
    p.add_argument("--quick", action="store_true",
                   help="Chạy thử nhanh: 10 epoch, model yolo26n-cls")
    p.add_argument("--model", default=None, help="Ghi đè model (vd: yolo26s-cls.pt)")
    p.add_argument("--epochs", type=int, default=None, help="Ghi đè số epoch")
    p.add_argument("--batch", type=int, default=None, help="Ghi đè batch size")
    return p.parse_args()


def main():
    args = parse_args()

    print_banner("PIPELINE HUẤN LUYỆN — NHẬN DIỆN BỆNH LÁ ĐẬU (YOLO26)")
    print("  Bước 1: Chia dữ liệu train/val/test")
    print("  Bước 2: Huấn luyện YOLO26-cls")
    print("  Bước 3: Đánh giá trên tập test + xuất báo cáo")

    train_args = []
    if args.quick:
        train_args.append("--quick")
    if args.model:
        train_args += ["--model", args.model]
    if args.epochs:
        train_args += ["--epochs", str(args.epochs)]
    if args.batch:
        train_args += ["--batch", str(args.batch)]

    # ---- Bước 1 ----------------------------------------------------------
    if not args.skip_prepare:
        if run_step("BƯỚC 1/3 — CHUẨN BỊ DỮ LIỆU", "prepare_dataset.py", []) != 0:
            print("\n[LỖI] Bước chuẩn bị dữ liệu thất bại. Dừng pipeline.")
            return 1
    else:
        print("\n  [SKIP] Bỏ qua bước chuẩn bị dữ liệu (--skip-prepare)")

    # ---- Bước 2 ----------------------------------------------------------
    if run_step("BƯỚC 2/3 — HUẤN LUYỆN", "train_classification.py", train_args) != 0:
        print("\n[LỖI] Huấn luyện thất bại. Dừng pipeline.")
        return 1

    # ---- Bước 3 ----------------------------------------------------------
    weights = os.path.join(RUNS_DIR,
                           "quick_test" if args.quick else RUN_NAME,
                           "weights", "best.pt")
    if not args.skip_evaluate:
        eval_args = ["--md"]
        if os.path.exists(weights):
            eval_args += ["--weights", weights]
        if args.quick:
            eval_args += ["--imgsz", "224"]
        if run_step("BƯỚC 3/3 — ĐÁNH GIÁ MÔ HÌNH", "evaluate_model.py",
                    eval_args) != 0:
            print("\n[CẢNH BÁO] Đánh giá thất bại, nhưng mô hình đã train xong.")
    else:
        print("\n  [SKIP] Bỏ qua bước đánh giá (--skip-evaluate)")

    # ---- Tổng kết --------------------------------------------------------
    print_banner("HOÀN TẤT PIPELINE")
    print(f"  Trọng số mô hình : {weights}")
    print(f"  Đã copy backend  : {os.path.abspath(BACKEND_WEIGHTS_DIR)}")
    print("\n  Kiểm tra kết quả:")
    print(f"    python predict_image.py \"dataset/Bean Leaf Disease Original Image/Rust\"")
    print("\n  Chạy web:")
    print("    ..\\START_BACKEND_AI.bat     (cửa sổ 1)")
    print("    ..\\START_WEB.bat            (cửa sổ 2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
