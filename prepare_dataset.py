from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys

from bean_config import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    RANDOM_SEED,
    RAW_DATASET_DIR,
    SPLIT_DATASET_DIR,
    SPLIT_RATIOS,
    check_raw_dataset,
    ensure_dirs,
    print_banner,
)

SPLITS = ("train", "val", "test")


def list_images(folder: str):
    """Lấy tên các file ảnh hợp lệ trong một thư mục (không đệ quy)."""
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def link_or_copy(src: str, dst: str, use_copy: bool):
    """
    Đưa file vào tập dữ liệu đích.

    hardlink (mặc định) : nhanh, không tốn đĩa, nhưng XÓA file ở thư mục gốc
                          cũng làm file trong datasets/ biến mất.
    copy (--copy)       : an toàn tuyệt đối, tốn gấp đôi dung lượng.
    """
    if os.path.exists(dst):
        return
    if use_copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)  # hardlink
        except OSError:
            shutil.copy2(src, dst)  # khác ổ đĩa / FS không hỗ trợ thì copy


def split_class(
    class_name: str, use_copy: bool, manifest: list
):
    """Chia ảnh của MỘT lớp thành 3 tập, giữ đúng tỉ lệ SPLIT_RATIOS."""
    src_dir = os.path.join(RAW_DATASET_DIR, class_name)
    files = list_images(src_dir)

    # Trộn với seed cố định, chạy lại luôn ra kết quả y hệt
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * SPLIT_RATIOS["train"])
    n_val = int(n_total * SPLIT_RATIOS["val"])
    # Phần còn lại dồn hết cho test để không mất ảnh nào do làm tròn
    buckets = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }

    counts = {}
    for split, subset in buckets.items():
        dst_dir = os.path.join(SPLIT_DATASET_DIR, split, class_name)
        ensure_dirs(dst_dir)
        for fname in subset:
            link_or_copy(
                os.path.join(src_dir, fname),
                os.path.join(dst_dir, fname),
                use_copy,
            )
            manifest.append({
                "split": split,
                "class_name": class_name,
                "file_name": fname,
                "source": os.path.join(src_dir, fname),
            })
        counts[split] = len(subset)

    return counts


def verify_split():
    """Kiểm tra thư mục đã chia có đủ 3 tập x 5 lớp và đếm số ảnh."""
    ok = True
    print_banner("KIỂM TRA DỮ LIỆU ĐÃ CHIA")
    print(f"  Thư mục: {SPLIT_DATASET_DIR}\n")

    header = f"  {'Lớp':<22}" + "".join(f"{s:>10}" for s in SPLITS) + f"{'Tổng':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    totals = dict.fromkeys(SPLITS, 0)
    for class_name in CLASS_NAMES:
        row = f"  {class_name:<22}"
        class_total = 0
        for split in SPLITS:
            d = os.path.join(SPLIT_DATASET_DIR, split, class_name)
            n = len(list_images(d)) if os.path.isdir(d) else 0
            if n == 0:
                ok = False
            totals[split] += n
            class_total += n
            row += f"{n:>10}"
        row += f"{class_total:>10}"
        print(row)

    print("  " + "-" * (len(header) - 2))
    grand = sum(totals.values())
    print(f"  {'TỔNG':<22}" + "".join(f"{totals[s]:>10}" for s in SPLITS) + f"{grand:>10}")
    print(f"\n  KẾT LUẬN: {'DỮ LIỆU HỢP LỆ' if ok else 'DỮ LIỆU CHƯA ĐẦY ĐỦ (thiếu tập trống)'}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Chia dataset ảnh thô thành train/val/test cho YOLO26-cls"
    )
    parser.add_argument("--force", action="store_true",
                        help="Xóa dữ liệu đã chia và chia lại từ đầu")
    parser.add_argument("--copy", action="store_true",
                        help="Copy file thật thay vì hardlink (tốn dung lượng)")
    parser.add_argument("--verify", action="store_true",
                        help="Chỉ kiểm tra dữ liệu đã chia, không chia lại")
    args = parser.parse_args()

    if args.verify:
        return 0 if verify_split() else 1

    check_raw_dataset()

    # Nếu đã chia rồi thì bỏ qua, trừ khi người dùng yêu cầu --force
    if os.path.isdir(SPLIT_DATASET_DIR) and not args.force:
        if verify_split():
            print("\n  [SKIP] Dữ liệu đã được chia sẵn. "
                  "Dùng --force nếu muốn chia lại từ đầu.\n")
            return 0

    if args.force and os.path.isdir(SPLIT_DATASET_DIR):
        print(f"  [--force] Đang xóa dữ liệu cũ: {SPLIT_DATASET_DIR}")
        shutil.rmtree(SPLIT_DATASET_DIR)

    print_banner("BƯỚC 1: CHIA DỮ LIỆU TRAIN / VAL / TEST")
    print(f"  Nguồn      : {RAW_DATASET_DIR}")
    print(f"  Đích       : {SPLIT_DATASET_DIR}")
    print(f"  Phương thức: {'COPY (tốn dung lượng)' if args.copy else 'HARDLINK (nhanh, không tốn đĩa)'}")
    print(f"  Tỉ lệ      : train {SPLIT_RATIOS['train']:.0%} | "
          f"val {SPLIT_RATIOS['val']:.0%} | test {SPLIT_RATIOS['test']:.0%}")
    print(f"  Seed       : {RANDOM_SEED} (kết quả tái lập được)\n")

    ensure_dirs(SPLIT_DATASET_DIR)
    manifest: list[dict] = []

    for class_name in CLASS_NAMES:
        n_src = len(list_images(os.path.join(RAW_DATASET_DIR, class_name)))
        counts = split_class(class_name, args.copy, manifest)
        print(f"  {class_name:<22} gốc={n_src:<5} "
              f"train={counts['train']:<5} val={counts['val']:<5} test={counts['test']}")

    # Lưu manifest để chứng minh tính tái lập khi bảo vệ đồ án
    manifest_path = os.path.join(SPLIT_DATASET_DIR, "split_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "class_name", "file_name", "source"]
        )
        writer.writeheader()
        writer.writerows(manifest)
    print(f"\n  Đã lưu danh sách chia dữ liệu: {manifest_path}")

    print()
    ok = verify_split()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
