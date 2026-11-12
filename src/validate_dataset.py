"""Kiểm các tệp JSONL trước khi train (chạy trên 100% mẫu, thoát mã 1 nếu có lỗi).

Kiểm: ảnh tồn tại và mở được; nhãn hợp lệ; prompt đúng hằng dùng chung build_prompt() (bản "cà phê",
tức sau khi sửa L1, không phải bản "cây trồng" đang chạy thật); target qua được parser thật;
không có nhóm nằm ở hai tập.
"""
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from common import LABELS, build_prompt, parse_production, parse_relaxed, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main():
    splits_dir = ROOT / "data" / "splits"
    raw = ROOT / "data" / "raw"
    prompt = build_prompt()
    errors, group_sets = Counter(), {}
    for sp in ("train", "val", "test"):
        p = splits_dir / f"{sp}.jsonl"
        if not p.exists():
            print(f"Thiếu {p}; chạy prepare_dataset.py trước")
            return 1
        rows = read_jsonl(p)
        group_sets[sp] = {r["group_id"] for r in rows}
        for r in rows:
            try:
                with Image.open(raw / r["image_path"]) as im:
                    im.verify()
            except Exception as e:  # noqa: BLE001
                errors[f"{sp}: ảnh lỗi/không mở được"] += 1
                print("  ", r["sample_id"], type(e).__name__)
            if r["label"] not in LABELS:
                errors[f"{sp}: nhãn lạ"] += 1
            if r["prompt"] != prompt:
                errors[f"{sp}: prompt khác hằng dùng chung build_prompt()"] += 1
            d, why = parse_production(r["target"])
            if d is None:
                # Với schema minimal đây là LỖI MONG ĐỢI (L2): parser thật sẽ trả failed.
                errors[f"{sp}: target KHÔNG qua parser thật ({why})"] += 1
            elif d["primary_diagnosis"] != r["label"]:
                errors[f"{sp}: target khác nhãn"] += 1
            if parse_relaxed(r["target"])[0] is None:
                errors[f"{sp}: target không phải JSON hợp lệ"] += 1
        print(f"{sp}: {len(rows)} mẫu, {len(group_sets[sp])} nhóm")
    names = list(group_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            both = group_sets[names[i]] & group_sets[names[j]]
            if both:
                errors[f"nhóm nằm ở cả {names[i]} và {names[j]}"] += len(both)
    if errors:
        print("\nLỖI:")
        for k, v in errors.items():
            print(f"  {v} x {k}")
        return 1
    print("\nOK: 0 lỗi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
