"""Sinh bộ ảnh GIẢ (không phải ảnh lá thật) để chạy thử cả pipeline trước khi tải RoCoLe.

Mục đích: kiểm môi trường, bộ nhớ GPU và luồng dữ liệu -> train -> đo. Điểm đạt được trên bộ này
KHÔNG có ý nghĩa nông học: mỗi lớp chỉ là một hoạ tiết màu dễ phân biệt.

  python src/make_synthetic.py            # tạo data/raw/synthetic + data/manifest.csv
"""
import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from common import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
SEVERITY = {"healthy_leaf": "low", "coffee_leaf_rust": "medium", "red_spider_mite": "low"}


def draw(label, rng, size=320):
    img = Image.new("RGB", (size, size), (rng.randint(150, 200), rng.randint(170, 210), rng.randint(140, 180)))
    d = ImageDraw.Draw(img)
    d.ellipse((20, 40, size - 20, size - 40), fill=(rng.randint(30, 70), rng.randint(110, 150), rng.randint(30, 70)))
    if label == "coffee_leaf_rust":  # đốm tròn cam, to
        for _ in range(rng.randint(6, 12)):
            x, y, r = rng.randint(60, size - 60), rng.randint(70, size - 70), rng.randint(8, 18)
            d.ellipse((x - r, y - r, x + r, y + r), fill=(rng.randint(210, 245), rng.randint(110, 150), 20))
    elif label == "red_spider_mite":  # chấm nhỏ đỏ nâu, dày
        for _ in range(rng.randint(60, 110)):
            x, y = rng.randint(50, size - 50), rng.randint(60, size - 60)
            d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(rng.randint(150, 190), 40, 30))
    return img


def draw_ood(kind, rng, size=320):
    """Ảnh NGOÀI LỀ: thứ model không nên gán bệnh. Không nằm trong train/val/test."""
    if kind == "noise":      # nhiễu hạt, không có vật thể
        return Image.frombytes("RGB", (size, size), rng.randbytes(size * size * 3))
    if kind == "dark":       # thiếu sáng: gần như đen
        return draw("healthy_leaf", rng, size).point(lambda v: int(v * 0.05))
    if kind == "blur":       # mờ nặng: không còn thấy triệu chứng
        return draw("coffee_leaf_rust", rng, size).filter(ImageFilter.GaussianBlur(28))
    img = Image.new("RGB", (size, size), (rng.randint(170, 210),) * 3)  # "object": không phải lá
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(3, 6)):
        x, y = rng.randint(10, size - 120), rng.randint(10, size - 120)
        d.rectangle((x, y, x + rng.randint(40, 110), y + rng.randint(40, 110)),
                    fill=(rng.randint(20, 90), rng.randint(40, 110), rng.randint(150, 240)))
    return img


def make_ood(out, listing, seed, per_kind=5):
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in ("noise", "dark", "blur", "object"):
        for k in range(per_kind):
            sid = f"ood_{kind}_{k}"
            draw_ood(kind, rng).save(out / f"{sid}.jpg", quality=90)
            rows.append({"sample_id": sid, "image_path": f"{out.name}/{sid}.jpg", "kind": kind})
    write_jsonl(listing, rows)
    print(f"Đã tạo {len(rows)} ảnh ngoài lề -> {out}\nDanh sách: {listing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", type=int, default=90, help="số cây giả; mỗi cây 4 ảnh như RoCoLe")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "raw" / "synthetic"))
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--force", action="store_true", help="cho phép ghi đè manifest đã có")
    ap.add_argument("--ood", default=str(ROOT / "data" / "ood.jsonl"), help="danh sách ảnh ngoài lề")
    ap.add_argument("--ood-only", action="store_true", help="chỉ tạo ảnh ngoài lề, không đụng manifest")
    a = ap.parse_args()

    make_ood(Path(a.out_dir).parent / "synthetic_ood", Path(a.ood), a.seed + 1)
    if a.ood_only:
        return

    manifest = Path(a.manifest)
    if manifest.exists() and not a.force:
        raise SystemExit(f"{manifest} đã có. Dùng --force nếu muốn ghi đè (cẩn thận: có thể là manifest thật).")

    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in range(a.plants):
        for k in range(4):
            label = rng.choices(["healthy_leaf", "coffee_leaf_rust", "red_spider_mite"], weights=[5, 3, 2])[0]
            sid = f"syn_{p:03d}_{k}"
            rel = f"synthetic/{sid}.jpg"
            draw(label, rng).save(out / f"{sid}.jpg", quality=90)
            rows.append({"sample_id": sid, "image_path": rel, "label": label, "severity": SEVERITY[label],
                         "group_id": f"plant_{p:03d}", "source_dataset": "synthetic"})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Đã tạo {len(rows)} ảnh giả từ {a.plants} cây -> {out}\nManifest: {manifest}")


if __name__ == "__main__":
    main()
