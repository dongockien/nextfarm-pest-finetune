"""Lấy RoCoLe (ảnh lá cà phê Robusta thật) qua API công khai của Mendeley và dựng manifest.

Nguồn: RoCoLe v2, DOI 10.17632/c5yvn32dzg.2, giấy phép CC BY 4.0 (đọc từ API ngày viết script).
Tệp nhãn RoCoLe-classes.xlsx có 3 cột: File, Binary.Label, Multiclass.Label (1.560 dòng, 390 cây x 4 ảnh).
Tên ảnh dạng C<lô>P<cây><H|E><số>.jpg, nên mã cây (group_id) là phần C..P.. ở đầu tên.

GIỚI HẠN ĐÃ BIẾT: API chỉ liệt kê 1.000 trong 1.560 ảnh (không phân trang được), nên manifest chỉ gồm
những ảnh tải được. Đủ cho smoke test; muốn đủ 1.560 ảnh phải tải gói RoCoLe-voc.tar.gz (~700 MB).

Hai pha, vì chỉ tải những ảnh mà prepare_dataset.py đã chọn (141 ảnh, ~190 MB):
  python src/fetch_rocole.py --force      # 1) tải tệp nhãn, ghi data/manifest.csv (--force: ghi đè manifest giả)
  python src/prepare_dataset.py           # 2) tách theo cây, chọn 33/7/7 ảnh mỗi nhãn
  python src/fetch_rocole.py --images     # 3) tải đúng các ảnh nằm trong data/splits
  python src/validate_dataset.py          # 4) phải ra OK: 0 lỗi
"""
import argparse
import csv
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

from common import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
API = "https://data.mendeley.com/public-api/datasets/c5yvn32dzg"
PHOTOS_FOLDER = "1ca51fea-39e6-483f-a3f4-7e54cf3b4fd8"
ANNOT_FOLDER = "201c7329-268c-48fb-a452-d9b435817774"
TRUSTED_PREFIX = "https://data.mendeley.com/"
NAME_RE = re.compile(r"^(C\d+P\d+)[HE][0-9E]\.jpg$")
# Ánh xạ nhãn RoCoLe -> (nhãn smoke, severity). Mốc mức nặng là ĐỀ XUẤT, cần kỹ sư bảo vệ thực vật chốt.
# RoCoLe không có mức nặng cho lá khoẻ và nhện đỏ: "low" ở đó chỉ là giá trị giữ chỗ.
LABEL_MAP = {
    "healthy": ("healthy_leaf", "low"),
    "red_spider_mite": ("red_spider_mite", "low"),
    "rust_level_1": ("coffee_leaf_rust", "low"),
    "rust_level_2": ("coffee_leaf_rust", "medium"),
    "rust_level_3": ("coffee_leaf_rust", "high"),
    "rust_level_4": ("coffee_leaf_rust", "high"),
}


def http_get(url, tries=4):
    if not url.startswith(TRUSTED_PREFIX):
        raise SystemExit(f"URL ngoài Mendeley, không tải: {url[:80]}")
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"  lỗi {type(e).__name__}, thử lại ...")
            time.sleep(3 * (i + 1))


def get_json(url, expect, tries=5):
    """API đôi khi trả HTTP 200 kèm {"error":502} (lỗi tạm thời phía máy chủ): thử lại tới khi đúng kiểu."""
    for i in range(tries):
        data = json.loads(http_get(url))
        if isinstance(data, expect) and "error" not in data:
            return data
        print(f"  API trả {str(data)[:60]!r}, thử lại ...")
        time.sleep(3 * (i + 1))
    raise SystemExit(f"API Mendeley không trả dữ liệu hợp lệ sau {tries} lần: {url}")


def list_folder(folder_id):
    data = get_json(f"{API}/files?folder_id={folder_id}&version=2", list)
    return {d["filename"]: d["content_details"]["download_url"] for d in data}


def build_manifest(a, dest):
    manifest = Path(a.manifest)
    if manifest.exists() and not a.force:
        raise SystemExit(f"{manifest} đã có (có thể là manifest giả). Dùng --force để ghi đè.")
    import openpyxl
    dest.mkdir(parents=True, exist_ok=True)
    ann = list_folder(ANNOT_FOLDER)
    xlsx = dest / "RoCoLe-classes.xlsx"
    xlsx.write_bytes(http_get(ann["RoCoLe-classes.xlsx"]))
    photos = list_folder(PHOTOS_FOLDER)
    (dest / "files.json").write_text(json.dumps(photos), encoding="utf-8")
    info = get_json(f"{API}?version=2", dict)
    lic = info.get("data_licence") or {}
    (dest / "SOURCE.txt").write_text(
        f"RoCoLe: A robusta coffee leaf images dataset, version {info.get('version')}\n"
        f"DOI: {(info.get('doi') or {}).get('id')}\nGiấy phép: {lic.get('short_name')} ({lic.get('url')})\n"
        f"Tải ngày: {time.strftime('%Y-%m-%d')} qua {API}\n", encoding="utf-8")
    print(f"Giấy phép theo API: {lic.get('short_name')}. Ghi nguồn ở {dest / 'SOURCE.txt'}")

    ws = openpyxl.load_workbook(xlsx, read_only=True).worksheets[0]
    table = list(ws.iter_rows(values_only=True))
    if tuple(table[0][:3]) != ("File", "Binary.Label", "Multiclass.Label"):
        raise SystemExit(f"Tệp nhãn đổi cấu trúc, cột hiện có: {table[0]}")
    rows, skipped = [], Counter()
    for name, _, multi in table[1:]:
        m = NAME_RE.match(str(name))
        if not m or multi not in LABEL_MAP:
            skipped["tên tệp hoặc nhãn lạ"] += 1
            continue
        if name not in photos:
            skipped["API không liệt kê ảnh (giới hạn 1.000 tệp)"] += 1
            continue
        label, severity = LABEL_MAP[multi]
        rows.append({"sample_id": name[:-4], "image_path": f"rocole/Photos/{name}", "label": label,
                     "severity": severity, "group_id": m.group(1), "source_dataset": "rocole_v2",
                     "source_label": multi})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Tệp nhãn: {len(table) - 1} dòng. Manifest: {len(rows)} ảnh, {len({r['group_id'] for r in rows})} cây -> {manifest}")
    print("Theo nhãn smoke:", dict(Counter(r["label"] for r in rows)))
    print("Theo nhãn gốc:", dict(Counter(r["source_label"] for r in rows)))
    print("Bỏ qua:", dict(skipped) or "không")
    print("Tiếp: python src/prepare_dataset.py  rồi  python src/fetch_rocole.py --images")


def fetch_images(a, dest):
    from PIL import Image
    listing = dest / "files.json"
    if not listing.exists():
        raise SystemExit("Chưa có files.json: chạy `python src/fetch_rocole.py --force` trước.")
    photos = json.loads(listing.read_text(encoding="utf-8"))
    wanted = []
    for sp in ("train", "val", "test"):
        p = Path(a.splits) / f"{sp}.jsonl"
        if not p.exists():
            raise SystemExit(f"Thiếu {p}: chạy prepare_dataset.py trước.")
        wanted += [r["image_path"] for r in read_jsonl(p) if r["image_path"].startswith("rocole/Photos/")]
    if not wanted:
        raise SystemExit("data/splits không chứa ảnh RoCoLe (đang là dữ liệu giả?). Chạy lại pha 1 và prepare_dataset.py.")
    out = dest / "Photos"
    out.mkdir(parents=True, exist_ok=True)
    done = total = 0
    for i, rel in enumerate(sorted(set(wanted)), 1):
        name = rel.rsplit("/", 1)[1]
        if not NAME_RE.match(name) or name not in photos:
            raise SystemExit(f"Ảnh không có trong danh sách tải: {name}")
        target = out / name
        if not target.exists():
            tmp = target.with_suffix(".part")
            tmp.write_bytes(http_get(photos[name]))
            with Image.open(tmp) as im:   # chỉ nhận tệp là ảnh JPEG thật
                if im.format != "JPEG":
                    raise SystemExit(f"{name} không phải JPEG")
                im.verify()
            tmp.replace(target)
            done += 1
        total += target.stat().st_size
        if i % 20 == 0 or i == len(set(wanted)):
            print(f"  {i}/{len(set(wanted))} ảnh ({total / 1e6:.0f} MB)", flush=True)
    print(f"Xong: tải mới {done} ảnh vào {out}. Tiếp: python src/validate_dataset.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", action="store_true", help="pha 2: tải các ảnh nằm trong data/splits")
    ap.add_argument("--force", action="store_true", help="pha 1: cho phép ghi đè data/manifest.csv")
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--splits", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--dest", default=str(ROOT / "data" / "raw" / "rocole"))
    a = ap.parse_args()
    (fetch_images if a.images else build_manifest)(a, Path(a.dest))


if __name__ == "__main__":
    main()
