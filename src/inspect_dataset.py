"""In ra cấu trúc thật của một bộ dữ liệu đã tải (không chạy, không giải nén tệp lạ ngoài thư mục chỉ định).

  python src/inspect_dataset.py data/raw/rocole

Kết quả cũng được lưu ở data/inspect_report.txt. Dán nội dung đó cho người viết build_manifest.py:
trang bài báo không nêu tên thư mục/tệp nên phải đọc từ bản tải thật.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
out_lines = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out_lines.append(s)


def short(obj, n=500):
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + " ..."


def peek_csv(path):
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(f, dialect))
    p(f"    dấu phân cách: {dialect.delimiter!r}; số dòng: {len(rows)}")
    for r in rows[:4]:
        p("    ", r)
    if rows:
        for i, name in enumerate(rows[0]):
            vals = Counter(r[i] for r in rows[1:] if i < len(r))
            if 0 < len(vals) <= 12:
                p(f"    cột {name!r}: {dict(vals)}")


def peek_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:  # noqa: BLE001
        p("    không đọc được JSON:", type(e).__name__)
        return
    if isinstance(data, dict):
        p("    JSON object, khoá cấp 1:", list(data)[:20])
        for k, v in list(data.items())[:6]:
            kind = f"list[{len(v)}]" if isinstance(v, list) else type(v).__name__
            p(f"    - {k}: {kind}; ví dụ: {short(v[0] if isinstance(v, list) and v else v, 300)}")
    elif isinstance(data, list):
        p(f"    JSON list, {len(data)} phần tử; phần tử đầu: {short(data[0] if data else None)}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Cách dùng: python src/inspect_dataset.py <thư mục dữ liệu>")
    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        raise SystemExit(f"Không phải thư mục: {root}")
    files = [f for f in root.rglob("*") if f.is_file()]
    p(f"Thư mục: {root}\nTổng số tệp: {len(files)}")
    p("Theo đuôi tệp:", dict(Counter(f.suffix.lower() or "(không đuôi)" for f in files)))
    dirs = Counter(str(f.parent.relative_to(root)) for f in files)
    p("\nThư mục chứa tệp (tối đa 30):")
    for d, c in dirs.most_common(30):
        p(f"  {d or '.'}: {c} tệp")
    imgs = [f for f in files if f.suffix.lower() in IMG_EXT]
    p(f"\nẢnh: {len(imgs)}; tên mẫu: {[f.name for f in sorted(imgs)[:5]]}")
    try:
        from PIL import Image
        for f in sorted(imgs)[:5]:
            with Image.open(f) as im:
                p(f"  {f.name}: {im.size[0]}x{im.size[1]} {im.mode}")
    except ImportError:
        p("  (cài pillow để xem kích thước ảnh)")
    p("\nTệp chú thích:")
    for f in sorted(files):
        ext = f.suffix.lower()
        if ext in IMG_EXT:
            continue
        rel = f.relative_to(root)
        p(f"- {rel} ({f.stat().st_size} byte)")
        if ext in {".csv", ".tsv", ".txt"} and f.stat().st_size < 50_000_000:
            peek_csv(f)
        elif ext == ".json":
            peek_json(f)
        elif ext in {".xml"}:
            p("    đầu tệp:", f.read_text(encoding="utf-8", errors="replace")[:400].replace("\n", " "))
        elif ext in {".xlsx", ".xls"}:
            try:
                import openpyxl
                wb = openpyxl.load_workbook(f, read_only=True)
                for ws in wb.worksheets:
                    p(f"    sheet {ws.title!r}")
                    for row in list(ws.iter_rows(values_only=True))[:4]:
                        p("     ", row)
            except ImportError:
                p("    (cài openpyxl để đọc xlsx: pip install openpyxl)")
    dest = ROOT / "data" / "inspect_report.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nĐã lưu: {dest}")


if __name__ == "__main__":
    main()
