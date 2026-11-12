"""manifest.csv -> tách train/val/test THEO NHÓM (cây/vườn) -> JSONL.

Hợp đồng manifest.csv (cột): sample_id, image_path (tương đối so với data/raw), label, severity,
group_id, source_dataset.  group_id là mã cây/vườn; ảnh cùng group_id luôn nằm cùng một tập.

  python src/prepare_dataset.py                       # mặc định: schema prod, 33/7/7 ảnh mỗi nhãn
  python src/prepare_dataset.py --schema minimal      # tái hiện lỗi L2 (thiếu confidence)
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import LABELS, SEVERITIES, build_prompt, read_manifest, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIDENCE = 0.9  # SỐ GIỮ CHỖ: chỉ để mẫu qua được parser thật; không phải xác suất (xem README)


def make_target(row, schema):
    t = {"primary_diagnosis": row["label"]}
    if schema == "prod":
        t["confidence"] = SMOKE_CONFIDENCE
    t["severity"] = row["severity"]
    return json.dumps(t, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--schema", choices=["prod", "minimal"], default="prod")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    ap.add_argument("--train-per-class", type=int, default=33)
    ap.add_argument("--val-per-class", type=int, default=7)
    ap.add_argument("--test-per-class", type=int, default=7)
    ap.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()

    rows = read_manifest(a.manifest)
    if not rows:
        raise SystemExit("Manifest rỗng")
    bad = [r["sample_id"] for r in rows if r["label"] not in LABELS or r["severity"] not in SEVERITIES]
    if bad:
        raise SystemExit(f"Nhãn/severity ngoài danh sách ở {len(bad)} dòng, ví dụ {bad[:3]}. Nhãn cho phép: {LABELS}")

    no_group = [r for r in rows if not (r.get("group_id") or "").strip()]
    if no_group:
        print(f"CẢNH BÁO: {len(no_group)} dòng không có group_id, dùng sample_id làm nhóm => ảnh cùng cây có thể "
              "rơi vào hai tập (rò rỉ). Điểm cao khi đó không đáng tin.")
        for r in no_group:
            r["group_id"] = r["sample_id"]

    # 1) tách NHÓM, không tách ảnh
    groups = sorted({r["group_id"] for r in rows})
    rng = random.Random(a.seed)
    rng.shuffle(groups)
    n = len(groups)
    n_tr, n_va = round(a.ratios[0] * n), round(a.ratios[1] * n)
    assign = {}
    for i, g in enumerate(groups):
        assign[g] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"

    # 2) giới hạn số ảnh mỗi nhãn trong từng tập (để tập nhỏ, cân bằng)
    caps = {"train": a.train_per_class, "val": a.val_per_class, "test": a.test_per_class}
    by = defaultdict(list)
    for r in rows:
        by[(assign[r["group_id"]], r["label"])].append(r)
    splits, warnings = defaultdict(list), []
    for (sp, lab), items in sorted(by.items()):
        items.sort(key=lambda r: r["sample_id"])
        rng.shuffle(items)
        if len(items) < caps[sp]:
            warnings.append(f"{sp}/{lab}: chỉ có {len(items)} ảnh, thiếu so với {caps[sp]}")
        splits[sp].extend(items[: caps[sp]])

    # 3) kiểm không rò rỉ nhóm
    seen = {}
    for sp, items in splits.items():
        for r in items:
            assert seen.setdefault(r["group_id"], sp) == sp, f"Nhóm {r['group_id']} nằm ở hai tập"

    prompt = build_prompt()
    out = Path(a.out_dir)
    summary = {"schema": a.schema, "seed": a.seed, "so_nhom": n, "canh_bao": warnings, "tap": {}}
    for sp in ("train", "val", "test"):
        items = sorted(splits[sp], key=lambda r: r["sample_id"])
        write_jsonl(out / f"{sp}.jsonl", [{
            "sample_id": r["sample_id"], "image_path": r["image_path"], "label": r["label"],
            "group_id": r["group_id"], "prompt": prompt, "target": make_target(r, a.schema),
        } for r in items])
        summary["tap"][sp] = {"anh": len(items), "nhom": len({r["group_id"] for r in items}),
                              "theo_nhan": dict(Counter(r["label"] for r in items))}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if warnings:
        print("\nCÓ CẢNH BÁO ở trên (thiếu ảnh cho một số nhãn).")


if __name__ == "__main__":
    main()
