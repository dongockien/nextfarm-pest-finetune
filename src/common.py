"""Hàm dùng chung: prompt, parser mô phỏng nextfarm-ai, chuẩn hoá nhãn, chỉ số.

Mọi thứ ở đây chạy bằng Python chuẩn, không cần GPU.
"""
import csv
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# Nhãn smoke test (RoCoLe). Mã nhãn ổn định, dịch sang tiếng Việt ở tầng hiển thị.
LABELS = ["healthy_leaf", "coffee_leaf_rust", "red_spider_mite"]
SEVERITIES = ["low", "medium", "high"]
CROP_NAME = "cà phê"
# Mã nhãn VÔ NGHĨA cho các lần chạy đối chứng: model gốc không thể đoán nghĩa từ tên,
# nên điểm đạt được chỉ có thể đến từ dữ liệu train.
OPAQUE = {"healthy_leaf": "class_a", "coffee_leaf_rust": "class_b", "red_spider_mite": "class_c"}
# Code đang chạy thật gọi Build("cây trồng", null) (lỗi L1 trong báo cáo): không truyền loại cây.
CROP_SERVED_TODAY = "cây trồng"


def build_prompt(crop=CROP_NAME, description=None, with_labels=False):
    """Sao khuôn PestDetectionPrompt.Build của nextfarm-ai (Infrastructure/Services/PestDetectionPrompt.cs).

    CHÚ Ý: mặc định crop="cà phê" là prompt SAU KHI SỬA lỗi L1, chưa phải prompt đang chạy thật
    (đang chạy thật: crop=CROP_SERVED_TODAY). Nhánh ft_serve_prompt trong train_eval.py đo độ lệch này.
    with_labels=True là biến thể B1 (prompt tốt hơn): thêm danh sách nhãn cho phép.
    Bản train và bản phục vụ PHẢI dùng cùng một prompt.
    """
    desc = description.strip() if description and description.strip() else "(không có)"
    text = (
        f"Phân tích ảnh bệnh/sâu hại trên cây {crop}.\n"
        f"Mô tả thêm từ nông dân: {desc}\n"
        "Chỉ trả về JSON đúng cấu trúc sau, không kèm văn bản nào khác:\n"
        "{\n"
        '  "primary_diagnosis": "...",\n'
        '  "confidence": 0.0,\n'
        '  "severity": "low|medium|high",\n'
        '  "diagnosis_detail": "...",\n'
        '  "suggestions": [ { "action": "...", "products": ["..."], "dosage": "..." } ]\n'
        "}"
    )
    if with_labels:
        text += "\nprimary_diagnosis phải là đúng một trong: " + ", ".join(LABELS) + "."
    return text


# ---------- parser ----------
def strip_fence(text):
    """Giống PestDetectionResponseParser.StripFence: bỏ rào ```json ... ```."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    nl = t.find("\n")
    if nl < 0:
        return t
    body = t[nl + 1:]
    last = body.rfind("```")
    return body[:last] if last >= 0 else body


def _load_obj(text):
    try:
        d = json.loads(strip_fence(text).strip())
    except (ValueError, TypeError):
        return None, "json_khong_hop_le"
    if not isinstance(d, dict):
        return None, "khong_phai_object"
    if not isinstance(d.get("primary_diagnosis"), str):
        return None, "thieu_primary_diagnosis"
    return d, None


def parse_production(text):
    """Luật của parser thật: bắt buộc primary_diagnosis (chuỗi) và confidence (số).

    Thiếu một trong hai thì nextfarm-ai ném FormatException và dòng kết quả thành status=failed.
    Trả về (dict, None) nếu hợp lệ, (None, lý_do) nếu không.
    """
    d, err = _load_obj(text)
    if d is None:
        return None, err
    c = d.get("confidence")
    if isinstance(c, bool) or not isinstance(c, (int, float)):
        return None, "thieu_confidence"
    return d, None


def parse_relaxed(text):
    """Chỉ đòi primary_diagnosis. Dùng để chấm nhãn khi schema chưa thống nhất."""
    return _load_obj(text)


# ---------- chuẩn hoá nhãn ----------
def _fold(text):
    t = unicodedata.normalize("NFD", str(text).lower().replace("đ", "d"))
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def normalize_label(text):
    """Đưa tên bệnh model tự viết (tiếng Việt/Anh) về mã nhãn; không nhận ra thì trả None.

    Baseline (model gốc) không biết mã nhãn của ta nên cần bước này. Nó ảnh hưởng điểm baseline,
    vì vậy báo cáo phải ghi rõ cách chấm.
    """
    t = _fold(text).replace("unhealthy", "")
    found = []
    if any(k in t for k in ("coffee_leaf_rust", "gi sat", "ri sat", "rust", "hemileia", "vastatrix")):
        found.append("coffee_leaf_rust")
    if any(k in t for k in ("red_spider_mite", "nhen do", "spider mite", "red mite", "mite", "tetranychus")):
        found.append("red_spider_mite")
    if any(k in t for k in ("healthy_leaf", "la khoe", "khoe manh", "healthy", "khong bi benh", "binh thuong")):
        found.append("healthy_leaf")
    return found[0] if len(found) == 1 else None


# ---------- chỉ số ----------
def score(rows, preds):
    """rows: list dict có sample_id, label. preds: {sample_id: raw_text}. Trả về dict chỉ số."""
    n = len(rows)
    valid_prod = valid_rel = correct = in_vocab = 0
    confusion = defaultdict(Counter)
    for r in rows:
        raw = preds.get(r["sample_id"], "")
        dprod, _ = parse_production(raw)
        drel, _ = parse_relaxed(raw)
        valid_prod += dprod is not None
        valid_rel += drel is not None
        if drel is None:
            pred = "INVALID"
        else:
            pred = normalize_label(drel["primary_diagnosis"]) or "OTHER"
            in_vocab += drel["primary_diagnosis"] in LABELS  # đúng nguyên văn một mã nhãn, không chấm lỏng
        confusion[r["label"]][pred] += 1
        correct += pred == r["label"]
    per_class = {}
    f1s = []
    for lab in LABELS:
        support = sum(confusion[lab].values())
        tp = confusion[lab][lab]
        fp = sum(confusion[o][lab] for o in LABELS if o != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab] = {"support": support, "precision": prec, "recall": rec, "f1": f1}
        if support:
            f1s.append(f1)
    return {
        "n": n,
        "json_valid_production": valid_prod / n if n else 0.0,
        "json_valid_relaxed": valid_rel / n if n else 0.0,
        "label_in_vocab": in_vocab / n if n else 0.0,
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class": per_class,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


# ---------- I/O ----------
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_manifest(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
