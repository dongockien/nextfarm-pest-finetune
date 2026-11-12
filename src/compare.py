"""So sánh các nhánh trên cùng test set, chạy các phép kiểm chéo, in bảng và ghi outputs/report.md.

  python src/compare.py
"""
import json
from collections import Counter
from pathlib import Path

from common import LABELS, OPAQUE, normalize_label, parse_production, parse_relaxed, read_jsonl, score

ROOT = Path(__file__).resolve().parents[1]
ARMS = [("baseline_plain", "B0 model gốc, prompt dùng chung"),
        ("baseline_with_labels", "B1 model gốc, prompt có danh sách nhãn"),
        ("ft", "FT sau fine-tune (model trong bộ nhớ)"),
        ("ft_reloaded", "FT nạp lại adapter từ đĩa"),
        ("ft_serve_prompt", "FT hỏi bằng prompt đang chạy thật (\"cây trồng\", L1)"),
        ("ft_opaque", "ĐỐI CHỨNG mã nhãn vô nghĩa, nhãn đúng ảnh"),
        ("ft_shuffled", "ĐỐI CHỨNG mã nhãn vô nghĩa + xáo trộn")]
CONTROL_ARMS = {"ft_opaque", "ft_shuffled"}
OOD_ARMS = [("baseline_plain_ood", "B0 model gốc"), ("ft_ood", "FT sau fine-tune")]
CONTROL_MAX_ACC = 0.6  # 21 ảnh, 3 nhãn: đoán mò ~33%; đạt >= 60% do may rủi có xác suất dưới 1%


def load_preds(out, tag):
    path = out / tag / "preds.jsonl"
    return read_jsonl(path) if path.exists() else None


def pred_label(raw):
    d, _ = parse_relaxed(raw)
    return "INVALID" if d is None else normalize_label(d["primary_diagnosis"]) or "OTHER"


def main():
    out = ROOT / "outputs"
    test = read_jsonl(ROOT / "data" / "splits" / "test.jsonl")
    synthetic = all(r["image_path"].startswith("synthetic/") for r in test)
    lines = [f"# Kết quả smoke test ({len(test)} ảnh test)\n",
             "Chỉ để kiểm tra pipeline. Số ảnh test nhỏ nên KHÔNG đủ để kết luận về độ chính xác.\n"]
    if synthetic:
        lines.append("Dữ liệu GIẢ: accuracy/F1 của B0, B1 không nói gì về chất lượng model gốc (nó nhìn hình vẽ, "
                     "không phải lá). Ở B0/B1 chỉ cột JSON hợp lệ là đáng đọc.\n")
    lines += ["Cột \"Nhãn đúng mã\": primary_diagnosis trùng nguyên văn một mã trong danh sách (không chấm lỏng). "
              "Accuracy thì chấm lỏng theo từ khoá (normalize_label). Độ trễ không so được giữa model gốc và FT vì độ dài output khác nhau.\n",
              "| Nhánh | n | JSON hợp lệ (parser thật) | JSON có primary_diagnosis | Nhãn đúng mã | Accuracy | Macro-F1 | Độ trễ TB (s) |",
              "|---|---|---|---|---|---|---|---|"]
    details, scores, raws = [], {}, {}
    for tag, desc in ARMS:
        recs = load_preds(out, tag)
        if recs is None:
            lines.append(f"| {desc} | | (chưa có {tag}/preds.jsonl) | | | | | |")
            continue
        preds = {r["sample_id"]: r["raw"] for r in recs}
        if tag in CONTROL_ARMS:  # đổi mã vô nghĩa về tên nhãn để chấm chung một thang
            for lab, code in OPAQUE.items():
                preds = {k: v.replace(f'"{code}"', f'"{lab}"') for k, v in preds.items()}
        rows = [r for r in test if r["sample_id"] in preds]
        s = score(rows, preds)
        scores[tag], raws[tag] = s, preds
        lat = [r["latency_s"] for r in recs if "latency_s" in r]
        lines.append(f"| {desc} | {s['n']} | {s['json_valid_production']:.0%} | {s['json_valid_relaxed']:.0%} | "
                     f"{s['label_in_vocab']:.0%} | {s['accuracy']:.0%} | {s['macro_f1']:.2f} | {sum(lat) / len(lat):.1f} |")
        pc = "; ".join(f"{k}: F1 {v['f1']:.2f} (n={v['support']})" for k, v in s["per_class"].items())
        details.append(f"\n## {desc}\n- Theo nhãn: {pc}\n- Ma trận nhầm (hàng = thật): {json.dumps(s['confusion'], ensure_ascii=False)}")

    # ---------- kiểm chéo ----------
    lines.append("\n## Kiểm chéo")
    if "ft_shuffled" in scores:
        acc, n = scores["ft_shuffled"]["accuracy"], scores["ft_shuffled"]["n"]
        verdict = (f"CẢNH BÁO: vẫn cao hơn {CONTROL_MAX_ACC:.0%}. Model đoán đúng dù nhãn train vô nghĩa => "
                   "nghi rò rỉ hoặc lỗi chấm điểm. Không tin điểm FT cho tới khi tìm ra." if acc >= CONTROL_MAX_ACC
                   else "ĐẠT: rơi về gần mức đoán mò, tức điểm FT thật sự đến từ nhãn.")
        lines.append(f"- Đối chứng nhãn xáo trộn: accuracy {acc:.0%} trên {n} ảnh (đoán mò ~{1 / len(LABELS):.0%}). {verdict}")
    else:
        lines.append("- Đối chứng nhãn xáo trộn: CHƯA CHẠY (`python src/train_eval.py --shuffle-labels --max-steps 30`).")
    if "ft_opaque" in scores:
        so = scores["ft_opaque"]
        lines.append(f"- Đối chứng mã nhãn vô nghĩa: accuracy {so['accuracy']:.0%}, macro-F1 {so['macro_f1']:.2f} trên {so['n']} ảnh. "
                     "Model gốc không thể biết class_a/b/c là gì, nên đây là phần học được từ ảnh + nhãn train. "
                     + ("Gần mức đoán mò: với ngần này dữ liệu và step, fine-tune CHƯA học được cách phân biệt."
                        if so["accuracy"] < 0.5 else "Cao hơn hẳn mức đoán mò: fine-tune có học được cách phân biệt từ dữ liệu."))
    else:
        lines.append("- Đối chứng mã nhãn vô nghĩa: CHƯA CHẠY (`python src/train_eval.py --opaque-labels --max-steps 30`).")
    if "ft" in raws and "ft_reloaded" in raws:
        ids = sorted(set(raws["ft"]) & set(raws["ft_reloaded"]))
        same_raw = sum(raws["ft"][i] == raws["ft_reloaded"][i] for i in ids)
        same_lab = sum(pred_label(raws["ft"][i]) == pred_label(raws["ft_reloaded"][i]) for i in ids)
        verdict = "ĐẠT." if same_lab == len(ids) else "CẢNH BÁO: adapter trên đĩa cho kết quả khác model lúc train."
        lines.append(f"- Nạp lại adapter: {same_lab}/{len(ids)} ảnh cùng nhãn, {same_raw}/{len(ids)} ảnh trùng nguyên văn. {verdict}")
    else:
        lines.append("- Nạp lại adapter: CHƯA CÓ nhánh ft_reloaded.")
    if "ft" in scores and "ft_serve_prompt" in scores:
        d = scores["ft_serve_prompt"]["accuracy"] - scores["ft"]["accuracy"]
        lines.append(f"- Lệch prompt L1 (train \"cà phê\", hỏi \"cây trồng\"): accuracy đổi {d:+.0%}, JSON hợp lệ "
                     f"{scores['ft_serve_prompt']['json_valid_production']:.0%}, nhãn đúng mã "
                     f"{scores['ft_serve_prompt']['label_in_vocab']:.0%} (so với {scores['ft']['label_in_vocab']:.0%} khi hỏi đúng prompt train).")

    ood_path = ROOT / "data" / "ood.jsonl"
    if ood_path.exists():
        kinds = {r["sample_id"]: r["kind"] for r in read_jsonl(ood_path)}
        lines.append("\n## Ảnh ngoài lề (nhiễu, thiếu sáng, mờ nặng, không phải lá)")
        lines.append("Bài smoke KHÔNG có nhãn \"không dùng được\", nên kỳ vọng là model FT gán bệnh cho cả ảnh rác. "
                     "Mục này đo mức độ đó; nó không phải cổng \"từ chối đúng >= 90%\" của báo cáo.\n")
        lines += ["| Nhánh | n | Bị gán một nhãn trong danh sách | Confidence TB khi bị gán | Theo loại ảnh (bị gán/tổng) |",
                  "|---|---|---|---|---|"]
        for tag, desc in OOD_ARMS:
            recs = load_preds(out, tag)
            if recs is None:
                lines.append(f"| {desc} | | (chưa có {tag}/preds.jsonl) | | |")
                continue
            hit, tot, confs = Counter(), Counter(), []
            for r in recs:
                k = kinds.get(r["sample_id"], "?")
                tot[k] += 1
                if pred_label(r["raw"]) in LABELS:
                    hit[k] += 1
                    d, _ = parse_production(r["raw"])
                    if d is not None:
                        confs.append(float(d["confidence"]))
            conf = f"{sum(confs) / len(confs):.2f}" if confs else "-"
            by = ", ".join(f"{k} {hit[k]}/{tot[k]}" for k in sorted(tot))
            lines.append(f"| {desc} | {len(recs)} | {sum(hit.values()) / len(recs):.0%} | {conf} | {by} |")

    lines += details
    for name, title in (("train.json", "Huấn luyện"), ("train_opaque.json", "Huấn luyện (đối chứng mã nhãn vô nghĩa)"),
                        ("train_shuffled.json", "Huấn luyện (đối chứng nhãn xáo trộn)")):
        meta = out / "meta" / name
        if not meta.exists():
            continue
        m = json.loads(meta.read_text(encoding="utf-8"))
        lines.append(f"\n## {title}\n- Model: {m['model']}; loss tính trên: {m.get('loss_on', 'all_tokens')}\n- Số step: {m['steps']}, loss train trung bình: {m['train_loss']:.3f}\n"
                     f"- Thời gian: {m['seconds']} giây; VRAM đỉnh: {m['peak_vram_gb']} GB")
        if m.get("val_curve"):
            lines.append("- Loss trên val (step: loss): " + ", ".join(f"{s}: {v}" for s, v in m["val_curve"]))
            vals = [v for _, v in m["val_curve"]]
            if len(vals) >= 2 and vals[-1] > min(vals) * 1.1:
                lines.append("- CHÚ Ý: loss val cuối cao hơn mức thấp nhất trên 10% => có dấu hiệu học vẹt, nên dừng sớm hơn.")
        else:
            lines.append("- Loss trên val: không có (đã tắt --eval-steps hoặc chưa tới step đo).")
    lines.append(f"\nNhãn: {', '.join(LABELS)}. Cách chấm tên bệnh tự do của model gốc: xem normalize_label trong common.py.")
    text = "\n".join(lines)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
