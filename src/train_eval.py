"""Baseline -> QLoRA -> đo lại, trong CÙNG một tiến trình và cùng một cách nạp model (so sánh công bằng).

CHƯA ĐƯỢC CHẠY THỬ TRÊN GPU khi viết (người viết không có GPU). Tên hàm Unsloth lấy từ tài liệu
Unsloth và các notebook mẫu; nếu lệch phiên bản thì báo lỗi cho người viết, đừng tự đoán.

Các nhánh đo (mỗi nhánh ghi outputs/<tag>/preds.jsonl):
  baseline_plain        B0: model gốc, prompt dùng chung
  baseline_with_labels  B1: model gốc, prompt có danh sách nhãn
  baseline_plain_ood    model gốc trên ảnh ngoài lề (ảnh rác)
  ft                    model vừa train, còn trong bộ nhớ
  ft_serve_prompt       model vừa train, nhưng hỏi bằng prompt code đang chạy thật ("cây trồng", lỗi L1)
  ft_ood                model vừa train trên ảnh ngoài lề
  ft_reloaded           adapter NẠP LẠI TỪ ĐĨA (thứ sẽ đem đi phục vụ), phải khớp với ft
  ft_opaque             đối chứng: train với mã nhãn vô nghĩa (class_a/b/c), nhãn ĐÚNG ảnh.
                        Điểm ở đây là phần model THẬT SỰ học từ dữ liệu, không nhờ hiểu tên nhãn
  ft_shuffled           đối chứng: mã nhãn vô nghĩa VÀ xáo trộn, phải rơi về mức đoán mò

Ví dụ:
  python src/train_eval.py --max-steps 3 --limit-test 3 --skip-baseline   # kiểm VRAM nhanh
  python src/train_eval.py --max-steps 30                                  # chạy đủ
  python src/train_eval.py --max-steps 30 --opaque-labels                  # đối chứng mã nhãn vô nghĩa
  python src/train_eval.py --max-steps 30 --shuffle-labels                 # đối chứng nhãn xáo trộn
"""
import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
                    help="repo HF 4-bit; trên T4 dùng unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit")
    ap.add_argument("--splits", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--ood", default=str(ROOT / "data" / "ood.jsonl"),
                    help="danh sách ảnh ngoài lề (make_synthetic.py tạo); không có tệp thì bỏ qua")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--img-size", type=int, default=336, help="cạnh dài tối đa của ảnh; 4GB VRAM: 224-336")
    ap.add_argument("--max-steps", type=int, default=60, help=">0: dùng max_steps; =0: dùng --epochs")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-new-tokens", type=int, default=96, help="giới hạn token khi đo model đã fine-tune")
    ap.add_argument("--baseline-max-new-tokens", type=int, default=512,
                    help="giới hạn token khi đo model gốc: nó viết đủ 5 trường của prompt, 96 token sẽ cắt giữa JSON")
    ap.add_argument("--loss-on-all-tokens", action="store_true",
                    help="tính loss trên cả prompt (cách cũ). Mặc định chỉ tính trên câu trả lời")
    ap.add_argument("--limit-test", type=int, default=0, help=">0: chỉ đo n ảnh test đầu tiên")
    ap.add_argument("--eval-steps", type=int, default=10,
                    help="đo loss trên tập val mỗi n step; 0 = tắt (dùng khi bước đo val gây lỗi)")
    ap.add_argument("--no-vision-layers", action="store_true",
                    help="không train phần nhìn (tiết kiệm VRAM; thử khi bị hết bộ nhớ)")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-reload", action="store_true", help="không nạp lại adapter từ đĩa để đo")
    ap.add_argument("--opaque-labels", action="store_true",
                    help="ĐỐI CHỨNG: thay tên nhãn bằng mã vô nghĩa class_a/b/c. Chỉ train + đo nhánh ft_opaque")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="ĐỐI CHỨNG: mã vô nghĩa VÀ xáo trộn nhãn train. Chỉ train + đo nhánh ft_shuffled")
    return ap.parse_args()


def load_image(raw, rel, size):
    from PIL import Image
    im = Image.open(Path(raw) / rel).convert("RGB")
    im.thumbnail((size, size))
    return im


def user_content(prompt, image=None):
    """Thứ tự ảnh-rồi-chữ, DÙNG CHUNG cho train và đo. Lúc đo ảnh được truyền riêng nên image=None."""
    img = {"type": "image"} if image is None else {"type": "image", "image": image}
    return [img, {"type": "text", "text": prompt}]


def predict(model, tokenizer, image, prompt, max_new_tokens):
    import torch
    messages = [{"role": "user", "content": user_content(prompt)}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, do_sample=False)
    n_in = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][n_in:], skip_special_tokens=True), time.time() - t0


def main():
    a = parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    import unsloth  # noqa: F401  PHẢI import trước transformers/trl
    from unsloth import FastVisionModel
    import torch
    from common import CROP_SERVED_TODAY, OPAQUE, build_prompt, read_jsonl, write_jsonl

    train_rows = read_jsonl(Path(a.splits) / "train.jsonl")
    val_rows = read_jsonl(Path(a.splits) / "val.jsonl")
    test_rows = read_jsonl(Path(a.splits) / "test.jsonl")
    ood_rows = read_jsonl(a.ood) if Path(a.ood).exists() else []
    if a.limit_test:
        test_rows, ood_rows = test_rows[: a.limit_test], ood_rows[: a.limit_test]
    out = Path(a.out)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    control = a.shuffle_labels or a.opaque_labels
    suffix = "_shuffled" if a.shuffle_labels else "_opaque" if a.opaque_labels else ""
    if control:
        def opaque(target):  # đổi tên nhãn thành mã vô nghĩa, giữ nguyên các trường khác
            d = json.loads(target)
            d["primary_diagnosis"] = OPAQUE[d["primary_diagnosis"]]
            return json.dumps(d, ensure_ascii=False)
        targets = [opaque(r["target"]) for r in train_rows]
        if a.shuffle_labels:
            random.Random(a.seed).shuffle(targets)
            print("ĐỐI CHỨNG: mã nhãn vô nghĩa + xáo trộn. Kỳ vọng accuracy trên test rơi về mức đoán mò.")
        else:
            print("ĐỐI CHỨNG: mã nhãn vô nghĩa, nhãn đúng ảnh. Accuracy ở đây là phần học được từ dữ liệu.")
        train_rows = [dict(r, target=t) for r, t in zip(train_rows, targets)]
        # val phải cùng bộ mã với train (không xáo), nếu không loss val của nhánh đối chứng vô nghĩa
        val_rows = [dict(r, target=opaque(r["target"])) for r in val_rows]

    def eval_rows(model, tokenizer, tag, rows, prompt_fn, max_new_tokens=a.max_new_tokens):
        FastVisionModel.for_inference(model)
        recs = []
        for i, r in enumerate(rows, 1):
            img = load_image(a.raw, r["image_path"], a.img_size)
            text, dt = predict(model, tokenizer, img, prompt_fn(r), max_new_tokens)
            recs.append({"sample_id": r["sample_id"], "raw": text, "latency_s": round(dt, 3)})
            print(f"  [{tag}] {i}/{len(rows)} {dt:.1f}s {text[:80]!r}")
        write_jsonl(out / tag / "preds.jsonl", recs)

    shared = lambda r: r.get("prompt") or build_prompt()  # noqa: E731  prompt dùng chung train/đo

    print(f"Nạp {a.model} (4-bit) ...")
    model, tokenizer = FastVisionModel.from_pretrained(a.model, load_in_4bit=True,
                                                       use_gradient_checkpointing="unsloth")

    if not a.skip_baseline and not control:
        big = a.baseline_max_new_tokens
        eval_rows(model, tokenizer, "baseline_plain", test_rows, shared, big)                          # B0
        eval_rows(model, tokenizer, "baseline_with_labels", test_rows,
                  lambda r: build_prompt(with_labels=True), big)                                       # B1
        if ood_rows:
            eval_rows(model, tokenizer, "baseline_plain_ood", ood_rows, shared, big)

    if a.skip_train:
        print("Xong (bỏ qua train). Chạy: python src/compare.py")
        return

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=not a.no_vision_layers, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=a.r, lora_alpha=a.r, lora_dropout=0, bias="none", random_state=a.seed,
        use_rslora=False, loftq_config=None, target_modules="all-linear",
    )
    FastVisionModel.for_training(model)
    from trl import SFTConfig, SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    def to_conv(r):
        return {"messages": [
            {"role": "user", "content": user_content(r["prompt"], load_image(a.raw, r["image_path"], a.img_size))},
            {"role": "assistant", "content": [{"type": "text", "text": r["target"]}]},
        ]}

    dataset = [to_conv(r) for r in train_rows]
    bf16 = torch.cuda.is_bf16_supported()
    cfg = dict(
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum, warmup_steps=5,
        learning_rate=a.lr, logging_steps=1, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=a.seed, output_dir=str(out / f"trainer{suffix}"), report_to="none",
        bf16=bf16, fp16=not bf16, remove_unused_columns=False, dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True}, dataloader_num_workers=0,
    )
    if a.max_steps > 0:
        cfg["max_steps"] = a.max_steps
    else:
        cfg["num_train_epochs"] = a.epochs
    eval_dataset = None
    if a.eval_steps > 0 and val_rows:
        # Loss trên tập val trong lúc train: val xấu đi khi train tốt lên là dấu hiệu học vẹt.
        # prediction_loss_only: chỉ giữ loss, không gom logits (gom logits sẽ ngốn hàng GB RAM).
        eval_dataset = [to_conv(r) for r in val_rows]
        every = min(a.eval_steps, a.max_steps) if a.max_steps > 0 else a.eval_steps
        cfg.update(eval_strategy="steps", eval_steps=every, per_device_eval_batch_size=1,
                   prediction_loss_only=True)
    try:  # tên tham số độ dài đổi theo phiên bản TRL; None = không cắt chuỗi (tránh cắt mất token ảnh)
        args = SFTConfig(**cfg, max_length=None)
    except TypeError:
        args = SFTConfig(**cfg, max_seq_length=4096)
    # Loss chỉ tính trên câu trả lời của assistant. Tính trên cả prompt thì đường loss chủ yếu đo việc
    # thuộc lòng prompt cố định (lần chạy đầu: loss bắt đầu ~5.0, không phân biệt được nhãn thật với nhãn xáo).
    # Hai mốc dưới là khuôn ChatML của Qwen-VL; đổi họ model thì phải đổi theo chat template của nó.
    # resize="max": collator KHÔNG tự thu ảnh. Mặc định ("min") nó thu ảnh về 512 lúc train, trong khi lúc đo
    # ảnh giữ cỡ --img-size => train và đo nhìn ảnh ở hai độ phân giải khác nhau (lần chạy RoCoLe 672 bị đúng lỗi này).
    if a.loss_on_all_tokens:
        collator = UnslothVisionDataCollator(model, tokenizer, resize="max")
    else:
        collator = UnslothVisionDataCollator(model, tokenizer, resize="max", train_on_responses_only=True,
                                             instruction_part="<|im_start|>user\n",
                                             response_part="<|im_start|>assistant\n")
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, data_collator=collator,
                         train_dataset=dataset, eval_dataset=eval_dataset, args=args)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    stats = trainer.train()
    hist = trainer.state.log_history
    meta = {"model": a.model, "steps": trainer.state.global_step, "train_loss": stats.training_loss,
            "seconds": round(time.time() - t0, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "train_curve": [[h["step"], round(h["loss"], 4)] for h in hist if "loss" in h],
            "val_curve": [[h["step"], round(h["eval_loss"], 4)] for h in hist if "eval_loss" in h],
            "loss_on": "all_tokens" if a.loss_on_all_tokens else "responses_only",
            "control": suffix.lstrip("_") or None, "args": vars(a)}
    (out / "meta" / f"train{suffix}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print("Huấn luyện xong:", {k: v for k, v in meta.items() if k not in ("args", "train_curve")})

    adapter_dir = out / f"adapter{suffix}"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    eval_rows(model, tokenizer, f"ft{suffix}", test_rows, shared)
    if control:
        print("Xong đối chứng. Chạy: python src/compare.py")
        return
    # L1: code đang chạy thật gọi Build("cây trồng", null). Đo xem model train với "cà phê" có chịu được không.
    eval_rows(model, tokenizer, "ft_serve_prompt", test_rows, lambda r: build_prompt(crop=CROP_SERVED_TODAY))
    if ood_rows:
        eval_rows(model, tokenizer, "ft_ood", ood_rows, shared)

    if not a.skip_reload:
        # Thứ đem đi phục vụ là adapter trên đĩa, không phải model trong bộ nhớ: nạp lại rồi đo, phải khớp nhánh ft.
        print(f"Giải phóng model, nạp lại adapter từ {adapter_dir} ...")
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        model, tokenizer = FastVisionModel.from_pretrained(str(adapter_dir), load_in_4bit=True)
        eval_rows(model, tokenizer, "ft_reloaded", test_rows, shared)
    print("Xong. Chạy: python src/compare.py")


if __name__ == "__main__":
    main()
