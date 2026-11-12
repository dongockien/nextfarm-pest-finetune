# Thực hành fine-tune nhận diện bệnh cà phê (smoke test)

Thư mục này **nằm ngoài** hai repo nextfarm-ai và nextfarm-fe, không chạm vào chúng.
Mục đích: chạy được cả vòng **dữ liệu → baseline → fine-tune → đo lại → so sánh** để hiểu bài toán.
Đây là bài thử pipeline trên model nhỏ, **không đại diện cho kết quả của Qwen3.8-27B**.

## Tiến độ
| Bước | Trạng thái |
|---|---|
| 1–3. Dữ liệu giả, tách theo cây, tái hiện lỗi L2 | **Xong** (trên máy) |
| 4–6. Train + đo trên dữ liệu giả | **Xong** (Colab T4, Qwen2.5-VL-3B). Kết quả: `outputs\outputs_colab\report.md` |
| 7. Dữ liệu thật RoCoLe | **Xong** (Colab T4, 60 step, ảnh 672). Kết quả: `outputs\outputs_rocole\report.md` |

## Đã kiểm và chưa kiểm
| Phần | Trạng thái |
|---|---|
| `common.py`, `make_synthetic.py`, `prepare_dataset.py`, `validate_dataset.py`, `compare.py`, `inspect_dataset.py` | Đã chạy trên máy (Python 3.12, Windows) |
| `fetch_rocole.py` | Đã chạy trên máy: manifest 1.000 ảnh / 250 cây, tách 99/21/21, tải ảnh, validator `OK: 0 lỗi` |
| `train_eval.py`: nạp model, B0/B1, train, đo val, lưu và nạp lại adapter, ảnh ngoài lề | Đã chạy trên Colab T4 (unsloth 2026.11.4, transformers 5.17.0, trl 1.15.0, torch 2.14.1) |
| `train_eval.py`: loss chỉ trên câu trả lời, `--opaque-labels`, giới hạn 512 token cho model gốc | **Chưa chạy trên GPU** (sửa sau lần chạy đầu). Tham số collator đã đối chiếu với mã nguồn unsloth-zoo |
| Máy 4 GB VRAM có train nổi không | Chưa thử. Trên T4, bản 3B ở ảnh 336px dùng đỉnh 3,08 GB (`max_memory_allocated`), nên có thể vừa sát |

## Bài học từ lần chạy dữ liệu giả (21 ảnh test, chỉ nói về quy trình)
1. **Fine-tune dạy được format**: JSON qua parser thật 21/21, adapter nạp lại từ đĩa cho kết quả trùng 21/21.
2. **Điểm 0% của B0/B1 là lỗi đo**: giới hạn 96 token cắt giữa JSON của model gốc. Lấy nhãn từ phần bị cắt thì B1 đúng 19/21,
   cao hơn FT (17/21). Đã sửa: model gốc được 512 token.
3. **Đối chứng nhãn xáo trộn vẫn đúng 20/21**: không phải rò rỉ. Model gốc tự nhận ra ba lớp ảnh giả và hiểu tên nhãn tiếng Anh,
   nên nhãn train gần như không đóng góp. Đã sửa: đối chứng dùng mã nhãn vô nghĩa `class_a/b/c`.
4. **Lệch prompt L1 có hậu quả đo được**: train với "cà phê", hỏi bằng "cây trồng" thì gỉ sắt từ 7/7 xuống 0/7, model bịa nhãn
   `aphid`, `powdery_mildew`, `scale_insect`.
5. **Model FT không từ chối ảnh rác**: 20/20 ảnh rác bị gán nhãn với confidence 0.9; 16 ảnh nhận nhãn `healthy_coffee_leaf`
   không có trong danh sách. Đã thêm cột "Nhãn đúng mã" để bắt nhãn bịa.
6. **Loss không phản ánh việc học nhãn**: bắt đầu ~5,0, lần thật và lần xáo trộn gần trùng nhau. Nghi do tính loss trên cả prompt
   (chưa xác minh). Đã sửa: loss chỉ tính trên câu trả lời; lần chạy tới sẽ cho biết giả thuyết đúng hay không.

## Bài học từ lần chạy RoCoLe (21 ảnh test thật)
1. **Fine-tune chỉ dạy format, chưa dạy nhìn**: FT đúng 12/21, B1 (không train, chỉ thêm danh sách nhãn vào prompt) đúng 11/21.
   Nhánh mã nhãn vô nghĩa chỉ đúng 6/21 (mức đoán mò), tức 33 ảnh/nhãn × 60 step chưa đủ để model 3B học phân biệt từ ảnh.
2. **B1 đã đạt JSON hợp lệ 100% và nhãn đúng mã 100%** mà không cần train. Format không phải lý do để fine-tune.
3. **Đối chứng xáo trộn giờ hoạt động**: 7/21, model dồn hết về một lớp.
4. **Nhện đỏ là lớp khó**: B1 gọi cả 7 ảnh nhện đỏ là gỉ sắt; FT đúng 3/7.
5. **L1 không lặp lại**: hỏi bằng "cây trồng" cho output trùng nguyên văn 21/21 với prompt train. Hiệu ứng lớn ở lần chạy dữ liệu giả
   nhiều khả năng do model lúc đó train chưa tới; đừng dùng con số đó làm bằng chứng.
6. **Loss val gần như không đo việc chọn nhãn**: trong ~28 token trả lời chỉ 1–2 token quyết định nhãn, nên đoán mò cũng cho loss ~0,04.
   Cảnh báo "học vẹt" trong report lần này là nhiễu. Cần đo accuracy trên val thay cho loss.
7. **Lỗi script lần chạy này (đã sửa, chưa chạy lại)**: collator của Unsloth tự thu ảnh train về 512 trong khi lúc đo dùng 672;
   loss val của hai nhánh đối chứng tính trên tên nhãn gốc nên vô nghĩa.

## Chạy trên Colab (bước 7: RoCoLe)
1. Vào <https://colab.research.google.com>, `File > Upload notebook`, chọn `pest_finetune.ipynb` (bản mới, tải lại từ máy).
2. `Runtime > Change runtime type > T4 GPU > Save`.
3. Ô 0: chọn `nextfarm-pest-finetune-colab.zip` (thư mục gốc dự án; bản mới). Phải in tên GPU `Tesla T4`.
4. Ô đặt biến: giữ `DATASET = "rocole"`. `IMG = 672` là cạnh dài ảnh đưa vào model: ảnh gốc 2048×1152, ở 336 thì vết nhện đỏ gần như không còn thấy. Nếu hết bộ nhớ GPU, hạ `IMG` xuống 448.
5. Chạy lần lượt:

| Ô | Việc | Kết quả đúng |
|---|---|---|
| 1 | Tải nhãn RoCoLe, manifest, tách theo cây, tải ~141 ảnh, validator | `Giấy phép theo API: CC BY 4.0`, rồi `OK: 0 lỗi.` |
| 1b | Hiện 2 ảnh train mỗi nhãn | Nhìn thấy lá thật và target tương ứng |
| 2 | Cài unsloth | In `True Tesla T4` |
| 3 | Kiểm nhanh 3 step, 3 ảnh | Chạy hết, in `peak_vram_gb` |
| 4 | Chạy đủ 60 step | `Xong. Chạy: python src/compare.py` |
| 5 | Đối chứng A: mã nhãn vô nghĩa | `Xong đối chứng` |
| 6 | Đối chứng B: mã vô nghĩa + xáo trộn | `Xong đối chứng` |
| 7 | Bảng so sánh | In báo cáo |
| 8 | Tải `outputs_rocole.zip` | Giải nén (Extract All) vào `outputs\` → `outputs\outputs_rocole\` |

Ô 3 lỗi thì dừng và gửi nguyên traceback. Phiên Colab miễn phí mất hết tệp khi ngắt kết nối: tải kết quả ngay sau ô 7.
Ảnh RoCoLe là dữ liệu công khai nên được phép; **không đưa ảnh khách hàng NextFarm lên cloud** khi chưa được duyệt.

## Dữ liệu RoCoLe: điều đã xác minh và giới hạn
- Nguồn: Mendeley, DOI 10.17632/c5yvn32dzg.2, giấy phép **CC BY 4.0** (đọc từ API của Mendeley). Dùng thì phải ghi nguồn.
- Tệp nhãn `RoCoLe-classes.xlsx`: 1.560 ảnh, 390 cây × 4 ảnh. Khoẻ 791, nhện đỏ 167, gỉ sắt 602 (mức 1: 344, 2: 166, 3: 62, 4: 30).
  Các số này đã đối chiếu trực tiếp với tệp nhãn.
- Tên ảnh `C<lô>P<cây><H|E><số>.jpg`; `group_id` là phần `C..P..`. Nhãn lấy từ tệp nhãn, không suy từ chữ H/E trong tên
  (có ảnh tên E nhưng nhãn là khoẻ).
- **Giới hạn**: API chỉ liệt kê 1.000/1.560 ảnh (250 cây: khoẻ 502, gỉ sắt 363, nhện đỏ 135). Smoke test chỉ chọn trong 1.000 ảnh này.
  Muốn đủ bộ phải tải `RoCoLe-voc.tar.gz` (~700 MB; tên tệp bên trong là mã Labelbox, phải ghép với cột `External ID` của `RoCoLE-csv.csv`).
- Ánh xạ mức nặng là **đề xuất, cần kỹ sư bảo vệ thực vật chốt**: gỉ sắt mức 1 → low, 2 → medium, 3–4 → high.
  RoCoLe không có mức nặng cho lá khoẻ và nhện đỏ; `low` ở đó là giá trị giữ chỗ.
- `confidence = 0.9` trong target là số giữ chỗ để qua parser, không phải xác suất.

## Đọc kết quả (`report.md`)
- Bảng chính: JSON hợp lệ theo parser thật, **Nhãn đúng mã** (chấm chặt), Accuracy và Macro-F1 (chấm lỏng theo từ khoá).
- Kiểm chéo:

| Phép kiểm | Nhánh | Kết quả đúng |
|---|---|---|
| Mã nhãn vô nghĩa | `ft_opaque` | Phần model thật sự học từ dữ liệu. Gần 33% ⇒ chưa học được cách phân biệt |
| Mã vô nghĩa + xáo trộn | `ft_shuffled` | ~33%. Từ 60% trở lên ⇒ rò rỉ hoặc chấm sai |
| Nạp lại adapter | `ft_reloaded` | Cùng nhãn với `ft` trên mọi ảnh |
| Lệch prompt L1 | `ft_serve_prompt` | Đo độ lệch khi hỏi bằng "cây trồng" như code đang chạy thật |
| Ảnh ngoài lề | `baseline_plain_ood`, `ft_ood` | Kỳ vọng FT gán bệnh cho ảnh rác: bài smoke chưa có nhãn "không dùng được" |
| Loss trên val | `val_curve` trong `meta\train.json` | Giảm theo train; tăng lại ⇒ học vẹt |

- 21 ảnh test: **không kết luận độ chính xác**. Mỗi ảnh là ~5 điểm phần trăm.
- Train và đo dùng cùng thứ tự ảnh-rồi-chữ; chưa đối chiếu với thứ tự mà `GatewayVisionService` gửi thật.

## Chạy trên máy (tuỳ chọn)
```powershell
cd C:\Users\Acer\Projects\nextfarm-pest-finetune
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"

# Dữ liệu giả (bước 2–3)
python src/make_synthetic.py --force
python src/prepare_dataset.py
python src/validate_dataset.py                 # OK: 0 lỗi
python src/prepare_dataset.py --schema minimal # tái hiện L2; nhớ chạy lại prepare_dataset.py sau đó

# Dữ liệu thật: tải 141 ảnh (~190 MB), GHI ĐÈ manifest và splits giả
python src/fetch_rocole.py --force
python src/prepare_dataset.py
python src/fetch_rocole.py --images
python src/validate_dataset.py

# Train trên máy (cần ≥ 15 GB ổ trống; cài theo trang Windows của Unsloth)
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install --no-cache-dir unsloth
python src/train_eval.py --skip-baseline --max-steps 3 --limit-test 3
python src/train_eval.py --max-steps 60
python src/train_eval.py --max-steps 60 --opaque-labels
python src/train_eval.py --max-steps 60 --shuffle-labels
python src/compare.py
```
Hết bộ nhớ GPU: thử lần lượt `--img-size 224`, `--no-vision-layers`, `--r 8`. Model mặc định khi không truyền `--model` là
`unsloth/Qwen2-VL-2B-Instruct-bnb-4bit`.

## Cấu trúc
```
src/common.py            prompt dùng chung, parser mô phỏng, chuẩn hoá nhãn, chỉ số
src/make_synthetic.py    sinh dữ liệu giả + ảnh ngoài lề
src/fetch_rocole.py      tải nhãn và ảnh RoCoLe, dựng manifest
src/inspect_dataset.py   xem cấu trúc một bộ dữ liệu tải tay (không cần cho RoCoLe nữa)
src/prepare_dataset.py   manifest -> tách theo nhóm -> JSONL
src/validate_dataset.py  kiểm ảnh, nhãn, prompt, parser, rò rỉ nhóm
src/train_eval.py        baseline -> QLoRA -> đo lại + các nhánh đối chứng (GPU)
src/compare.py           bảng so sánh + kiểm chéo
pest_finetune.ipynb      chạy các bước trên bằng notebook (Colab)
```
