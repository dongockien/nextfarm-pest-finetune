# Kết quả smoke test (21 ảnh test)

Chỉ để kiểm tra pipeline. Số ảnh test nhỏ nên KHÔNG đủ để kết luận về độ chính xác.

Cột "Nhãn đúng mã": primary_diagnosis trùng nguyên văn một mã trong danh sách (không chấm lỏng). Accuracy thì chấm lỏng theo từ khoá (normalize_label). Độ trễ không so được giữa model gốc và FT vì độ dài output khác nhau.

| Nhánh | n | JSON hợp lệ (parser thật) | JSON có primary_diagnosis | Nhãn đúng mã | Accuracy | Macro-F1 | Độ trễ TB (s) |
|---|---|---|---|---|---|---|---|
| B0 model gốc, prompt dùng chung | 21 | 100% | 100% | 0% | 0% | 0.00 | 5.6 |
| B1 model gốc, prompt có danh sách nhãn | 21 | 100% | 100% | 100% | 52% | 0.44 | 4.2 |
| FT sau fine-tune (model trong bộ nhớ) | 21 | 100% | 100% | 100% | 57% | 0.57 | 1.3 |
| FT nạp lại adapter từ đĩa | 21 | 100% | 100% | 100% | 57% | 0.57 | 1.2 |
| FT hỏi bằng prompt đang chạy thật ("cây trồng", L1) | 21 | 100% | 100% | 100% | 57% | 0.57 | 1.2 |
| ĐỐI CHỨNG mã nhãn vô nghĩa, nhãn đúng ảnh | 21 | 100% | 100% | 100% | 29% | 0.24 | 1.5 |
| ĐỐI CHỨNG mã nhãn vô nghĩa + xáo trộn | 21 | 100% | 100% | 100% | 33% | 0.17 | 1.5 |

## Kiểm chéo
- Đối chứng nhãn xáo trộn: accuracy 33% trên 21 ảnh (đoán mò ~33%). ĐẠT: rơi về gần mức đoán mò, tức điểm FT thật sự đến từ nhãn.
- Đối chứng mã nhãn vô nghĩa: accuracy 29%, macro-F1 0.24 trên 21 ảnh. Model gốc không thể biết class_a/b/c là gì, nên đây là phần học được từ ảnh + nhãn train. Gần mức đoán mò: với ngần này dữ liệu và step, fine-tune CHƯA học được cách phân biệt.
- Nạp lại adapter: 21/21 ảnh cùng nhãn, 21/21 ảnh trùng nguyên văn. ĐẠT.
- Lệch prompt L1 (train "cà phê", hỏi "cây trồng"): accuracy đổi +0%, JSON hợp lệ 100%, nhãn đúng mã 100% (so với 100% khi hỏi đúng prompt train).

## Ảnh ngoài lề (nhiễu, thiếu sáng, mờ nặng, không phải lá)
Bài smoke KHÔNG có nhãn "không dùng được", nên kỳ vọng là model FT gán bệnh cho cả ảnh rác. Mục này đo mức độ đó; nó không phải cổng "từ chối đúng >= 90%" của báo cáo.

| Nhánh | n | Bị gán một nhãn trong danh sách | Confidence TB khi bị gán | Theo loại ảnh (bị gán/tổng) |
|---|---|---|---|---|
| B0 model gốc | 20 | 0% | - | blur 0/5, dark 0/5, noise 0/5, object 0/5 |
| FT sau fine-tune | 20 | 100% | 0.90 | blur 5/5, dark 5/5, noise 5/5, object 5/5 |

## B0 model gốc, prompt dùng chung
- Theo nhãn: healthy_leaf: F1 0.00 (n=7); coffee_leaf_rust: F1 0.00 (n=7); red_spider_mite: F1 0.00 (n=7)
- Ma trận nhầm (hàng = thật): {"coffee_leaf_rust": {"OTHER": 7}, "red_spider_mite": {"OTHER": 7}, "healthy_leaf": {"OTHER": 7}}

## B1 model gốc, prompt có danh sách nhãn
- Theo nhãn: healthy_leaf: F1 0.77 (n=7); coffee_leaf_rust: F1 0.55 (n=7); red_spider_mite: F1 0.00 (n=7)
- Ma trận nhầm (hàng = thật): {"coffee_leaf_rust": {"coffee_leaf_rust": 6, "healthy_leaf": 1}, "red_spider_mite": {"coffee_leaf_rust": 7}, "healthy_leaf": {"coffee_leaf_rust": 2, "healthy_leaf": 5}}

## FT sau fine-tune (model trong bộ nhớ)
- Theo nhãn: healthy_leaf: F1 0.67 (n=7); coffee_leaf_rust: F1 0.63 (n=7); red_spider_mite: F1 0.43 (n=7)
- Ma trận nhầm (hàng = thật): {"coffee_leaf_rust": {"coffee_leaf_rust": 5, "red_spider_mite": 2}, "red_spider_mite": {"coffee_leaf_rust": 3, "red_spider_mite": 3, "healthy_leaf": 1}, "healthy_leaf": {"coffee_leaf_rust": 1, "healthy_leaf": 4, "red_spider_mite": 2}}

## FT nạp lại adapter từ đĩa
- Theo nhãn: healthy_leaf: F1 0.67 (n=7); coffee_leaf_rust: F1 0.63 (n=7); red_spider_mite: F1 0.43 (n=7)
- Ma trận nhầm (hàng = thật): {"coffee_leaf_rust": {"coffee_leaf_rust": 5, "red_spider_mite": 2}, "red_spider_mite": {"coffee_leaf_rust": 3, "red_spider_mite": 3, "healthy_leaf": 1}, "healthy_leaf": {"coffee_leaf_rust": 1, "healthy_leaf": 4, "red_spider_mite": 2}}

## FT hỏi bằng prompt đang chạy thật ("cây trồng", L1)
- Theo nhãn: healthy_leaf: F1 0.67 (n=7); coffee_leaf_rust: F1 0.63 (n=7); red_spider_mite: F1 0.43 (n=7)
- Ma trận nhầm (hàng = thật): {"coffee_leaf_rust": {"coffee_leaf_rust": 5, "red_spider_mite": 2}, "red_spider_mite": {"coffee_leaf_rust": 3, "red_spider_mite": 3, "healthy_leaf": 1}, "healthy_leaf": {"coffee_leaf_rust": 1, "healthy_leaf": 4, "red_spider_mite": 2}}

## ĐỐI CHỨNG mã nhãn vô nghĩa, nhãn đúng ảnh
- Theo nhãn: healthy_leaf: F1 0.00 (n=7); coffee_leaf_rust: F1 0.43 (n=7); red_spider_mite: F1 0.29 (n=7)
- Ma trận nhầm (hàng = thật): {"coffee_leaf_rust": {"red_spider_mite": 4, "coffee_leaf_rust": 3}, "red_spider_mite": {"coffee_leaf_rust": 4, "red_spider_mite": 3}, "healthy_leaf": {"red_spider_mite": 7}}

## ĐỐI CHỨNG mã nhãn vô nghĩa + xáo trộn
- Theo nhãn: healthy_leaf: F1 0.50 (n=7); coffee_leaf_rust: F1 0.00 (n=7); red_spider_mite: F1 0.00 (n=7)
- Ma trận nhầm (hàng = thật): {"coffee_leaf_rust": {"healthy_leaf": 7}, "red_spider_mite": {"healthy_leaf": 7}, "healthy_leaf": {"healthy_leaf": 7}}

## Huấn luyện
- Model: unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit; loss tính trên: responses_only
- Số step: 60, loss train trung bình: 0.222
- Thời gian: 314.4 giây; VRAM đỉnh: 3.11 GB
- Loss trên val (step: loss): 10: 0.1388, 20: 0.0322, 30: 0.0458, 40: 0.0311, 50: 0.0276, 60: 0.0318
- CHÚ Ý: loss val cuối cao hơn mức thấp nhất trên 10% => có dấu hiệu học vẹt, nên dừng sớm hơn.

## Huấn luyện (đối chứng mã nhãn vô nghĩa)
- Model: unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit; loss tính trên: responses_only
- Số step: 60, loss train trung bình: 0.269
- Thời gian: 320.5 giây; VRAM đỉnh: 3.11 GB
- Loss trên val (step: loss): 10: 0.5205, 20: 0.8409, 30: 0.8721, 40: 0.874, 50: 0.8918, 60: 0.895
- CHÚ Ý: loss val cuối cao hơn mức thấp nhất trên 10% => có dấu hiệu học vẹt, nên dừng sớm hơn.

## Huấn luyện (đối chứng nhãn xáo trộn)
- Model: unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit; loss tính trên: responses_only
- Số step: 60, loss train trung bình: 0.274
- Thời gian: 317.8 giây; VRAM đỉnh: 3.11 GB
- Loss trên val (step: loss): 10: 0.5121, 20: 0.8258, 30: 0.9178, 40: 0.9227, 50: 0.9268, 60: 0.9322
- CHÚ Ý: loss val cuối cao hơn mức thấp nhất trên 10% => có dấu hiệu học vẹt, nên dừng sớm hơn.

Nhãn: healthy_leaf, coffee_leaf_rust, red_spider_mite. Cách chấm tên bệnh tự do của model gốc: xem normalize_label trong common.py.