# Kết quả smoke test (21 ảnh test)

Chỉ để kiểm tra pipeline. Số ảnh test nhỏ nên KHÔNG đủ để kết luận về độ chính xác.

Dữ liệu GIẢ: accuracy/F1 của B0, B1 không nói gì về chất lượng model gốc (nó nhìn hình vẽ, không phải lá). Ở B0/B1 chỉ cột JSON hợp lệ là đáng đọc.

| Nhánh | n | JSON hợp lệ (parser thật) | JSON có primary_diagnosis | Accuracy | Macro-F1 | Độ trễ TB (s) |
|---|---|---|---|---|---|---|
| B0 model gốc, prompt dùng chung | 21 | 0% | 0% | 0% | 0.00 | 3.2 |
| B1 model gốc, prompt có danh sách nhãn | 21 | 0% | 0% | 0% | 0.00 | 2.5 |
| FT sau fine-tune (model trong bộ nhớ) | 21 | 100% | 100% | 81% | 0.79 | 1.2 |
| FT nạp lại adapter từ đĩa | 21 | 100% | 100% | 81% | 0.79 | 1.1 |
| FT hỏi bằng prompt đang chạy thật ("cây trồng", L1) | 21 | 100% | 100% | 67% | 0.64 | 1.1 |
| ĐỐI CHỨNG train nhãn xáo trộn | 21 | 100% | 100% | 95% | 0.95 | 1.5 |

## Kiểm chéo
- Đối chứng nhãn xáo trộn: accuracy 95% trên 21 ảnh (đoán mò ~33%). CẢNH BÁO: vẫn cao hơn 60%. Model đoán đúng dù nhãn train vô nghĩa => nghi rò rỉ hoặc lỗi chấm điểm. Không tin điểm FT cho tới khi tìm ra.
- Nạp lại adapter: 21/21 ảnh cùng nhãn, 21/21 ảnh trùng nguyên văn. ĐẠT.
- Lệch prompt L1 (train "cà phê", hỏi "cây trồng"): accuracy đổi -14%, JSON hợp lệ 100%. Trên dữ liệu giả, lệch nhỏ chỉ chứng tỏ bài này dễ, không chứng tỏ L1 vô hại.

## Ảnh ngoài lề (nhiễu, thiếu sáng, mờ nặng, không phải lá)
Bài smoke KHÔNG có nhãn "không dùng được", nên kỳ vọng là model FT gán bệnh cho cả ảnh rác. Mục này đo mức độ đó; nó không phải cổng "từ chối đúng >= 90%" của báo cáo.

| Nhánh | n | Bị gán một nhãn trong danh sách | Confidence TB khi bị gán | Theo loại ảnh (bị gán/tổng) |
|---|---|---|---|---|
| B0 model gốc | 20 | 0% | - | blur 0/5, dark 0/5, noise 0/5, object 0/5 |
| FT sau fine-tune | 20 | 100% | 0.90 | blur 5/5, dark 5/5, noise 5/5, object 5/5 |

## B0 model gốc, prompt dùng chung
- Theo nhãn: healthy_leaf: F1 0.00 (n=7); coffee_leaf_rust: F1 0.00 (n=7); red_spider_mite: F1 0.00 (n=7)
- Ma trận nhầm (hàng = thật): {"red_spider_mite": {"INVALID": 7}, "healthy_leaf": {"INVALID": 7}, "coffee_leaf_rust": {"INVALID": 7}}

## B1 model gốc, prompt có danh sách nhãn
- Theo nhãn: healthy_leaf: F1 0.00 (n=7); coffee_leaf_rust: F1 0.00 (n=7); red_spider_mite: F1 0.00 (n=7)
- Ma trận nhầm (hàng = thật): {"red_spider_mite": {"INVALID": 7}, "healthy_leaf": {"INVALID": 7}, "coffee_leaf_rust": {"INVALID": 7}}

## FT sau fine-tune (model trong bộ nhớ)
- Theo nhãn: healthy_leaf: F1 1.00 (n=7); coffee_leaf_rust: F1 0.78 (n=7); red_spider_mite: F1 0.60 (n=7)
- Ma trận nhầm (hàng = thật): {"red_spider_mite": {"red_spider_mite": 3, "coffee_leaf_rust": 4}, "healthy_leaf": {"healthy_leaf": 7}, "coffee_leaf_rust": {"coffee_leaf_rust": 7}}

## FT nạp lại adapter từ đĩa
- Theo nhãn: healthy_leaf: F1 1.00 (n=7); coffee_leaf_rust: F1 0.78 (n=7); red_spider_mite: F1 0.60 (n=7)
- Ma trận nhầm (hàng = thật): {"red_spider_mite": {"red_spider_mite": 3, "coffee_leaf_rust": 4}, "healthy_leaf": {"healthy_leaf": 7}, "coffee_leaf_rust": {"coffee_leaf_rust": 7}}

## FT hỏi bằng prompt đang chạy thật ("cây trồng", L1)
- Theo nhãn: healthy_leaf: F1 1.00 (n=7); coffee_leaf_rust: F1 0.00 (n=7); red_spider_mite: F1 0.93 (n=7)
- Ma trận nhầm (hàng = thật): {"red_spider_mite": {"red_spider_mite": 7}, "healthy_leaf": {"healthy_leaf": 7}, "coffee_leaf_rust": {"OTHER": 6, "red_spider_mite": 1}}

## ĐỐI CHỨNG train nhãn xáo trộn
- Theo nhãn: healthy_leaf: F1 0.92 (n=7); coffee_leaf_rust: F1 0.93 (n=7); red_spider_mite: F1 1.00 (n=7)
- Ma trận nhầm (hàng = thật): {"red_spider_mite": {"red_spider_mite": 7}, "healthy_leaf": {"healthy_leaf": 6, "coffee_leaf_rust": 1}, "coffee_leaf_rust": {"coffee_leaf_rust": 7}}

## Huấn luyện
- Model: unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit
- Số step: 30, loss train trung bình: 1.759
- Thời gian: 165.6 giây; VRAM đỉnh: 3.08 GB
- Loss trên val (step: loss): 10: 2.0281, 20: 0.3349, 30: 0.0278

## Huấn luyện (đối chứng nhãn xáo trộn)
- Model: unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit
- Số step: 30, loss train trung bình: 1.765
- Thời gian: 174.6 giây; VRAM đỉnh: 3.08 GB
- Loss trên val (step: loss): 10: 2.0283, 20: 0.3519, 30: 0.036

Nhãn: healthy_leaf, coffee_leaf_rust, red_spider_mite. Cách chấm tên bệnh tự do của model gốc: xem normalize_label trong common.py.