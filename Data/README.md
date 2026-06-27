# Vấn đề dữ liệu CHƯA được tiền xử lý giải quyết (Limitations)

> Tài liệu này liệt kê các **hạn chế còn lại** của dữ liệu sau bước tiền xử lý, dùng cho 3 bộ thí nghiệm
> **lựa chọn đặc trưng trên Linear Regression (Lasso / Boruta)**: Obesity, Communities & Crime, Riboflavin.
> EDA & tiền xử lý chi tiết: [EDA_REPORT.md](EDA_REPORT.md) và các notebook trong [data_processing/](data_processing/).

Tiền xử lý hiện tại mới giải quyết **missing · trùng lặp · encoding**. Các vấn đề **bản chất** dưới đây
**vẫn còn** (một số cố ý nhường cho bước lựa chọn đặc trưng) — cần lưu ý khi diễn giải kết quả.

## Tóm tắt 3 dataset

| Dataset | Bài toán | n × p (sau xử lý) | Target | Xử lý chính |
| --- | --- | --- | --- | --- |
| Obesity | Hồi quy ($n \gg p$) | 2.087 × 20 | `Weight` | bỏ trùng + encode theo loại biến |
| Communities & Crime | Hồi quy ($n \approx p$) | 1.994 × 100 | `ViolentCrimesPerPop` | bỏ 5 cột định danh + 22 cột thiếu ~84% + điền median |
| Riboflavin | Hồi quy ($p \gg n$) | 71 × 4.088 | `y` | giữ toàn bộ gen (**không lọc** phương sai) |

## Chung cho cả 3 (giả định của hồi quy tuyến tính)

- **Phi tuyến & tương tác:** không bước nào kiểm tra/biến đổi → nếu quan hệ thật phi tuyến, mô hình underfit và lựa chọn đặc trưng có thể loại nhầm biến hữu ích.
- **Target lệch không transform:** Communities (skew ≈ +1,52), Riboflavin (≈ −0,84) → có thể vi phạm đồng phương sai (homoscedasticity) của phần dư, làm RMSE/$R^2$ thiên lệch.
- **Đa cộng tuyến:** scaling/encoding **không** khử được — chỉ giảm *gián tiếp* qua Lasso/selection, không "giải quyết" ở khâu tiền xử lý.

## Obesity

- ⚠️ **Rò rỉ khái niệm:** `NObeyesdad`/`Height` gần như **định nghĩa** `Weight` (qua BMI $=\text{Weight}/\text{Height}^2$) → $R^2$ cao "ảo", **lu mờ tác dụng** của lựa chọn đặc trưng.
- ⚠️ **Đa cộng tuyến do `MTRANS` one-hot không `drop_first`:** 5 cột dummy cộng lại $=1=$ cột intercept → ma trận thiết kế **suy biến** (dummy trap); hệ số OLS thuần mất ý nghĩa diễn giải (Lasso/regularization thì không sao).
- **Biến phân loại cực mất cân bằng:** `SMOKE` (44 "yes"), `CALC` ("Always" = 1 mẫu) → hệ số các mức hiếm không tin cậy.
- **Dữ liệu một phần là synthetic (SMOTE)** — chưa kiểm chứng so với phân phối thật.

## Communities & Crime

- **49 cặp |corr| > 0,9 vẫn còn nguyên** → OLS "full" có hệ số bất ổn (đúng mục tiêu thí nghiệm, nhưng dữ liệu chưa "sạch" về collinearity).
- **Mất thông tin** do bỏ 22 cột chỉ số cảnh sát (có thể có giá trị dự đoán, không khôi phục được).
- **Target lệch phải mạnh** chưa được log-transform.

## Riboflavin

- **$p \gg n$ là vấn đề gốc, tiền xử lý KHÔNG giải quyết** (nay còn rõ hơn vì **không lọc** gen) → OLS không có nghiệm duy nhất, phụ thuộc hoàn toàn vào regularization.
- **Bất ổn của chính feature selection:** với 4.088 gen / 71 mẫu, tập gen Lasso/Boruta chọn **dao động mạnh theo seed/fold** — khó tái lập; CV nhiều seed chỉ *đo* được độ bất ổn chứ không khử.
- **Đa cộng tuyến giữa gen cùng pathway** chưa xử lý.
- **Test chỉ ~15 mẫu** → mọi chỉ số một-lần-split rất nhiễu (nên báo cáo kèm cross-validation).
