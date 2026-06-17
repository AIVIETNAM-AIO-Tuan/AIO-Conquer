# EDA Report — Melbourne Housing (`melb_data.csv`)

> 🇬🇧 English version: [README.en.md](README.en.md)
> 📓 Notebook: [preproc_data.ipynb](preproc_data.ipynb)

Báo cáo khảo sát & tiền xử lý dữ liệu nhà Melbourne, tổng hợp từ notebook theo luồng **clean-first**.
Mọi số liệu trong báo cáo lấy trực tiếp từ output thật khi chạy notebook.

---

## 1. Tổng quan dataset & dữ liệu thiếu

- **Kích thước:** 13.580 dòng × 21 cột (13 cột số, 8 cột `object`).
- **Target:** `Price` (đô, cỡ hàng triệu), lệch phải mạnh.
- **Cột thiếu dữ liệu:**

| Cột | Non-null | Thiếu | Tỷ lệ thiếu |
| --- | --- | --- | --- |
| `BuildingArea` | 7.130 | 6.450 | ~47% |
| `YearBuilt` | 8.205 | 5.375 | ~40% |
| `CouncilArea` | 12.211 | 1.369 | ~10% |
| `Car` | 13.518 | 62 | ~0,5% |

`BuildingArea` và `YearBuilt` thiếu rất nặng → không thể bỏ dòng, phải điền cẩn thận.
`Postcode` tuy kiểu số nhưng thực chất là **mã định danh**, không phải biến liên tục.

## 2. Phân phối đơn biến

- **Biến số lệch phải mạnh:** `Price`, `Landsize`, `BuildingArea` dồn về trái với đuôi dài do vài giá trị cực lớn.
- **Biến đếm rời rạc:** `Rooms`, `Bedroom2`, `Bathroom`, `Car` tập trung ở giá trị nhỏ.
- **Biến phân loại mất cân bằng:** `Type` chủ yếu 'h'; `Regionname`/`Method` dồn vào vài nhóm; `Suburb`/`SellerG`/`Address` cardinality rất cao.
- **Cardinality (số giá trị duy nhất):**

| Cột | n_unique | Ghi chú |
| --- | --- | --- |
| `Address` | 13.378 | ≈ số dòng → nên **bỏ** |
| `Suburb` | 314 | cao → frequency/target encoding |
| `SellerG` | 268 | cao → frequency/target encoding |
| `Date` | 58 | nên tách year/month |
| `CouncilArea` | 33 | (+1.369 missing) |
| `Regionname` | 8 | → one-hot |
| `Method` | 5 | → one-hot |
| `Type` | 3 | → one-hot |

## 3. Phát hiện & xử lý outlier

**Phát hiện** (boxplot 1.5·IQR + scatter vs Price):

- **Lỗi dữ liệu rõ ràng:** `Landsize` ~430.000 m², `BuildingArea` ~44.500 m², `YearBuilt` ~1196, `Bedroom2` = 20.
- **Giá trị thật hiếm:** `Price` cao (tới 9 triệu) — nhà đắt thật, **không xoá**.
- `Postcode vs Price` cho các vạch dọc rời rạc → khẳng định là mã định danh.
- `Landsize/BuildingArea vs Price` dạng chữ L: outlier cực lớn kéo giãn trục, che mất quan hệ thật.

**Xử lý:**

- Loại **2.075** giá trị `Landsize` (≤0 hoặc > P99 = 2.960); **89** giá trị `BuildingArea` (≤0 hoặc > P99 = 466); **1** giá trị `YearBuilt` phi lý (1196).
- **Log transform** (`log1p`) cho `Landsize`/`BuildingArea` → phân phối chuyển từ lệch phải cực mạnh sang **gần đối xứng**.
- **Không** log `YearBuilt` (là mốc năm, log vô nghĩa); nếu cần nên đổi sang tuổi nhà `2018 − YearBuilt`.

## 4. Xử lý missing — Random sampling vs KNN

So sánh hai cách điền cho biến số, dùng heatmap tương quan làm bằng chứng:

- **Random sampling** (bốc ngẫu nhiên từ giá trị non-null): giữ đúng **phân phối biên** từng cột, nhưng điền *độc lập* với các cột khác → **phá vỡ tương quan đa biến**.
- **KNN (`weights='distance'`, có scale trước):** điền dựa trên các dòng tương tự → **giữ tương quan đa biến**.

**Bằng chứng — tương quan của cột thiếu nhiều (giá trị thật vs sau điền):**

| Cặp biến | Thật | Random | KNN |
| --- | --- | --- | --- |
| BuildingArea – Price | +0,091 | **+0,046** | +0,105 |
| YearBuilt – Price | −0,324 | **−0,206** | −0,336 |
| BuildingArea – Rooms | +0,124 | **+0,048** | +0,136 |
| BuildingArea – Bathroom | +0,112 | **+0,049** | +0,122 |

→ Random kéo tương quan **về ~0** (làm loãng tín hiệu); KNN giữ **bám sát bản thật**.
Các cột không bị điền (Rooms, Bathroom…) giữ nguyên tương quan ở cả ba cách.

> **Kết luận:** dùng **KNN** cho dữ liệu mang đi mô hình (bảo toàn pattern/coefficient).
> Random sampling chỉ dùng cho **trực quan hoá phân phối đơn biến**.

## 5. Quan hệ với target `Price`

Đo bằng |Pearson| cho biến số và **correlation ratio η** cho biến phân loại:

- **Biến số mạnh nhất:** `Rooms` 0,497, `Bedroom2` 0,476, `Bathroom` 0,467 — nhưng nhóm này **đa cộng tuyến** (Rooms–Bedroom2 = 0,94).
- **Phân loại đáng tin (cardinality thấp):** `Type` η=0,415, `CouncilArea` η=0,451, `Regionname` η=0,358.
- ⚠️ **Cảnh báo cardinality cao gây η ảo:** `Address` η=0,997 (n_unique=13.378), `Suburb` η=0,560, `SellerG` η=0,478 — bị **thổi phồng** do mỗi nhóm quá ít mẫu (overfit), không phải tín hiệu thật.
- `Landsize`/`BuildingArea` có |Pearson| thấp trên `df_knn` raw (0,038 / 0,105) vì **outlier che lấp**. **Sau khi loại cực trị + log** (mục 10b trong notebook): **`BuildingArea_log` lên 0,496** (gần ngang `Rooms`, top-3) và `Landsize_log` lên 0,192 → diện tích xây dựng thực ra là yếu tố **rất mạnh** quyết định giá.

## 6. Vấn đề chất lượng dữ liệu phát hiện được

- Giá trị phi lý: `YearBuilt` = 1196; `Landsize`/`BuildingArea` cực lớn (lỗi đơn vị/nhập liệu); `Bedroom2` = 20; có `Landsize`/`BuildingArea` = 0.
- `Postcode` là mã định danh, không nên dùng như biến số liên tục.
- **Đa cộng tuyến** trong nhóm `Rooms`/`Bedroom2`/`Bathroom`.
- `Address` gần như duy nhất → không có giá trị mô hình hoá trực tiếp.

## 7. Kế hoạch encoding

- **Cardinality cao** (`Suburb`, `SellerG`, `CouncilArea`): **frequency encoding** (làm trước, an toàn, không leakage).
- **Cardinality thấp** (`Type`, `Method`, `Regionname`): **one-hot**.
- *Ghi chú:* cân nhắc **target encoding** khi làm model — bắt buộc kèm **smoothing + out-of-fold** để chống leakage.
- `Address` → bỏ; `Date` → parse year/month; `Postcode` → xử lý như category.

## 8. Bước tiếp theo

1. **Log-transform target `Price`**.
2. Xử lý `Postcode`/`Date`/`Address`; cân nhắc tạo `Age = 2018 − YearBuilt`.
3. Thêm cờ `is_missing` cho `BuildingArea`/`YearBuilt`.
4. Đóng gói tiền xử lý vào **`Pipeline` + split train/test** (impute/encode/scale fit **chỉ trên train**) để chống leakage.

---

## Phụ lục — Layout thư mục dữ liệu

| Subfolder | Purpose |
| --- | --- |
| `raw/` | Original, immutable source data (never edit in place) |
| `processed/` | Cleaned and transformed data ready for modeling |
| `interim/` | Intermediate files produced during processing |
| `external/` | Third-party or reference datasets |

- Giữ file lớn/nhạy cảm ngoài version control; dùng công cụ data-versioning hoặc lưu trữ ngoài.
- Ghi rõ nguồn, license và ngày cho mỗi dataset.
