# Báo cáo EDA — 3 Dataset: Obesity · Communities&Crime · Riboflavin

> Báo cáo phân tích khám phá dữ liệu (EDA) chi tiết cho 3 bộ dữ liệu dùng trong thí nghiệm **feature selection**
> (Linear Regression / Lasso / Boruta). Số liệu lấy từ output thật của các notebook tiền xử lý
> ([preproc_obesity.ipynb](preproc_obesity.ipynb), [preproc_communities.ipynb](preproc_communities.ipynb),
> [preproc_riboflavin.ipynb](preproc_riboflavin.ipynb)). Biểu đồ nằm trong [eda_assets/](eda_assets/).

| Dataset | Bài toán | Kích thước (sau xử lý) | Đặc thù |
|---|---|---|---|
| **Obesity** | Phân loại (7 lớp) | 2.087 × 17 → encode 2.087 × **20** | Nhiều biến phân loại → **binary mapping** |
| **Communities & Crime** | Hồi quy | 1.994 × **101** | p vừa, nhiều feature yếu + đa cộng tuyến |
| **Riboflavin** | Hồi quy | 71 × **3.680** | **p ≫ n** (gen biểu hiện), bắt buộc regularization |

---

# 1. Obesity (`ObesityDataSet.csv`)

**Mục tiêu:** dự đoán mức béo phì `NObeyesdad` (7 lớp có thứ tự). Đây là bộ duy nhất có biến phân loại, nên là nơi
áp dụng **binary mapping** thay cho one-hot.

## 1.1 Tổng quan & chất lượng dữ liệu
- **Kích thước gốc:** 2.111 dòng × 17 cột (8 số, 9 phân loại). **0 ô missing**.
- **Trùng lặp:** 24 dòng trùng (do phần dữ liệu tổng hợp SMOTE của bộ gốc) → **loại bỏ** còn **2.087 dòng**.
- **Không** có outlier lỗi: `Age` có vài giá trị cao (tới 61) nhưng là người thật; `Weight` 39–173 kg, `Height` 1,45–1,98 m.

## 1.2 Cân bằng lớp target — *thuận lợi*
![Cân bằng lớp](eda_assets/ob_classbalance.png)

| Lớp | Số mẫu |
|---|---|
| Insufficient_Weight | 267 |
| Normal_Weight | 282 |
| Overweight_Level_I | 276 |
| Overweight_Level_II | 290 |
| Obesity_Type_I | 351 |
| Obesity_Type_II | 297 |
| Obesity_Type_III | 324 |

→ **Khá cân bằng** (267–351 mẫu/lớp), không cần xử lý mất cân bằng lớp.

## 1.3 Biến số — thống kê & độ lệch

| Biến | mean | std | min | max | skew | Ghi chú |
|---|---|---|---|---|---|---|
| Age | 24,35 | 6,37 | 14 | 61 | **+1,51** | lệch phải mạnh, đa số 20–26 tuổi |
| Height | 1,70 | 0,09 | 1,45 | 1,98 | −0,02 | gần chuẩn |
| Weight | 86,86 | 26,19 | 39 | 173 | +0,24 | gần đối xứng |
| FCVC (ăn rau) | 2,42 | 0,53 | 1 | 3 | −0,45 | thang rời rạc 1–3 |
| NCP (số bữa) | 2,70 | 0,76 | 1 | 4 | **−1,14** | tụ ở 3 bữa |
| CH2O (nước) | 2,00 | 0,61 | 1 | 3 | −0,11 | thang 1–3 |
| FAF (vận động) | 1,01 | 0,85 | 0 | 3 | +0,49 | nhiều người ít vận động |
| TUE (thời gian dùng TB) | 0,66 | 0,61 | 0 | 2 | +0,61 | thang 0–2 |

→ Các cột hành vi (`FCVC,NCP,CH2O,FAF,TUE`) bản chất **rời rạc** nhưng lưu dạng số thực → không có outlier, không cần log.

## 1.4 Biến phân loại — cardinality & mất cân bằng

| Biến | Loại | Phân bố |
|---|---|---|
| Gender | nhị phân | Male 1052 / Female 1035 — **cân bằng** |
| family_history_with_overweight | nhị phân | yes 1722 / no 365 |
| FAVC (đồ ăn nhanh) | nhị phân | yes 1844 / no 243 |
| SMOKE | nhị phân | no 2043 / **yes 44** — rất lệch |
| SCC (theo dõi calo) | nhị phân | no 1991 / yes 96 |
| CAEC (ăn vặt) | thứ bậc (4) | Sometimes 1761 / Frequently 236 / Always 53 / no 37 |
| CALC (rượu) | thứ bậc (4) | Sometimes 1380 / no 636 / Frequently 70 / **Always 1** |
| MTRANS | danh nghĩa (5) | Public 1558 / Auto 456 / Walking 55 / Motorbike 11 / Bike 7 |

## 1.5 Quan hệ với target & tương quan
![Weight theo lớp](eda_assets/ob_weight_by_class.png)
![Ảnh hưởng lên target](eda_assets/ob_assoc.png)

**Xếp hạng ảnh hưởng** (η cho biến số, Cramér's V cho phân loại — đều ∈ [0,1]):

| Hạng | Feature | Độ mạnh | Loại |
|---|---|---|---|
| 1 | **Weight** | **0,921** | numeric (η) |
| 2 | Gender | 0,561 | category (V) |
| 3 | family_history_with_overweight | 0,544 | category (V) |
| 4 | FCVC | 0,492 | numeric (η) |
| 5 | Age | 0,424 | numeric (η) |
| 6 | CAEC | 0,340 | category (V) |
| 7 | FAVC | 0,333 | category (V) |
| … | … | … | … |
| 15 | TUE | 0,150 | numeric (η) |
| 16 | SMOKE | 0,124 | category (V) |

> ⚠️ **Cảnh báo leakage khái niệm:** `Weight` (η=0,921) gần như **định nghĩa** nhãn béo phì. Nếu muốn dự đoán từ
> **lối sống**, nên cân nhắc bỏ `Weight`/`Height`. Khi đó các tín hiệu "thật" mạnh nhất là **tiền sử gia đình, giới tính,
> thói quen ăn rau (FCVC) và tuổi**. Yếu nhất: `SMOKE`, `TUE`, `MTRANS`.

![Tương quan biến số](eda_assets/ob_corr.png)
Các biến số tương quan **yếu** với nhau (trừ Weight–Height vừa phải) → ít dư thừa thông tin.

## 1.6 ⭐ Encoding — Binary mapping (thay cho one-hot)

**Vấn đề của one-hot với biến nhị phân:** one-hot một biến 2 mức tạo **2 cột** thoả `A + B = 1` →
**tương quan = −1 (đa cộng tuyến hoàn hảo)**; 1 cột hoàn toàn dư thừa, làm hệ số Linear/Lasso bất ổn.

**Sơ đồ encoding mới:**

| Nhóm | Biến | Cách encode | Số cột |
|---|---|---|---|
| Nhị phân | Gender, family_history, FAVC, SMOKE, SCC | **0/1 (binary mapping)** | 5 (trước: 10) |
| Thứ bậc | CAEC, CALC | ordinal `no=0<Sometimes=1<Frequently=2<Always=3` | 2 (trước: 8) |
| Danh nghĩa | MTRANS (5 mức) | one-hot `drop_first=True` | 4 (trước: 5) |
| Target | NObeyesdad | ordinal 0..6 theo thứ tự béo | 1 |

→ Kết quả: `obesity_encoded.csv` giảm từ **32 → 20 cột**, **toàn số, không còn cặp cột đa cộng tuyến hoàn hảo**.

---

# 2. Communities & Crime (`communities.data`)

**Mục tiêu:** hồi quy `ViolentCrimesPerPop` (tỉ lệ tội phạm bạo lực, chuẩn hoá [0,1]). **Toàn biến số** → không có encoding.

## 2.1 Tổng quan & làm sạch
- **Gốc:** 1.994 dòng × 128 cột (gần như toàn `float64` chuẩn hoá [0,1]; chỉ `communityname` là chuỗi).
- **Bỏ 5 cột định danh:** `state, county, community, communityname, fold` → còn 123 cột.
- Sau làm sạch: **1.994 × 101** (100 feature + target), **0 missing**.

## 2.2 Phân tích missing — *lệch hẳn về nhóm cảnh sát*
![Missing](eda_assets/cm_missing.png)

- **23 cột có missing**, trong đó **22 cột thiếu ~84%** (1.675/1.994 ô) — gần như toàn nhóm chỉ số cảnh sát
  (`PolicPerPop, PolicBudgPerPop, RacialMatchCommPol, LemasSwFT*`, …).
- → Điền cho cột thiếu 84% là **bịa dữ liệu** → **bỏ hẳn 22 cột**.
- `OtherPerCap` chỉ thiếu **1 ô** → điền median. Target **không thiếu**.

## 2.3 Phân phối target & tương quan
![Target & corr](eda_assets/cm_target_corr.png)

- Target **lệch phải mạnh (skew = 1,52)**: mean 0,238, median 0,15 — nhiều cộng đồng tội phạm thấp, đuôi cao thưa.

**Tương quan dương mạnh nhất:**

| Feature | corr |
|---|---|
| PctIlleg (tỉ lệ sinh ngoài giá thú) | +0,738 |
| racepctblack | +0,631 |
| pctWPubAsst (hộ nhận trợ cấp) | +0,575 |
| FemalePctDiv / TotalPctDiv | +0,556 / +0,553 |

**Tương quan âm mạnh nhất:**

| Feature | corr |
|---|---|
| PctKids2Par (trẻ sống với 2 cha mẹ) | −0,738 |
| PctFam2Par | −0,707 |
| racePctWhite | −0,685 |
| PctYoungKids2Par / PctTeen2Par | −0,666 / −0,662 |

→ Tín hiệu mạnh nhất là **cấu trúc gia đình & nhân khẩu**.

## 2.4 Đặc điểm quan trọng cho feature selection
- **18/100 feature có |corr| < 0,1** (yếu, gần nhiễu) → đây là phần Lasso/Boruta có thể loại.
- **49 cặp feature có |corr| > 0,9** (đa cộng tuyến nặng), ví dụ:
  - `PctRecImmig8 ~ PctRecImmig10` = 0,996
  - `OwnOccLowQuart ~ OwnOccMedVal` = 0,994
  - `population ~ numbUrban` = 0,993
- → Linear Regression "full" sẽ có hệ số bất ổn → **bộ này rất hợp** để chứng minh giá trị của regularization / selection.
  Feature đã chuẩn hoá [0,1] nhưng vẫn nên `StandardScaler` cho Lasso để phạt công bằng.

---

# 3. Riboflavin (`riboflavin_dataset.csv`)

**Mục tiêu:** hồi quy `y` (log sản lượng riboflavin) từ **biểu hiện gen**. Bài toán kinh điển **p ≫ n**.

## 3.1 Tổng quan & chất lượng
- **Kích thước:** 71 mẫu × 4.089 cột (1 target + **4.088 gen**), **toàn `float64`**.
- **Tỉ lệ p/n ≈ 57,6** — số feature gấp ~58 lần số mẫu.
- **0 missing, 0 dòng trùng** → dữ liệu microarray sạch sẵn.

## 3.2 Phân phối target `y`
![y](eda_assets/rb_y.png)

- Khoảng **[−9,97; −5,67]**, mean −7,16, std 0,92, **lệch trái nhẹ (skew = −0,84)**, không outlier cực đoan → giữ nguyên target.

## 3.3 Phương sai gen & tương quan với target
![variance](eda_assets/rb_var.png)
![top genes](eda_assets/rb_topgenes.png)

- **Phương sai gen:** min 0,0099 — median 0,132 — max 3,40. Không có gen phương sai = 0, nhưng nhóm đuôi trái gần như hằng số.
- **Tương quan với `y`:** gen mạnh nhất chỉ đạt **|corr| ≈ 0,65** (`x.XHLA_at, x.XHLB_at, x.YXLD_at, x.YCKE_at`, …).
- **43% gen (1.757/4.088) có |corr| < 0,1** → tín hiệu chỉ nằm ở **một nhúm nhỏ gen**; phần lớn là nhiễu.

## 3.4 Tiền xử lý & hệ quả mô hình
- **Lọc 10% gen phương sai thấp nhất** (gần hằng số) → bỏ **409 gen**, còn **71 × 3.680** (`riboflavin_processed.csv`).
- Vẫn **p ≫ n** (p/n ≈ 52) → thông điệp không đổi:
  - **OLS full sẽ overfit hoàn hảo** (train R² = 1,0 do nội suy 56 mẫu bằng hàng nghìn feature).
  - **Bắt buộc feature selection / regularization** (Lasso, Boruta).
  - **Luôn báo cáo cross-validation:** n=71, một lần split (test 15 mẫu) dao động rất mạnh theo seed.

---

# 4. Tổng kết & khuyến nghị

| Tiêu chí | Obesity | Communities | Riboflavin |
|---|---|---|---|
| Bài toán | Phân loại 7 lớp | Hồi quy | Hồi quy |
| n × p (sau xử lý) | 2.087 × 19 feat | 1.994 × 100 feat | 71 × 3.679 feat |
| Missing | 0 | điền/bỏ (22 cột) | 0 |
| Vấn đề chính | leakage Weight; encode biến phân loại | feature yếu + đa cộng tuyến | **p ≫ n**, overfit |
| Xử lý đặc thù | bỏ trùng + **binary mapping** | bỏ cột thiếu 84% + median | lọc gen var thấp |
| Phù hợp để test | Boruta/Lasso trên dữ liệu hỗn hợp | Lasso/RFE vs LR full | Lasso/Boruta + CV bắt buộc |

**Khuyến nghị chung:**
1. **Obesity** — dùng `obesity_encoded.csv` (20 cột). Cân nhắc bỏ `Weight`/`Height` nếu mục tiêu là dự đoán từ lối sống.
2. **Communities** — `StandardScaler` + Lasso để xử lý 18 feature nhiễu và 49 cặp đa cộng tuyến.
3. **Riboflavin** — ưu tiên **dự đoán bằng chính Lasso (regularized)**, tránh refit OLS trần trên feature đã chọn; báo cáo CV nhiều seed.
