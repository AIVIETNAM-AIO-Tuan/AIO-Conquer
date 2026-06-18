# EDA Report — Melbourne Housing (`melb_data.csv`)

> 🇻🇳 Bản tiếng Việt: [README.md](README.md)
> 📓 Notebook: [preproc_data.ipynb](preproc_data.ipynb)

Exploratory data analysis & preprocessing report for the Melbourne housing dataset, summarized from the
notebook following a **clean-first** flow. Every figure in this report is taken directly from the actual
notebook run output.

---

## 1. Dataset overview & missing data

- **Shape:** 13,580 rows × 21 columns (13 numeric, 8 `object`).
- **Target:** `Price` (AUD, in the millions), strongly right-skewed.
- **Columns with missing values:**

| Column | Non-null | Missing | Missing % |
| --- | --- | --- | --- |
| `BuildingArea` | 7,130 | 6,450 | ~47% |
| `YearBuilt` | 8,205 | 5,375 | ~40% |
| `CouncilArea` | 12,211 | 1,369 | ~10% |
| `Car` | 13,518 | 62 | ~0.5% |

`BuildingArea` and `YearBuilt` are missing heavily → dropping rows is not an option; imputation must be careful.
`Postcode` is numeric in dtype but is really an **identifier**, not a continuous variable.

## 2. Univariate distributions

- **Strongly right-skewed numerics:** `Price`, `Landsize`, `BuildingArea` pile up on the left with a long tail driven by a few extreme values.
- **Discrete counts:** `Rooms`, `Bedroom2`, `Bathroom`, `Car` concentrate on small values.
- **Imbalanced categoricals:** `Type` mostly 'h'; `Regionname`/`Method` dominated by a few groups; `Suburb`/`SellerG`/`Address` very high cardinality.
- **Cardinality (unique values):**

| Column | n_unique | Note |
| --- | --- | --- |
| `Address` | 13,378 | ≈ row count → **drop** |
| `Suburb` | 314 | high → frequency/target encoding |
| `SellerG` | 268 | high → frequency/target encoding |
| `Date` | 58 | split into year/month |
| `CouncilArea` | 33 | (+1,369 missing) |
| `Regionname` | 8 | → one-hot |
| `Method` | 5 | → one-hot |
| `Type` | 3 | → one-hot |

## 3. Outlier detection & treatment

**Detection** (1.5·IQR boxplots + scatter vs Price):

- **Clear data errors:** `Landsize` ~430,000 m², `BuildingArea` ~44,500 m², `YearBuilt` ~1196, `Bedroom2` = 20.
- **Genuine rare values:** high `Price` (up to 9M) — real expensive homes, **not removed**.
- `Postcode vs Price` shows discrete vertical bands → confirms it is an identifier.
- `Landsize/BuildingArea vs Price` are L-shaped: extreme outliers stretch the axis and hide the real relationship.

**Treatment:**

- Removed **2,075** `Landsize` values (≤0 or > P99 = 2,960); **89** `BuildingArea` values (≤0 or > P99 = 466); **1** implausible `YearBuilt` (1196).
- **Log transform** (`log1p`) for `Landsize`/`BuildingArea` → distribution turns from heavily right-skewed to **near-symmetric**.
- `YearBuilt` is **not** logged (it is a year; log is meaningless); consider house age `2018 − YearBuilt` instead.

## 4. Missing-value handling — Random sampling vs KNN

Two numeric imputation strategies compared, using correlation heatmaps as evidence:

- **Random sampling** (draw from observed non-null values): preserves each column's **marginal distribution**, but draws are *independent* of other columns → **destroys multivariate correlation**.
- **KNN (`weights='distance'`, scaled first):** imputes from the most similar rows → **preserves multivariate correlation**.

**Evidence — correlations of heavily-missing columns (true vs imputed):**

| Pair | True | Random | KNN |
| --- | --- | --- | --- |
| BuildingArea – Price | +0.091 | **+0.046** | +0.105 |
| YearBuilt – Price | −0.324 | **−0.206** | −0.336 |
| BuildingArea – Rooms | +0.124 | **+0.048** | +0.136 |
| BuildingArea – Bathroom | +0.112 | **+0.049** | +0.122 |

→ Random pulls correlations **toward ~0** (signal dilution); KNN keeps them **close to the true values**.
Non-imputed columns (Rooms, Bathroom…) keep identical correlations across all three.

> **Conclusion:** use **KNN** for the modeling dataset (preserves patterns/coefficients).
> Random sampling is only for **univariate distribution visualization**.

## 5. Relationship with the target `Price`

Measured via |Pearson| for numerics and **correlation ratio η** for categoricals:

- **Strongest numerics:** `Rooms` 0.497, `Bedroom2` 0.476, `Bathroom` 0.467 — but this group is **collinear** (Rooms–Bedroom2 = 0.94).
- **Reliable categoricals (low cardinality):** `Type` η=0.415, `CouncilArea` η=0.451, `Regionname` η=0.358.
- ⚠️ **High-cardinality inflation of η:** `Address` η=0.997 (n_unique=13,378), `Suburb` η=0.560, `SellerG` η=0.478 — **inflated** because each group has too few samples (overfit), not a true signal.
- `Landsize`/`BuildingArea` show low |Pearson| on raw `df_knn` (0.038 / 0.105) because **outliers mask them**. **After extreme removal + log** (section 10b in the notebook): **`BuildingArea_log` rises to 0.496** (near `Rooms`, top-3) and `Landsize_log` to 0.192 → building area is in fact a **very strong** driver of price.

## 6. Data-quality issues found

- Implausible values: `YearBuilt` = 1196; extreme `Landsize`/`BuildingArea` (unit/entry errors); `Bedroom2` = 20; some `Landsize`/`BuildingArea` = 0.
- `Postcode` is an identifier, not a continuous variable.
- **Multicollinearity** among `Rooms`/`Bedroom2`/`Bathroom`.
- `Address` is near-unique → no direct modeling value.

## 7. Encoding plan

- **High cardinality** (`Suburb`, `SellerG`, `CouncilArea`): **frequency encoding** (done first; safe, no leakage).
- **Low cardinality** (`Type`, `Method`, `Regionname`): **one-hot**.
- *Note:* consider **target encoding** at modeling time — must include **smoothing + out-of-fold** to avoid leakage.
- `Address` → **dropped** in the processed set (see section 9); `Date` → parse year/month; `Postcode` → treat as category.

## 8. Next steps

1. **Log-transform the target `Price`**.
2. Handle `Postcode` (as category) and `Date` (parse year/month). *(`Age` and dropping `Address` already applied — see section 9.)*
3. Add `is_missing` flags for `BuildingArea`/`YearBuilt`.
4. Wrap preprocessing in a **`Pipeline` + train/test split** (impute/encode/scale fit **on train only**) to prevent leakage.

## 9. Processed dataset — `dataset/melb_data_processed.csv`

The notebook exports a clean, model-ready dataset (section 11 in the notebook):

- **Shape:** 13,580 rows × **20 columns**, **0 missing**.
- **Steps applied:**
  - Extreme removal + `log1p` for Landsize/BuildingArea (→ `Landsize_log`, `BuildingArea_log`; raw dropped).
  - **KNN imputation** (`weights='distance'`, scaled first) for all remaining numeric missing.
  - `CouncilArea` missing → `'Unknown'`.
  - **`YearBuilt` → `Age = 2018 − YearBuilt`** (year replaced by house age).
  - **Dropped `Address`** (near-unique cardinality, no modeling value).
- **Not done yet (left to the training Pipeline):** categorical encoding, scaling/normalization, target log-transform.
- ⚠️ **Leakage:** this set is imputed on the full data (including `Price`). For a fully clean setup, move impute/encode/scale into a `Pipeline` fit on train only.

---

## Appendix — Data folder layout

| Subfolder | Purpose |
| --- | --- |
| `raw/` | Original, immutable source data (never edit in place) |
| `processed/` | Cleaned and transformed data ready for modeling |
| `interim/` | Intermediate files produced during processing |
| `external/` | Third-party or reference datasets |

- Keep large or sensitive files out of version control; track them with a data versioning tool or external storage.
- Document the source, license, and date for each dataset.
