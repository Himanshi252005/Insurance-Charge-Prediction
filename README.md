# Insurance Charges Prediction

An end-to-end regression pipeline predicting individual medical insurance charges from demographic and lifestyle features — covering cleaning, encoding, domain-driven feature creation, scaling, and model evaluation.

**Final model: Linear Regression — Adjusted R² = 0.796** on a held-out 20% test set.

---

## Problem

Insurance pricing depends on which personal attributes genuinely drive cost. The goal here was not just to maximise accuracy, but to build a pipeline where every feature transformation is deliberate and every retained variable is justified.

---

## Dataset

1,338 policyholder records (1,337 after de-duplication), 7 columns:

| Feature | Type | Description |
|---|---|---|
| `age` | numeric | Policyholder age (18–64) |
| `sex` | categorical | Gender |
| `bmi` | numeric | Body mass index (15.96–53.13) |
| `children` | numeric | Number of dependents (0–5) |
| `smoker` | categorical | Smoking status |
| `region` | categorical | US region (4 levels) |
| `charges` | numeric | **Target** — annual medical cost billed |

Mean charge: $13,270 · Std: $12,110 — a strongly right-skewed target, confirmed in the distribution plots.

---

## Pipeline

**1. Exploratory analysis**
Distribution plots (histogram + KDE) for all numeric features, count plot for dependents, null-value audit (dataset is complete — zero nulls).

**2. Cleaning**
Removed 1 exact duplicate row (1,338 → 1,337).

**3. Encoding**
- Binary label encoding for `sex` → `is_female` and `smoker` → `is_smoker`, with explicit renaming so the column name states what `1` means
- One-hot encoding for `region` with `drop_first=True` to avoid the dummy variable trap

**4. Feature engineering**
Created a `bmi_category` feature from clinical thresholds rather than arbitrary bins — underweight (<18.5), normal (18.5–24.9), overweight (25–29.9), obese (30+) — then one-hot encoded it. This lets the model capture the non-linear jump in cost at the obesity threshold that raw BMI alone would smooth over.

**5. Scaling**
`StandardScaler` applied to `age`, `bmi`, and `children` so no single feature dominates by magnitude.

**6. Modelling**
80/20 train-test split (`random_state=42`), Linear Regression baseline, evaluated with **Adjusted R²** rather than plain R² — the adjusted metric penalises the added dummy columns and gives an honest read on whether the engineered features earned their place.

---

## Result

| Metric | Value |
|---|---|
| Adjusted R² (test) | **0.7963** |
| Model | Linear Regression (OLS) |
| Test size | 268 records |

The model explains roughly 80% of variance in insurance charges. Smoking status is the dominant cost driver — the separation is visible in the exploratory plots well before modelling.

---

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn`

---

## Run Locally

```bash
git clone https://github.com/Himanshi252005/Insurance-Charge-Prediction.git
cd Insurance-Charge-Prediction
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook insurance.ipynb
```

---

## Next Steps

- Compare against Ridge, Random Forest and Gradient Boosting to test whether the relationship is genuinely linear
- Add RMSE and MAE alongside R² for an error figure in dollars
- Formally quantify the smoker effect with a hypothesis test rather than relying on visual separation

---

**Author:** Himanshi Rathore · [LinkedIn](https://www.linkedin.com/in/himanshi-rathore-hr2520) · [GitHub](https://github.com/Himanshi252005)
