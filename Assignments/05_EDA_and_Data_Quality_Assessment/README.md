# Credit Score Dataset: EDA & Data Quality Assessment

**Author:** Abubakar Iqbal  
**Program:** NAVTTC / NIAI (NETSOL Technologies)  
**Dataset:** `CreditScore.csv` (150,000 records, 13 variables)  
**Deliverable:** [`Credit_score_EDA.ipynb`](Credit_score_EDA.ipynb)

---

## 1. Project Overview
This project performs an in-depth exploratory data analysis and data quality audit on a credit bureau delinquency dataset of 150,000 borrowers. The primary target is `SeriousDlqin2yrs` (binary: 0 = non-delinquent, 1 = serious 90+ days delinquency).

---

## 2. Key Findings & Data Quality Audit
- **Severe Class Imbalance:** Only 6.68% (10,026 borrowers) experienced serious delinquency within 2 years, requiring class-weighting or resampling (SMOTE) before machine learning modeling.
- **Missing Value Patterns:** `MonthlyIncome` had 29,731 missing values (19.82%) and `NumberOfDependents` had 3,924 missing values (2.62%). Median imputation by age cohort was applied to avoid distortion.
- **Heavy Outliers:** `RevolvingUtilizationOfUnsecuredLines` contained extreme outliers (> 1.0 and up to 50,708), and `DebtRatio` exceeded thousands due to asset-poor or zero-income borrowers. Values were robustly Winsorized at the 99th percentile.
- **Past-Due Clustering:** The features `NumberOfTimes90DaysLate`, `NumberOfTime60-89DaysPastDueNotWorse`, and `NumberOfTime30-59DaysPastDueNotWorse` showed strong correlation with default probability, with severe delinquent spikes when past-due count $\ge 2$.

---

## 3. Pre-Modeling Checklist for ML
1. Stratified train/test splitting to preserve the 6.68% default ratio.
2. Standard/Robust scaling for continuous numerical features.
3. PR-AUC and ROC-AUC evaluation rather than raw accuracy due to class imbalance.
