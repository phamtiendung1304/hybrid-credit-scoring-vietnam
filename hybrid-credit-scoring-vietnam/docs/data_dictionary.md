# Data Dictionary
## Hybrid Credit Scoring — Vietnam

This document provides complete definitions, data types, valid ranges, and
source descriptions for all features used in the Hybrid Credit Scoring model.

---

## 1. Target Variable

| Column | Type | Values | Definition |
|---|---|---|---|
| `default_flag` | Binary integer | {0, 1} | 1 if the borrower was ≥ 90 days past due at any point within the 36-month observation window (NPL definition per SBV Circular 11/2021); 0 otherwise. |
| `application_id` | String | UUID | Unique application identifier. Not used in modeling. Retained for traceability only. |

---

## 2. CIC Bureau Features

Features derived from the Credit Information Center of Vietnam (CIC) bureau records, available at the time of credit application.

| Column | Type | Valid Range | Missing Rate | Definition |
|---|---|---|---|---|
| `cic_payment_history_score` | Float | [0.0, 1.0] | ~15% (thin-file) | Normalized score reflecting payment behavior on existing credit obligations. Higher = better payment history. Computed from CIC bureau records; 0 = no record. |
| `cic_outstanding_balance_ratio` | Float | [0.0, 1.0] | ~15% | Ratio of total outstanding balance to total approved credit limit across all active products. 1.0 = fully utilized. |
| `cic_num_inquiries_6m` | Integer | [0, 20] | ~15% | Number of hard credit inquiries recorded by CIC in the past 6 months. High inquiry count signals credit-seeking behavior and potential financial stress. |
| `cic_account_age_months` | Integer | [0, 240] | ~15% | Age of the borrower's oldest credit account in months. Proxy for depth of credit history. |
| `cic_num_delinquencies` | Integer | [0, 50] | ~15% | Cumulative count of historical delinquency events (any product) recorded in CIC. |
| `cic_credit_utilization` | Float | [0.0, 1.0] | ~15% | Average credit utilization ratio over the past 12 months. Calculated as average balance / credit limit. |
| `cic_num_active_accounts` | Integer | [0, 20] | ~15% | Number of currently active credit products (loans, credit cards, overdrafts) registered in CIC. |

### CIC Missingness Flags (engineered)

For each CIC bureau feature above, a binary missingness indicator is automatically engineered by the `DataCleaner`:

| Column | Type | Definition |
|---|---|---|
| `cic_payment_history_score_missing` | Binary {0,1} | 1 if original CIC feature was absent (thin-file applicant) |
| `cic_outstanding_balance_ratio_missing` | Binary {0,1} | — |
| *(and so on for each bureau feature)* | | |

---

## 3. Digital Behavioral Features

Features derived from mobile banking transaction logs and digital payment records.
All behavioral features are computed at application time from a backward-looking window.

| Column | Type | Valid Range | Missing Rate | Definition |
|---|---|---|---|---|
| `txn_frequency_30d` | Integer | [0, 500] | <5% | Number of mobile banking / digital payment transactions in the 30-day window prior to application. |
| `txn_frequency_90d` | Integer | [0, 1500] | <5% | Transaction count in the 90-day window. Higher values indicate more active financial behavior. |
| `avg_txn_amount_30d` | Float | [0, 50,000,000 VND] | <5% | Average single transaction amount (VND) in the 30-day window. Proxy for income level and spending scale. |
| `merchant_category_diversity` | Integer | [1, 50] | <5% | Number of distinct merchant categories (MCC codes) in which the applicant transacted in the 90-day window. High diversity indicates broader financial activity. |
| `digital_payment_regularity` | Float | [0.0, 1.0] | <5% | Regularity score for bill payments, loan installments, and subscription charges. Computed as 1 − (std of payment intervals / mean payment interval). Higher = more consistent. |
| `mobile_app_engagement_score` | Float | [0.0, 10.0] | <5% | Composite score based on mobile banking app login frequency, session duration, and feature usage breadth. Higher = more engaged digital customer. |
| `spending_volatility_90d` | Float | [0, ∞] | <5% | Standard deviation of weekly total spending (VND) over the 90-day window. High volatility may indicate irregular income or impulsive spending. |
| `salary_credit_regularity` | Binary {0, 1} | — | <5% | 1 if regular salary/payroll credits are detected in the account (at least 3 monthly credits of similar amount in the 90-day window); 0 otherwise. Strong indicator of formal employment and income stability. |

---

## 4. Demographic Features

Features collected from the loan application form at origination.

| Column | Type | Valid Range | Missing Rate | Definition |
|---|---|---|---|---|
| `age_at_application` | Integer | [18, 65] | <1% | Applicant's age in years at time of application. |
| `employment_type` | Categorical | See below | <3% | Employment status category. Encoded as integer in pipeline. |
| `declared_income_bracket` | Ordinal integer | [1, 5] | <5% | Self-reported income bracket. 1 = lowest (<5M VND/month), 5 = highest (>30M VND/month). Not independently verified. |
| `education_level` | Ordinal integer | [1, 5] | <3% | Highest completed education level. 1 = primary, 2 = secondary, 3 = vocational/college, 4 = undergraduate, 5 = postgraduate. |

### Employment Type Encoding

| Label Encoding | Category | Description |
|---|---|---|
| 0 | `formal` | Full-time employee with formal labor contract |
| 1 | `informal` | Part-time, gig economy, or unregistered employment |
| 2 | `self_employed` | Business owner or freelancer |
| 3 | `student` | Currently enrolled full-time student |

---

## 5. Engineered Features

Features computed during the Feature Engineering phase (`src/features/`).

| Column | Source Module | Definition |
|---|---|---|
| `woe_*` | `traditional_features.py` | Weight-of-Evidence transformation of bureau features for Logistic Regression. |
| `txn_freq_ratio_30_90d` | `behavioral_features.py` | `txn_frequency_30d / (txn_frequency_90d / 3 + 1)` — measures recency of transaction acceleration. |
| `digital_activity_composite` | `behavioral_features.py` | Weighted composite of regularity, engagement, and salary indicator scores. |
| `cic_missing_count` | `cleaner.py` | Count of missing CIC bureau features per observation (0–7). High values indicate thin-file applicants. |

---

## 6. Excluded Variables (Not Used in Final Model)

| Column | Reason for Exclusion |
|---|---|
| `application_id` | Identifier — no predictive power |
| `application_date` | Time variable — introduces temporal leakage if used naively |
| `loan_amount_requested` | Endogenous — determined jointly with credit decision |
| `phone_number` | PII — excluded for privacy compliance |
| `home_address_region` | Geographic variable — flagged for potential proxy discrimination; requires fairness audit before inclusion |

---

## 7. Data Quality Notes

- **CIC missing data (thin-file):** Approximately 15% of observations have missing CIC features due to thin-file status. This is treated as informative missingness, not random. Missingness indicator flags are engineered for all CIC features.
- **Behavioral data gap:** Applicants who do not hold a digital banking relationship with the issuing bank will have missing behavioral features. These applicants are scored using a bureau-only fallback model.
- **Declared income reliability:** `declared_income_bracket` is self-reported and unverified. Income proxies from transaction data (`avg_txn_amount_30d`) may be more reliable.
- **Temporal consistency:** All behavioral features are computed using a rolling window anchored at `application_date`. No look-ahead data is included.

---

*Data Dictionary Version 1.0 | 2024 | NEU Hybrid Credit Scoring Research Project*
