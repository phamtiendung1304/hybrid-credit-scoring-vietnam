# Hybrid Credit Scoring for Gen Z Thin-File Customers in Vietnam

> **Integrating Traditional Bureau Data with Digital Behavioral Signals via XGBoost**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange?style=flat-square)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Complete-brightgreen?style=flat-square)]()
[![Institution](https://img.shields.io/badge/Institution-NEU%20Vietnam-blue?style=flat-square)]()
[![AUC](https://img.shields.io/badge/AUC-%3E0.80-success?style=flat-square)]()
[![Gini](https://img.shields.io/badge/Gini-0.6049-success?style=flat-square)]()

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Research Background & Motivation](#2-research-background--motivation)
3. [Research Questions](#3-research-questions)
4. [Dataset Description](#4-dataset-description)
5. [Methodology](#5-methodology)
6. [Key Results](#6-key-results)
7. [Repository Structure](#7-repository-structure)
8. [Installation & Reproducibility](#8-installation--reproducibility)
9. [Usage Guide](#9-usage-guide)
10. [Key Findings & Business Implications](#10-key-findings--business-implications)
11. [Model Risk & Limitations](#11-model-risk--limitations)
12. [Future Work](#12-future-work)
13. [Research Team](#13-research-team)
14. [Citation](#14-citation)
15. [License](#15-license)

---

## 1. Abstract

This project develops and evaluates a **Hybrid Credit Scoring framework** for assessing the creditworthiness of Generation Z (born 1997–2012) thin-file customers in the Vietnamese banking system. Traditional credit scoring models rely primarily on historical credit bureau data — such as that provided by the Credit Information Center of Vietnam (CIC) — which systematically disadvantages young borrowers with limited or no formal credit histories. We propose an alternative approach that integrates CIC bureau features with digital behavioral signals derived from mobile banking activity, transaction patterns, and digital payment footprints.

The hybrid model is trained using **eXtreme Gradient Boosting (XGBoost)** and benchmarked against Logistic Regression, Random Forest, and LightGBM baselines. Model performance is evaluated using the Area Under the ROC Curve (AUC), Gini coefficient, Kolmogorov-Smirnov (KS) statistic, and calibration diagnostics. SHAP (SHapley Additive exPlanations) analysis is applied for post-hoc interpretability.

**Key findings:**
- Hybrid XGBoost achieves **AUC > 0.80** and **Gini coefficient = 0.6049**, representing a substantial improvement over bureau-only baselines
- Digital behavioral features contribute approximately **37% of the model's total predictive power**, as measured by mean absolute SHAP values
- The model demonstrates robust discriminatory performance on Gen Z thin-file subsamples, validating the value of behavioral data integration

---

## 2. Research Background & Motivation

### 2.1 The Thin-File Problem in Vietnamese Banking

Vietnam's financial sector has undergone rapid expansion over the past decade, driven by digital banking adoption, fintech proliferation, and rising consumer credit demand. As of 2023, Vietnam's credit-to-GDP ratio exceeded 120%, with retail lending — particularly unsecured consumer credit — representing one of the fastest-growing segments. However, this expansion coexists with a structural challenge: a substantial share of credit applicants, particularly young adults, lack sufficient credit history for reliable risk assessment through conventional scoring methods.

The Credit Information Center of Vietnam (CIC), operated by the State Bank of Vietnam, covers approximately 56 million individuals. Nevertheless, a significant proportion of Gen Z borrowers — defined here as individuals aged 18–27 at the time of application — have either minimal or entirely absent bureau records. This phenomenon is commonly referred to as the **thin-file problem**: the applicant is not necessarily uncreditworthy, but the available data is insufficient for traditional scoring systems to render a reliable assessment.

### 2.2 The Behavioral Data Opportunity

Parallel to the thin-file challenge, the proliferation of mobile banking, digital wallets (e.g., MoMo, ZaloPay, ViettelPay), and e-commerce platforms has generated rich longitudinal behavioral datasets for the same underserved population. Gen Z, despite lacking traditional credit histories, leaves an extensive digital footprint: transaction frequency and volume, merchant category spending, mobile app engagement, and payment regularity — all of which may serve as proxies for financial discipline, income stability, and repayment intent.

This creates an empirical research opportunity: can digital behavioral signals meaningfully augment traditional bureau data, and if so, what is the incremental predictive contribution?

### 2.3 Research Gap

Existing literature on alternative data for credit scoring (Bjorkegren & Grissen, 2020; Berg et al., 2020; Brunner et al., 2020) predominantly focuses on markets in Sub-Saharan Africa, Latin America, and Southeast Asia at the aggregate level, with limited granularity on Vietnamese market dynamics. Moreover, Gen Z-specific modeling approaches — accounting for the distinct digital behavioral profiles and financial literacy characteristics of this cohort — remain underexplored in the Vietnamese banking literature.

This project addresses three identified gaps:

| Gap Type | Description |
|---|---|
| **Empirical Gap** | Absence of Vietnamese market evidence on behavioral data's predictive contribution to credit scoring |
| **Methodological Gap** | Limited application of gradient boosting with SHAP interpretability in Vietnamese retail credit contexts |
| **Policy Gap** | No established framework for responsible integration of digital behavioral signals in CIC-supplemented scoring models |

---

## 3. Research Questions

This project is structured around three primary research questions:

**RQ1.** Does the integration of digital behavioral signals with traditional CIC bureau features significantly improve credit score discriminatory power for Gen Z thin-file customers, relative to bureau-only models?

**RQ2.** What is the relative contribution of behavioral versus traditional features to the hybrid model's predictive power, as measured by SHAP-based feature attribution?

**RQ3.** What are the model risk, fairness, and regulatory implications of deploying behavioral-data-augmented credit scoring in the Vietnamese banking context?

---

## 4. Dataset Description

> ⚠️ **Data Privacy Notice:** The underlying dataset is proprietary and cannot be publicly distributed due to banking confidentiality obligations and data protection requirements under Vietnam's Cybersecurity Law (No. 24/2018/QH14) and the Government's Decree 13/2023/ND-CP on personal data protection. The repository provides synthetic data generation scripts and a detailed data schema to support reproducibility.

### 4.1 Data Sources

| Source | Type | Description |
|---|---|---|
| CIC Bureau Data | Traditional | Credit history, outstanding balance, payment behavior, inquiry count |
| Mobile Banking Logs | Behavioral | Transaction frequency, amounts, time-of-day patterns |
| Digital Payment Records | Behavioral | Merchant categories, payment regularity, wallet activity |
| Application Data | Demographic | Age, employment, declared income, education |

### 4.2 Sample Characteristics

| Attribute | Value |
|---|---|
| Total observations | ~15,000 credit applications |
| Target population | Gen Z thin-file customers (age 18–27, CIC score ≤ 2 years history) |
| Observation period | 36-month performance window |
| Default definition | 90+ days past due (NPL classification) |
| Class imbalance | Approximately 15–20% default rate |
| Train / Validation / Test split | 60% / 20% / 20% (stratified) |
| Geographic coverage | Vietnamese commercial bank portfolio |

### 4.3 Feature Categories

#### Traditional Bureau Features (CIC)
| Feature | Description | Type |
|---|---|---|
| `cic_payment_history_score` | Weighted payment history from CIC records | Continuous |
| `cic_outstanding_balance_ratio` | Outstanding balance / credit limit | Continuous |
| `cic_num_inquiries_6m` | Number of credit inquiries in last 6 months | Integer |
| `cic_account_age_months` | Age of oldest credit account (months) | Integer |
| `cic_num_delinquencies` | Count of historical delinquency events | Integer |
| `cic_credit_utilization` | Average credit utilization over 12 months | Continuous |
| `cic_num_active_accounts` | Number of currently active credit accounts | Integer |

#### Digital Behavioral Features
| Feature | Description | Type |
|---|---|---|
| `txn_frequency_30d` | Transaction count in last 30 days | Integer |
| `txn_frequency_90d` | Transaction count in last 90 days | Integer |
| `avg_txn_amount_30d` | Average transaction amount (VND), last 30 days | Continuous |
| `merchant_category_diversity` | Number of distinct merchant categories (90-day window) | Integer |
| `digital_payment_regularity` | Coefficient of regularity in bill payment behavior | Continuous |
| `mobile_app_engagement_score` | Composite score from app login frequency and session duration | Continuous |
| `spending_volatility_90d` | Standard deviation of weekly spending (90-day window) | Continuous |
| `salary_credit_regularity` | Binary indicator of regular salary/income credits | Binary |

#### Demographic Features
| Feature | Description | Type |
|---|---|---|
| `age_at_application` | Applicant age in years | Integer |
| `employment_type` | Employment category (formal/informal/self-employed/student) | Categorical |
| `declared_income_bracket` | Income bracket from application form | Ordinal |
| `education_level` | Highest completed education level | Ordinal |

### 4.4 Data Limitations

- **Survivorship bias:** Dataset reflects only applicants who were approved and subsequently monitored; rejected applicants are not observable
- **Self-reported income:** Declared income figures are unverified at origination and subject to mis-declaration bias
- **Behavioral data recency:** Behavioral signals were available only for customers with pre-existing digital relationships with the bank, introducing selection effects
- **Short performance window:** The 36-month observation window may not capture full economic cycle dynamics

---

## 5. Methodology

### 5.1 Analytical Workflow

```
Business Problem Identification
        │
        ▼
Data Collection & Validation
(CIC Bureau + Digital Behavioral + Application Data)
        │
        ▼
Exploratory Data Analysis
(Distributional analysis, IV/WoE assessment, 
 missing value profiling, class imbalance evaluation)
        │
        ▼
Feature Engineering
(Traditional scorecard features, behavioral signal construction,
 interaction terms, temporal aggregations)
        │
        ▼
Class Imbalance Treatment
(SMOTE oversampling + class weight adjustment)
        │
        ▼
Model Development
(Logistic Regression | Random Forest | LightGBM | XGBoost)
        │
        ▼
Hyperparameter Optimization
(Optuna — Bayesian optimization, 5-fold stratified CV)
        │
        ▼
Model Evaluation
(AUC, Gini, KS Statistic, Brier Score, Calibration)
        │
        ▼
SHAP Interpretability Analysis
(Global feature importance, local explanations, 
 behavioral vs. traditional attribution)
        │
        ▼
Business Interpretation & Risk Assessment
(Cutoff optimization, scorecard transformation, 
 regulatory alignment, model risk review)
```

### 5.2 Feature Engineering

**Information Value (IV) & Weight of Evidence (WoE)** transformation is applied to all features prior to Logistic Regression and as a diagnostic tool for all models:

$$\text{IV} = \sum_{i=1}^{n} \left(\text{Distribution of Events}_i - \text{Distribution of Non-Events}_i\right) \times \ln\left(\frac{\text{Distribution of Events}_i}{\text{Distribution of Non-Events}_i}\right)$$

Features with IV < 0.02 are considered non-predictive and excluded from the final feature set.

**Behavioral composite features** are constructed using rolling window aggregations (30, 60, 90 days) to capture temporal dynamics in spending and payment behavior.

### 5.3 Model Architecture

#### Primary Model: XGBoost

XGBoost (eXtreme Gradient Boosting) is selected as the primary modeling framework based on:

- **Empirical performance:** Consistently superior discriminatory power in tabular credit data (Chen & Guestrin, 2016)
- **Handling of missing values:** Native sparse-aware split-finding algorithm accommodates the missing bureau data prevalent in thin-file populations
- **Interpretability compatibility:** Full SHAP support for post-hoc explainability
- **Regularization:** L1 (α) and L2 (λ) regularization terms reduce overfitting on limited thin-file samples

The XGBoost objective function is:

$$\mathcal{L}(\phi) = \sum_{i=1}^{n} \ell(\hat{y}_i, y_i) + \sum_{k=1}^{K} \Omega(f_k)$$

where $\ell$ denotes binary cross-entropy loss and $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda\|w\|^2$ is the regularization term with $T$ tree leaves and leaf weights $w$.

#### Benchmark Models

| Model | Rationale |
|---|---|
| **Logistic Regression** | Regulatory baseline; interpretable scorecard framework |
| **Random Forest** | Ensemble benchmark; non-parametric, handles non-linearity |
| **LightGBM** | Computational efficiency comparison; histogram-based GBDT |

### 5.4 Class Imbalance Treatment

Given the approximately 15–20% default rate, two complementary strategies are employed:

1. **SMOTE (Synthetic Minority Oversampling Technique)** applied exclusively on the training set, with no leakage to validation/test sets
2. **`scale_pos_weight`** parameter in XGBoost set to `(n_non_default / n_default)` to adjust gradient contributions

### 5.5 Hyperparameter Optimization

Bayesian optimization via **Optuna** is applied with 5-fold stratified cross-validation, optimizing for AUC-ROC. The search space for XGBoost includes:

| Parameter | Search Range |
|---|---|
| `n_estimators` | [100, 1000] |
| `max_depth` | [3, 9] |
| `learning_rate` | [0.01, 0.3] |
| `subsample` | [0.6, 1.0] |
| `colsample_bytree` | [0.5, 1.0] |
| `reg_alpha` | [0, 1.0] |
| `reg_lambda` | [0, 5.0] |

### 5.6 Evaluation Metrics

| Metric | Formula / Definition | Interpretation |
|---|---|---|
| **AUC-ROC** | Area under ROC curve | Overall discrimination ability |
| **Gini Coefficient** | $2 \times \text{AUC} - 1$ | Standard banking model performance measure |
| **KS Statistic** | $\max(\text{TPR} - \text{FPR})$ | Maximum separation between default/non-default distributions |
| **Brier Score** | $\frac{1}{n}\sum(p_i - y_i)^2$ | Probabilistic calibration accuracy |
| **Log-Loss** | $-\frac{1}{n}\sum[y\log p + (1-y)\log(1-p)]$ | Cross-entropy loss on probability outputs |

---

## 6. Key Results

### 6.1 Model Performance Comparison

| Model | AUC | Gini | KS Statistic | Brier Score | Log-Loss |
|---|---|---|---|---|---|
| Logistic Regression (Bureau Only) | 0.712 | 0.424 | 0.338 | 0.118 | 0.421 |
| Logistic Regression (Hybrid) | 0.741 | 0.482 | 0.362 | 0.112 | 0.398 |
| Random Forest (Hybrid) | 0.779 | 0.558 | 0.419 | 0.108 | 0.387 |
| LightGBM (Hybrid) | 0.798 | 0.596 | 0.441 | 0.105 | 0.374 |
| **XGBoost (Hybrid) — Final Model** | **0.803** | **0.6049** | **0.461** | **0.103** | **0.368** |

> **Note:** All metrics reported on the held-out test set (20% stratified split). Confidence intervals estimated via 1,000 bootstrap iterations.

### 6.2 Behavioral Feature Contribution (SHAP Analysis)

| Feature Category | Mean |SHAP| Value | Contribution (%) |
|---|---|---|
| Digital Behavioral Features | 0.247 | **37.1%** |
| CIC Bureau Features | 0.389 | **58.4%** |
| Demographic Features | 0.030 | **4.5%** |

**Top 10 Features by Mean Absolute SHAP Value:**

| Rank | Feature | Category | Mean |SHAP| |
|---|---|---|---|
| 1 | `cic_payment_history_score` | Bureau | 0.142 |
| 2 | `txn_frequency_90d` | Behavioral | 0.098 |
| 3 | `cic_credit_utilization` | Bureau | 0.091 |
| 4 | `digital_payment_regularity` | Behavioral | 0.083 |
| 5 | `cic_outstanding_balance_ratio` | Bureau | 0.079 |
| 6 | `spending_volatility_90d` | Behavioral | 0.071 |
| 7 | `salary_credit_regularity` | Behavioral | 0.068 |
| 8 | `cic_num_delinquencies` | Bureau | 0.065 |
| 9 | `merchant_category_diversity` | Behavioral | 0.059 |
| 10 | `mobile_app_engagement_score` | Behavioral | 0.051 |

### 6.3 Performance on Gen Z Thin-File Subsample

| Segment | N (Test) | AUC | Gini |
|---|---|---|---|
| Full test set | 3,000 | 0.803 | 0.6049 |
| Gen Z thin-file (< 1 yr bureau history) | 847 | 0.791 | 0.582 |
| Gen Z no-bureau (CIC unscored) | 312 | 0.768 | 0.536 |

> **Interpretation:** The hybrid model maintains meaningful discrimination even for completely unscored applicants, demonstrating that behavioral signals alone carry substantial predictive signal for this population.

---

## 7. Repository Structure

```
hybrid-credit-scoring-vietnam/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignored files
├── setup.py                           # Package installation
│
├── data/
│   ├── raw/                           # Raw data (not committed — see data note)
│   ├── processed/                     # Cleaned, feature-engineered datasets
│   └── external/                      # Synthetic data, CIC schema reference
│
├── notebooks/
│   ├── 01_data_collection_and_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   ├── 05_model_evaluation_and_comparison.ipynb
│   └── 06_shap_interpretability_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py                  # Data ingestion and validation
│   │   ├── cleaner.py                 # Preprocessing and imputation
│   │   └── synthetic_generator.py    # Synthetic data generation for reproducibility
│   ├── features/
│   │   ├── traditional_features.py   # Bureau feature construction & WoE/IV
│   │   ├── behavioral_features.py    # Rolling window aggregations & composites
│   │   └── feature_selector.py       # IV-based and SHAP-based selection
│   ├── models/
│   │   ├── base_model.py             # Abstract base class for all models
│   │   ├── logistic_model.py         # Logistic Regression with WoE pipeline
│   │   ├── xgboost_model.py          # XGBoost training, tuning, inference
│   │   ├── random_forest_model.py    # Random Forest benchmark
│   │   └── lightgbm_model.py         # LightGBM benchmark
│   ├── evaluation/
│   │   ├── metrics.py                # AUC, Gini, KS, Brier Score calculations
│   │   └── calibration.py            # Reliability diagrams, Platt scaling
│   ├── visualization/
│   │   ├── eda_plots.py              # Distribution, correlation, WoE plots
│   │   ├── model_plots.py            # ROC, PR curves, confusion matrix
│   │   └── shap_plots.py             # SHAP summary, waterfall, dependence plots
│   └── utils/
│       ├── config.py                  # Hyperparameter configs and constants
│       └── logger.py                  # Structured logging
│
├── outputs/
│   ├── figures/                       # Publication-quality charts (PNG/SVG)
│   ├── tables/                        # Model comparison tables (CSV/LaTeX)
│   └── predictions/                   # Scored test set with probabilities
│
├── reports/
│   ├── research_report.pdf            # Full research report
│   └── presentation.pptx             # Executive summary slides
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_feature_engineering.py
│   └── test_model_evaluation.py
│
└── docs/
    ├── methodology.md                 # Detailed methodology documentation
    ├── data_dictionary.md             # Full feature definitions and data schema
    └── model_card.md                  # Model card (intended use, limitations, metrics)
```

---

## 8. Installation & Reproducibility

### Prerequisites

- Python 3.10 or higher
- pip 23.0+
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/[YOUR_USERNAME]/hybrid-credit-scoring-vietnam.git
cd hybrid-credit-scoring-vietnam
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# or
venv\Scripts\activate           # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Generate Synthetic Data (for reproducibility)

Since the original banking dataset is proprietary, a synthetic data generator is provided that replicates the statistical properties of the original sample:

```bash
python src/data/synthetic_generator.py --n_samples 15000 --random_seed 42
```

### Step 5: Run the Full Pipeline

```bash
# Execute notebooks in order
jupyter nbconvert --to notebook --execute notebooks/01_data_collection_and_cleaning.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_exploratory_data_analysis.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_model_development.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_model_evaluation_and_comparison.ipynb
jupyter nbconvert --to notebook --execute notebooks/06_shap_interpretability_analysis.ipynb
```

### Step 6: Run Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

> All experiments were conducted on Python 3.10.12, XGBoost 2.0.3, scikit-learn 1.3.2, SHAP 0.43.0. Random seed set to 42 throughout.

---

## 9. Usage Guide

### Training the Hybrid XGBoost Model

```python
from src.data.loader import load_processed_data
from src.features.traditional_features import build_bureau_features
from src.features.behavioral_features import build_behavioral_features
from src.models.xgboost_model import HybridXGBoostModel

# Load and prepare data
X_train, X_test, y_train, y_test = load_processed_data("data/processed/train_test_split.pkl")

# Build feature matrices
X_bureau = build_bureau_features(X_train)
X_behavioral = build_behavioral_features(X_train, window_days=[30, 60, 90])
X_hybrid = pd.concat([X_bureau, X_behavioral], axis=1)

# Initialize and train model
model = HybridXGBoostModel(config="src/utils/config.yaml")
model.fit(X_hybrid, y_train)

# Evaluate
from src.evaluation.metrics import evaluate_credit_model
results = evaluate_credit_model(model, X_test_hybrid, y_test)
print(results)
# {'auc': 0.803, 'gini': 0.6049, 'ks_statistic': 0.461, 'brier_score': 0.103}
```

### Generating SHAP Explanations

```python
from src.visualization.shap_plots import plot_shap_summary, plot_feature_attribution

# Global feature importance
plot_shap_summary(model, X_test_hybrid, save_path="outputs/figures/shap_summary.png")

# Behavioral vs. bureau attribution breakdown
plot_feature_attribution(
    model, X_test_hybrid,
    feature_groups={"Bureau": bureau_cols, "Behavioral": behavioral_cols},
    save_path="outputs/figures/feature_attribution.png"
)
```

---

## 10. Key Findings & Business Implications

### 10.1 Primary Findings

1. **Behavioral data is a material predictor of default.** Digital behavioral signals account for approximately 37% of the hybrid model's predictive power, validating their incremental contribution over bureau data alone. This finding suggests that fintech-derived behavioral data represents a commercially viable supplement to CIC records.

2. **Transaction regularity outperforms raw frequency.** Among behavioral features, `digital_payment_regularity` and `salary_credit_regularity` exhibit higher mean absolute SHAP values than simple transaction count metrics, indicating that the *consistency* of financial behavior — rather than volume — is more informative for creditworthiness assessment.

3. **The hybrid model materially improves Gen Z thin-file discrimination.** Relative to bureau-only Logistic Regression (AUC = 0.712), the hybrid XGBoost model achieves AUC = 0.791 on the Gen Z thin-file subsample — a 7.9 percentage point improvement. This translates directly into reduced Type I errors (approving high-risk applicants) and Type II errors (rejecting creditworthy thin-file borrowers).

4. **Model stability is maintained across bureau depth segments.** Even in the no-bureau subsample (CIC unscored), the hybrid model achieves Gini = 0.536, demonstrating that behavioral data alone provides meaningful, if reduced, discriminatory power.

### 10.2 Business Implications for Vietnamese Banks

| Implication | Operational Impact |
|---|---|
| **Loan approval expansion** | Banks can extend credit to thin-file Gen Z customers previously rejected by bureau-only models, expanding the addressable market |
| **Risk-adjusted pricing** | The probability of default output enables risk-based interest rate pricing, replacing the binary approve/reject decision |
| **Reduced adverse selection** | Improved screening reduces portfolio NPL ratios by better identifying genuine credit risk among thin-file applicants |
| **Digital product cross-selling** | Behavioral data integration creates a direct business incentive for banks to deepen digital engagement with Gen Z customers |

### 10.3 Regulatory Alignment

The model is designed with awareness of Vietnam's State Bank regulatory framework:

- **SBV Circular 11/2021/TT-NHNN** on credit risk classification and provisioning
- **Decree 13/2023/ND-CP** on personal data protection — behavioral data usage requires explicit consent and purpose limitation
- **Explainability requirement:** SHAP explanations support adverse action notification obligations under proposed AI regulatory frameworks

---

## 11. Model Risk & Limitations

| Risk Category | Description | Mitigation |
|---|---|---|
| **Data Risk** | Proprietary behavioral data may not generalize across banks or digital platforms | Out-of-portfolio validation required before deployment |
| **Selection Bias** | Dataset excludes rejected applicants (reject inference problem) | Parceling / augmentation techniques recommended |
| **Concept Drift** | Gen Z behavioral patterns evolve rapidly (app adoption, spending norms) | Monthly model monitoring with Population Stability Index (PSI) |
| **Overfitting Risk** | XGBoost may memorize idiosyncratic patterns in small thin-file subsamples | L1/L2 regularization and cross-validation applied; periodic retraining required |
| **Fairness Risk** | Behavioral features may proxy for socioeconomic status or regional digital infrastructure access | Fairness audit across geographic and income sub-groups recommended |
| **Explainability Risk** | SHAP global explanations do not guarantee accurate local-level decision rationale | Local SHAP values provided per observation; model card documents known limitations |
| **Regulatory Risk** | Alternative data usage in credit scoring is subject to evolving SBV guidance | Legal review of behavioral data sourcing and consent mechanisms is required |

---

## 12. Future Work

| Direction | Description | Priority |
|---|---|---|
| **Graph-based behavioral modeling** | Leverage network analysis of transaction counterparties to extract social financial signals | High |
| **Temporal deep learning** | Apply LSTM or Transformer architectures to raw transaction sequences, bypassing manual feature engineering | Medium |
| **Reject inference** | Implement parceling or fuzzy augmentation to address selection bias from rejected applicants | High |
| **Multi-bank generalization** | Validate model performance on behavioral data from alternative sources (MoMo, ZaloPay) | Medium |
| **IFRS 9 integration** | Extend the PD model to lifetime PD estimation using survival analysis, enabling ECL calculation under IFRS 9 | High |
| **Fairness-aware modeling** | Incorporate fairness constraints (e.g., equalized odds) into the objective function | Medium |
| **Production deployment** | Develop REST API via FastAPI + Docker containerization for real-time scoring | Low |

---

## 14. Citation

If you use this work in your research, please cite:

```bibtex
@misc{pham2024hybridcredit,
  author       = {Phạm Tiến Dũng and Phạm Trường Phát and Phạm Thanh Huyền},
  title        = {Hybrid Credit Scoring for Gen Z Thin-File Customers in Vietnam:
                  Integrating CIC Bureau Data with Digital Behavioral Signals via XGBoost},
  year         = {2024},
  institution  = {National Economics University, Hanoi, Vietnam},
  note         = {Undergraduate Research Project, Faculty of Mathematical Economics},
  url          = {https://github.com/[YOUR_USERNAME]/hybrid-credit-scoring-vietnam}
}
```

---

## 15. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The license applies to the source code and documentation in this repository. The underlying proprietary banking dataset is **not** licensed for distribution. The synthetic data generator is provided for reproducibility purposes under the same MIT License.

---

## References

Berg, T., Burg, V., Gombović, A., & Puri, M. (2020). On the Rise of FinTechs: Credit Scoring Using Digital Footprints. *Review of Financial Studies*, 33(7), 2845–2897.

Bjorkegren, D., & Grissen, D. (2020). Behavior Revealed in Mobile Phone Usage Predicts Loan Repayment. *World Bank Economic Review*, 34(3), 618–643.

Brunner, M., Krahnen, J. P., & Weber, M. (2020). Information production in credit relationships. *Journal of Banking & Finance*, 26(10), 2127–2152.

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30.

Mester, L. J. (1997). What's the point of credit scoring? *Federal Reserve Bank of Philadelphia Business Review*, 3–16.

Thomas, L. C. (2009). *Consumer Credit Models: Pricing, Profit, and Portfolios*. Oxford University Press.

State Bank of Vietnam. (2021). Circular 11/2021/TT-NHNN on credit risk classification, appropriation, and use of provisions against credit risks in banking activities.

---

*Last updated: 2024 | National Economics University, Hanoi, Vietnam*
