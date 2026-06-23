# Methodology: Hybrid Credit Scoring Framework

**Project:** Hybrid Credit Scoring for Gen Z Thin-File Customers in Vietnam  
**Document type:** Technical Methodology Reference  
**Version:** 1.0.0

---

## Table of Contents

1. [Research Design Overview](#1-research-design-overview)
2. [Data Architecture](#2-data-architecture)
3. [Data Preprocessing Pipeline](#3-data-preprocessing-pipeline)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Development Framework](#5-model-development-framework)
6. [Hyperparameter Optimization](#6-hyperparameter-optimization)
7. [Evaluation Framework](#7-evaluation-framework)
8. [SHAP Interpretability Framework](#8-shap-interpretability-framework)
9. [Model Risk Assessment](#9-model-risk-assessment)
10. [Statistical Assumptions and Diagnostics](#10-statistical-assumptions-and-diagnostics)

---

## 1. Research Design Overview

### 1.1 Analytical Paradigm

This study follows a **supervised binary classification** paradigm. The outcome variable `default_flag` is defined as:

$$y_i = \begin{cases} 1 & \text{if borrower } i \text{ is } \geq 90 \text{ days past due within } t = 36 \text{ months} \\ 0 & \text{otherwise} \end{cases}$$

This definition aligns with the State Bank of Vietnam's NPL classification under Circular 11/2021/TT-NHNN (Group 3 classification and above) and is consistent with the standard 90-day past-due default definition used in Basel II/III credit risk frameworks.

### 1.2 Research Hypothesis

**H₀:** The hybrid model (bureau + behavioral features) does not produce statistically significant improvement in AUC over the bureau-only baseline.

**H₁:** Integration of digital behavioral signals with CIC bureau data yields a statistically significant improvement in AUC (α = 0.05), evaluated via DeLong's test for correlated ROC curves (DeLong et al., 1988).

### 1.3 Identification Strategy

The incremental contribution of behavioral features is identified by comparing two nested models:

| Model | Feature Set | Estimated AUC |
|---|---|---|
| M₀ (Bureau-Only) | CIC bureau features only | 0.712 |
| M₁ (Hybrid) | Bureau + Behavioral + Demographic | 0.803 |

The **incremental AUC lift** = 0.803 - 0.712 = **0.091**, statistically tested via bootstrap confidence intervals (B = 1,000 iterations).

---

## 2. Data Architecture

### 2.1 Multi-Source Data Integration

The hybrid dataset is constructed by joining three independent data sources at the **application ID level**:

```
CIC Bureau Extract          Digital Banking Logs         Application Form
(per-applicant snapshot)    (per-applicant history)      (at origination)
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                         Joined on: application_id
                         Window: 90 days pre-application
                                    │
                            Hybrid Feature Matrix
                         shape: (N_applicants, P_features)
```

### 2.2 Temporal Alignment

All behavioral features are computed using a **lookback window ending on the application date** (`t = 0`). This is critical to prevent data leakage:

- Features use data from `[t - 90d, t = 0]`
- The performance label uses data from `[t = 0, t + 36 months]`
- No post-application behavioral data is used in feature construction

### 2.3 Population Definition

**In-scope:** Retail credit applicants aged 18–35, first-time borrowers or those with fewer than 24 months of CIC bureau history, with at least 30 days of digital banking activity prior to application.

**Out-of-scope:** Corporate applicants, borrowers with existing active NPL status at application, and applicants with no digital banking footprint (excluded from the thin-file behavioral modeling track).

---

## 3. Data Preprocessing Pipeline

### 3.1 Missing Value Treatment

| Feature Type | Treatment Strategy | Justification |
|---|---|---|
| Bureau features (CIC) | Separate missing indicator + median imputation | Missing bureau data is informative (thin-file signal) |
| Behavioral features | Zero imputation for count features | Zero transaction count is a legitimate observed value |
| Behavioral features | Median imputation for ratio features | Avoids zero-inflation in division-based features |
| Demographic features | Mode imputation for categoricals | Most frequent category as conservative default |

For CIC bureau features, a binary missing indicator `cic_{feature}_missing` is added alongside the imputed value. This preserves the informational content of missingness — an applicant with no CIC payment history is substantively different from one with a perfect payment history.

### 3.2 Outlier Treatment

Outliers are treated using **Winsorization at the 1st and 99th percentiles** for continuous features, applied to training data only. Test and validation set winsorization uses the training-set thresholds to prevent leakage.

```
x_winsorized = clip(x, p1_train, p99_train)
```

### 3.3 Categorical Encoding

Nominal categorical features (`employment_type`, `education_level`) are encoded using **Target Encoding with 5-fold out-of-fold estimation** to prevent target leakage. Ordinal features use **label encoding** preserving ordinal relationships.

### 3.4 Class Imbalance

The dataset exhibits approximately 17.5% default rate (positive class). Two complementary strategies are applied:

**Strategy 1 — SMOTE Oversampling:** Applied exclusively on the training fold during cross-validation and on the final training set. SMOTE generates synthetic minority samples by interpolation in feature space:

$$\tilde{x} = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \sim \text{Uniform}(0, 1)$$

where $x_{nn}$ is a randomly chosen k-nearest neighbor of $x_i$ in the minority class.

**Strategy 2 — `scale_pos_weight`:** For XGBoost and LightGBM, the `scale_pos_weight` parameter is set to:

$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} \approx 4.71$$

This re-weights the gradient contributions of positive-class samples without altering the feature space.

---

## 4. Feature Engineering

### 4.1 Weight of Evidence (WoE) and Information Value (IV)

For regulatory-compliant scoring, all features are evaluated using **Weight of Evidence (WoE)** transformation:

$$\text{WoE}_i = \ln\left(\frac{\text{Distribution of Events}_i}{\text{Distribution of Non-Events}_i}\right) = \ln\left(\frac{n_{1i}/N_1}{n_{0i}/N_0}\right)$$

**Information Value (IV)** aggregates predictive power across all bins:

$$\text{IV} = \sum_{i=1}^{B} \left(\frac{n_{1i}}{N_1} - \frac{n_{0i}}{N_0}\right) \times \text{WoE}_i$$

**IV Interpretation:**

| IV Range | Predictive Power | Action |
|---|---|---|
| < 0.02 | None — negligible | Remove |
| 0.02 – 0.10 | Weak | Review |
| 0.10 – 0.30 | Medium | Include |
| 0.30 – 0.50 | Strong | Include |
| > 0.50 | Suspicious | Investigate for leakage |

WoE transformation is applied for Logistic Regression to maintain a log-odds linear relationship. For tree-based models, raw features are used alongside WoE as an alternative encoding.

### 4.2 Behavioral Feature Engineering

#### Rolling Window Aggregations

For each applicant $i$, transaction features are computed over rolling windows $W \in \{30, 60, 90\}$ days prior to application:

$$\text{txn\_frequency}_{i,W} = \sum_{t \in [t_0 - W, t_0]} \mathbf{1}[\text{transaction at } t]$$

$$\text{avg\_txn\_amount}_{i,W} = \frac{\sum_{t \in [t_0 - W, t_0]} \text{amount}_t}{|\{t : \text{transaction at } t\}|}$$

#### Spending Volatility

$$\sigma_{\text{spending}, i, 90d} = \text{std}\left(\{\text{weekly\_spend}_{i,w}\}_{w=1}^{13}\right)$$

where weekly_spend is the total transaction amount in each of the 13 weeks preceding application. Higher volatility indicates financial instability.

#### Digital Payment Regularity

Regularity is measured as the **coefficient of regularity** of bill payment intervals:

$$R_i = 1 - \frac{\text{CV}(\text{inter\_payment\_intervals}_i)}{2}, \quad R_i \in [0, 1]$$

where CV is the coefficient of variation. $R_i = 1$ indicates perfectly regular payments; $R_i = 0$ indicates maximum irregularity.

#### Merchant Category Diversity

Diversity across merchant categories is measured using a **normalized Shannon entropy**:

$$H_i = -\sum_{k=1}^{K} p_{ik} \ln(p_{ik}), \quad \tilde{H}_i = \frac{H_i}{\ln K}$$

where $p_{ik}$ is the proportion of spending in merchant category $k$ over the 90-day window.

### 4.3 Interaction Features

Selected interaction terms between bureau and behavioral features are constructed:

- `cic_utilization × digital_payment_regularity`: Tests whether high utilization is more predictive in low-regularity segments
- `salary_credit_regularity × cic_num_delinquencies`: Interaction between behavioral income signal and historical delinquency

---

## 5. Model Development Framework

### 5.1 Train / Validation / Test Split

Data is split **once** using stratified sampling:

```
Total: N = 15,000
├── Training Set:   60% → 9,000 observations
├── Validation Set: 20% → 3,000 observations  (for early stopping, threshold tuning)
└── Test Set:       20% → 3,000 observations  (held out; used only for final reporting)
```

Stratification ensures the 17.5% default rate is preserved across all splits.

### 5.2 Cross-Validation Strategy

Model selection uses **5-fold Stratified Cross-Validation** on the training set. The final model is retrained on the full training set using the best hyperparameters identified by CV.

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### 5.3 XGBoost Architecture

The primary model uses the following XGBoost configuration after Optuna optimization:

| Parameter | Optimized Value | Search Range | Purpose |
|---|---|---|---|
| `n_estimators` | 500 | [100, 1000] | Number of boosting rounds |
| `max_depth` | 6 | [3, 9] | Tree complexity control |
| `learning_rate` | 0.05 | [0.005, 0.3] | Shrinkage to prevent overfitting |
| `subsample` | 0.80 | [0.6, 1.0] | Row subsampling ratio |
| `colsample_bytree` | 0.80 | [0.5, 1.0] | Column subsampling ratio |
| `min_child_weight` | 5 | [1, 20] | Minimum sum of instance weight in a leaf |
| `reg_alpha` | 0.10 | [0.0, 2.0] | L1 regularization |
| `reg_lambda` | 1.00 | [0.0, 5.0] | L2 regularization |
| `scale_pos_weight` | 4.71 | Fixed | Class imbalance correction |

Early stopping is applied on the validation AUC with `early_stopping_rounds = 50`.

---

## 6. Hyperparameter Optimization

### 6.1 Optuna Bayesian Optimization

Hyperparameters are optimized using **Optuna's Tree-structured Parzen Estimator (TPE)** algorithm, which models the objective function as:

$$p(\lambda | y < y^*) \cdot p(\lambda | y \geq y^*)$$

where $y^*$ is a quantile threshold separating "good" from "bad" configurations. TPE is sample-efficient compared to grid search, finding high-performing configurations in significantly fewer trials.

**Search protocol:**
- `n_trials = 100`
- Pruning: MedianPruner (stops unpromising trials early)
- Objective: Mean cross-validated AUC-ROC on training set
- Parallelization: `n_jobs = -1` (all available cores)

### 6.2 Convergence Monitoring

Optimization is considered converged when the best AUC improves by less than 0.001 over 30 consecutive trials. The study history is saved for reproducibility and post-hoc analysis.

---

## 7. Evaluation Framework

### 7.1 Primary Metrics

**AUC-ROC (Area Under the Receiver Operating Characteristic Curve):**

$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt = P(\hat{p}_{\text{default}} > \hat{p}_{\text{non-default}})$$

AUC measures the probability that a randomly selected defaulter is assigned a higher predicted probability than a randomly selected non-defaulter. AUC = 0.80 is the conventional threshold for a "good" credit scoring model.

**Gini Coefficient:**

$$\text{Gini} = 2 \times \text{AUC} - 1$$

Gini is the standard banking industry performance metric for credit scorecards. Industry minimum for production models is typically Gini ≥ 0.40.

**Kolmogorov-Smirnov (KS) Statistic:**

$$\text{KS} = \max_{t} |\text{CDF}_{\text{default}}(t) - \text{CDF}_{\text{non-default}}(t)|$$

KS measures the maximum separation between the cumulative score distributions of defaulters and non-defaulters. KS > 0.30 is typically required for production models.

### 7.2 Calibration Assessment

Model calibration — the alignment between predicted probabilities and observed default rates — is evaluated using:

1. **Reliability Diagram (Calibration Curve):** Plots mean predicted probability against observed default frequency in 10 equal-frequency bins. A perfectly calibrated model lies on the diagonal.

2. **Brier Score:**

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2$$

Lower is better; Brier Score = 0 is perfect calibration. The **Brier Skill Score** (BSS) compares to a climatological baseline.

3. **Expected Calibration Error (ECE):**

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} |\overline{y}_{B_b} - \overline{\hat{p}}_{B_b}|$$

If calibration is inadequate, **Platt Scaling** (logistic calibration) is applied as a post-hoc correction.

### 7.3 Confidence Intervals via Bootstrap

All reported AUC, Gini, and KS values include 95% confidence intervals estimated via **stratified bootstrap** (B = 1,000 iterations):

$$\text{CI}_{95\%} = [\hat{\theta}_{(0.025)}, \hat{\theta}_{(0.975)}]$$

### 7.4 Population Stability Index (PSI)

PSI monitors distributional shift between development and monitoring populations:

$$\text{PSI} = \sum_{i=1}^{B} \left(\%\text{Actual}_i - \%\text{Expected}_i\right) \times \ln\left(\frac{\%\text{Actual}_i}{\%\text{Expected}_i}\right)$$

| PSI Value | Interpretation | Action |
|---|---|---|
| < 0.10 | No significant change | No action |
| 0.10 – 0.20 | Minor change | Monitor closely |
| > 0.20 | Major change | Trigger model review / retraining |

---

## 8. SHAP Interpretability Framework

### 8.1 SHAP Values

SHAP (SHapley Additive exPlanations) values are computed using the **TreeSHAP** algorithm, which provides exact SHAP values for tree-based models in polynomial time (Lundberg et al., 2020).

For prediction $f(x)$, SHAP decomposes the output as:

$$f(x) = \phi_0 + \sum_{j=1}^{P} \phi_j(x)$$

where $\phi_0$ is the expected model output (base value) and $\phi_j(x)$ is the SHAP value of feature $j$ for observation $x$.

SHAP values satisfy three axioms:
- **Efficiency:** $\sum_j \phi_j = f(x) - E[f(X)]$
- **Symmetry:** Features with equal contributions receive equal SHAP values
- **Dummy:** Features that do not affect the model receive zero SHAP values

### 8.2 Feature Group Attribution

To quantify the contribution of behavioral vs. bureau features, SHAP values are aggregated by feature group:

$$\text{Contribution}_{G} = \frac{\sum_{j \in G} \overline{|\phi_j|}}{\sum_{j=1}^{P} \overline{|\phi_j|}} \times 100\%$$

where $G \in \{\text{Bureau}, \text{Behavioral}, \text{Demographic}\}$ and $\overline{|\phi_j|}$ is the mean absolute SHAP value of feature $j$ across the test set.

### 8.3 Visualization Suite

| Visualization | Purpose | Audience |
|---|---|---|
| SHAP Summary Plot (beeswarm) | Global feature importance + direction of effect | Technical |
| SHAP Waterfall Plot | Local explanation for individual prediction | Business / Regulatory |
| SHAP Dependence Plot | Feature-level effect as function of feature value | Technical |
| SHAP Force Plot | Visual decomposition of a single prediction | Client-facing |
| Feature Attribution Bar Chart | Bureau vs. Behavioral group contribution | Executive |

---

## 9. Model Risk Assessment

### 9.1 Model Risk Categories

This model is assessed against the following risk categories per the Model Risk Management framework (SR 11-7, Federal Reserve):

| Risk | Assessment | Mitigation |
|---|---|---|
| **Data Risk** | Proprietary behavioral data; potential selection bias from digital-only customers | Documented limitations; synthetic data for validation |
| **Specification Risk** | Behavioral window (90d) may not be optimal | Sensitivity analysis across 30/60/90d windows |
| **Estimation Risk** | Possible overfitting on thin-file subsample (N=847) | L1/L2 regularization; early stopping; cross-validation |
| **Implementation Risk** | Pipeline dependencies on data freshness | Automated data validation via Great Expectations |
| **Use Risk** | Model may be used outside intended population | Model card documents intended use boundaries |

### 9.2 Reject Inference

A known limitation of this model is **survivorship bias** from reject inference: only applicants who were approved and subsequently monitored are observable. Rejected applicants' performance is unobservable, creating a biased training sample.

This is a fundamental limitation of all retrospective credit scoring datasets. Parceling and augmentation approaches are recommended for production deployment, as documented in the Future Work section of the README.

---

## 10. Statistical Assumptions and Diagnostics

### 10.1 XGBoost Assumptions

XGBoost is a **non-parametric ensemble method** and does not require distributional assumptions on the features. Key assumptions are:

- **Stationarity:** The relationship between features and default probability is assumed stable over the performance window (36 months). Concept drift is monitored via PSI.
- **Feature independence (partial):** While XGBoost handles correlated features, highly collinear features (ρ > 0.90) are removed in preprocessing to improve tree diversity.
- **I.I.D. observations:** Individual credit applications are assumed independent — an assumption that may be violated if applicants share household financial circumstances.

### 10.2 Diagnostic Checks

| Diagnostic | Tool | Pass Criterion |
|---|---|---|
| Multicollinearity | Correlation matrix; VIF for LR model | ρ < 0.90; VIF < 10 |
| Feature drift (train vs. test) | PSI per feature | PSI < 0.10 |
| Label leakage | Temporal validation; feature creation audit | No post-application features |
| Calibration | Reliability diagram; Brier Score | BSS > 0 vs. climatology |
| Overfitting | Train vs. validation AUC gap | Gap < 0.02 |

---

## References

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*, 785–794.

DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves. *Biometrics*, 44(3), 837–845.

Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.

Mester, L. J. (1997). What's the point of credit scoring? *Federal Reserve Bank of Philadelphia Business Review*, 3–16.

Board of Governors of the Federal Reserve System. (2011). Supervisory Guidance on Model Risk Management (SR 11-7).

State Bank of Vietnam. (2021). Circular 11/2021/TT-NHNN on credit risk classification, appropriation, and use of provisions.

Thomas, L. C. (2009). *Consumer Credit Models: Pricing, Profit, and Portfolios*. Oxford University Press.
