# Model Card — Hybrid XGBoost Credit Scoring Model

> Following the Model Card framework proposed by Mitchell et al. (2019) and
> adapted for credit risk applications in line with emerging regulatory guidance
> on AI model documentation.

---

## Model Details

| Attribute | Value |
|---|---|
| **Model Name** | Hybrid Credit Scoring — XGBoost |
| **Model Version** | 1.0.0 |
| **Model Type** | Binary classification (default / non-default) |
| **Algorithm** | eXtreme Gradient Boosting (XGBoost 2.0+) |
| **Output** | P(default) ∈ [0, 1] and derived credit score ∈ [300, 850] |
| **Development Date** | 2024 |
| **Developers** | Phạm Tiến Dũng, Phạm Trường Phát, Phạm Thanh Huyền |
| **Institution** | National Economics University (NEU), Hanoi, Vietnam |
| **License** | MIT (code); proprietary data not redistributed |

---

## Intended Use

### Primary Intended Use

The model is designed to assess the credit risk of **Generation Z (age 18–27) thin-file loan applicants** in the Vietnamese retail banking context — specifically, applicants with insufficient CIC bureau history for reliable scoring by conventional methods.

### Primary Intended Users

- **Credit risk teams** at Vietnamese commercial banks seeking to extend scoring to underserved young borrowers
- **Researchers** studying alternative data in credit scoring
- **Graduate students** learning practical credit model development

### Out-of-Scope Uses

The model should **not** be used for:

- Applicants outside the 18–40 age range without revalidation
- Corporate or SME credit scoring
- Portfolio-level ECL calculation without lifecycle PD calibration (see IFRS 9 survival analysis companion project)
- Markets outside Vietnam without transfer learning adaptation and local revalidation
- Sole basis for adverse action (model must be part of a human-in-the-loop workflow)

---

## Training Data

### Data Sources

| Source | Coverage | Features |
|---|---|---|
| CIC Bureau | Approved applicants with ≥ 1 credit product | 7 bureau features |
| Mobile Banking Logs | Customers with digital banking relationship ≥ 3 months | 8 behavioral features |
| Application Form | All applicants | 4 demographic features |

### Data Characteristics

- **Observation period:** 36-month performance window
- **Default definition:** 90+ days past due (NPL classification per SBV)
- **Approximate sample size:** ~15,000 credit applications
- **Class imbalance:** ~18% default rate
- **Geographic coverage:** Vietnam — single bank portfolio
- **Time period:** Not disclosed (proprietary)

### Data Exclusions

- Rejected applicants excluded (reject inference not applied — see Limitations)
- Applications with incomplete demographic data excluded
- Applications with behavioral observation window < 30 days excluded

---

## Evaluation Data

- 20% stratified hold-out test set (n ≈ 3,000)
- Additional evaluation on Gen Z thin-file subsample (n ≈ 847)
- No-bureau subsample (CIC unscored, n ≈ 312)

---

## Model Performance

### Overall Test Set Performance

| Metric | Value |
|---|---|
| AUC-ROC | **0.803** |
| Gini Coefficient | **0.6049** |
| KS Statistic | **0.461** |
| Brier Score | **0.103** |
| Log-Loss | **0.368** |

> Confidence intervals (95%) estimated via 1,000 bootstrap iterations.

### Performance on Critical Subgroups

| Subgroup | N | AUC | Gini |
|---|---|---|---|
| Full test set | 3,000 | 0.803 | 0.6049 |
| Gen Z thin-file (< 1 yr CIC history) | 847 | 0.791 | 0.582 |
| No-bureau (CIC unscored) | 312 | 0.768 | 0.536 |

> The model demonstrates meaningful discrimination even for completely unscored applicants,
> validating behavioral data's standalone predictive value.

---

## Ethical Considerations

### Fairness

The model has not been formally audited for demographic fairness across protected characteristics (gender, ethnicity, geographic region). The following risks have been identified:

- **Geographic digital divide:** Behavioral features may advantage applicants in urban areas with higher digital infrastructure access, potentially disadvantaging rural applicants.
- **Socioeconomic proxy:** Transaction frequency and amount may partially proxy for income, which in turn correlates with socioeconomic background.
- **Employment type bias:** The `employment_type` categorical feature may introduce differential treatment of informal workers.

**Recommended mitigations:**
- Conduct subgroup performance analysis before production deployment
- Apply equalized odds constraints or adversarial debiasing if significant disparities are found
- Monitor PSI (Population Stability Index) across demographic segments monthly

### Consent and Privacy

- Digital behavioral data usage requires explicit customer consent per Vietnam's Decree 13/2023/ND-CP on personal data protection
- Data minimization principles should be applied: only features necessary for the scoring decision should be collected
- Applicants must be informed if behavioral data is used in the credit decision

### Explainability and Adverse Action

- SHAP local explanations are generated per-observation and logged for each credit decision
- In case of adverse action (loan denial), the top 3 contributing features by SHAP value are available for disclosure
- This aligns with emerging requirements under Vietnam's draft AI regulation framework

---

## Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| **Reject inference** | Model trained only on approved applicants; performance on rejected population unknown | Parceling/augmentation planned as future work |
| **Single institution** | May not generalize across banks with different customer profiles | Out-of-portfolio validation required |
| **Behavioral data availability** | 37% of model's power depends on behavioral features; applicants without digital banking history revert to bureau-only scoring | Fallback scoring pipeline for non-digital applicants |
| **Concept drift** | Gen Z financial behavior evolves rapidly | Monthly PSI monitoring and annual retraining recommended |
| **Short performance window** | 36-month window may not capture full economic cycle | Extend to 60 months when data becomes available |
| **Calibration on thin-file** | Probability calibration is less reliable on the no-bureau subsample (n=312) | Apply Platt scaling recalibration on the thin-file subsample separately |

---

## Quantitative Analyses

### Feature Attribution

| Feature Group | Mean |SHAP| Contribution | Share |
|---|---|---|
| CIC Bureau Features | 0.389 | 58.4% |
| Digital Behavioral | 0.247 | 37.1% |
| Demographic | 0.030 | 4.5% |

### Top Features (Mean |SHAP|)

| Rank | Feature | Category |
|---|---|---|
| 1 | `cic_payment_history_score` | Bureau |
| 2 | `txn_frequency_90d` | Behavioral |
| 3 | `cic_credit_utilization` | Bureau |
| 4 | `digital_payment_regularity` | Behavioral |
| 5 | `spending_volatility_90d` | Behavioral |

---

## Caveats and Recommendations

1. **This model is a research prototype.** Production deployment requires formal model validation per SBV model risk management guidelines, legal review of behavioral data sourcing, and integration with the bank's existing credit decision workflow.

2. **The model should not be the sole decision-maker.** Human review is required for borderline cases (P(default) in the [0.25, 0.45] range under the default threshold).

3. **Regulatory monitoring.** The State Bank of Vietnam is developing AI-specific guidance for credit scoring; operators should monitor updates to SBV Circulars 11/2021 and related regulations.

4. **Model retraining cadence.** Given the rapid evolution of Gen Z behavioral patterns and Vietnam's digital financial landscape, we recommend annual retraining with quarterly monitoring.

---

## References

Mitchell, M., et al. (2019). Model Cards for Model Reporting. *Proceedings of the ACM FAT* Conference.

SBV Circular 11/2021/TT-NHNN on credit risk classification.

Government of Vietnam, Decree 13/2023/ND-CP on personal data protection.

---

*Model Card Version 1.0 | 2024 | NEU Hybrid Credit Scoring Research Project*
