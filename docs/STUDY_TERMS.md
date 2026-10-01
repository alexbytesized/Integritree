# Integritree terms and concepts

Study companion, checked on **2026-10-01**. Return to the
[study index](STUDY_GUIDE.md) or continue to the [code guide](STUDY_CODE_GUIDE.md).
Definitions below describe this project's implementation. Numerical teaching
examples are explicitly identified and are not experimental findings.

## 1. Data and research terms

| Term | Plain-language explanation | Meaning in Integritree |
| --- | --- | --- |
| E-wallet | An application/service used to hold or transfer electronic value. | The application demonstrates selected GCash screenshot workflows. |
| PaySim | A simulator associated with synthetic mobile-money transaction data. | The supplied CSV is the research dataset; it is not a real GCash customer database. |
| Synthetic dataset | Artificially generated records that represent a scenario. | PaySim records already are synthetic before SMOTE is used. |
| Transaction / record / row | One observation in the dataset. | A transaction has source attributes, an identity, and, for evaluation, a fraud label. |
| Ground truth / target / label | The reference answer used for training or checking predictions. | `isFraud`: 1 means labeled fraud; 0 means labeled legitimate. |
| Predictor | Information given to a model to produce a prediction. | The final model input contains eleven engineered features, never `isFraud`. |
| Feature engineering | Calculating useful model inputs from source attributes. | For example, converting amount into `log_amount` and `is_zero_amount`. |
| Class imbalance | One label occurs much less often than the other. | Only about 0.129082% of the supplied records are labeled fraud. |
| Minority / majority class | The less/more frequent label. | Fraud is the minority; legitimate is the majority. |
| Training set | Records used to learn preprocessing values and model behavior. | 80% of the cleaned source records. |
| Validation set | Separate records used to choose settings. | 10%; used in all three selection stages. |
| Held-out test set | Records reserved to assess the finalized procedure. | Remaining 10%; official evaluation is pending. |
| Stratification | Splitting while approximately preserving label proportions. | Both split operations stratify on `isFraud`. |
| Data leakage | Allowing information from evaluation data to influence fitting or selection improperly. | Fitting a scaler on all records or applying SMOTE before splitting would violate the research procedure. |
| Preprocessing | Preparing inputs into the model's expected representation. | Includes allowed amount imputation, feature engineering, and saved scaling. |
| Imputation | Filling a permitted missing value using an established rule. | A missing amount uses the original training median. The target is never imputed. |
| One-hot indicators | Separate numeric columns indicating category membership. | Five `type_*` columns; known real records have one active type. Missing type has all five zero. |
| Normalization / Min-Max scaling | Expressing numeric values using a scale learned from training extrema. | Applied only to log amount, hour, and day; future out-of-range values are not clipped. |
| Exact duplicate | A repeated original record under the defined comparison. | All eleven typed raw columns are compared; zero exact duplicates were found. |
| Seed | A value controlling a pseudorandom procedure. | Split, model, and SMOTE seeds are 42; seeds are not searched for favorable outcomes. |
| Provenance | The record of where data/results came from and how they were produced. | File hashes, split identity, configuration, versions, model run, and mapping revision. |
| SHA-256 / fingerprint | A digest used to check file or input identity. | Detects changed inputs/artifacts; it does not anonymize a retained receipt or prove its authenticity. |

### Transaction categories

| Category | Basic interpretation | Current receipt demonstration |
| --- | --- | --- |
| `TRANSFER` | Transfer between accounts. | Express Send initially selects this category. |
| `PAYMENT` | Payment for goods/services. | Pay Online initially selects this category; merchant QR recognition remains pending. |
| `DEBIT` | A debit transaction category in PaySim. | Bank Transfer initially selects this category as a demonstration mapping. |
| `CASH_IN` | Funds enter a wallet/account through a cash-in operation. | Selectable in confirmation, but Cash In screenshot recognition is pending. |
| `CASH_OUT` | Funds leave through a cash-out operation. | Selectable in confirmation, but Cash Out screenshot recognition is pending. |

The receipt mappings are assumptions for this demonstration. A PaySim category
name is not proof of equivalent real-world behavior. Original fraud-positive
examples occur only in TRANSFER and CASH_OUT, although all five types remain in
the research dataset and models.

## 2. Models, SMOTE, and scoring

| Term | Explanation in this project |
| --- | --- |
| Decision tree | A model that follows feature-based branches to a leaf and produces class probabilities. |
| Random Forest (RF) | An ensemble of decision trees. Its fraud score averages the trees' fraud-class probability estimates. |
| Ensemble | Multiple models combined into one prediction procedure; here, the individual trees form a forest. |
| Benchmark / control | The comparison model trained on the original imbalanced training data. It is named `rf` in code. |
| RF-SMOTE / experimental model | The same classifier family/settings trained on SMOTE-augmented training data. It is named `rf_smote`. |
| SMOTE | Synthetic Minority Over-sampling Technique: adds minority training vectors by interpolation between minority neighbors. |
| Sampling ratio | Desired minority-to-majority ratio after resampling. Here 1:100 means approximately one fraud training vector for every 100 legitimate ones, not one fraud vector per 100 total records. |
| `k_neighbors` | Number of minority neighbors available to SMOTE's interpolation procedure; configured as 5. At least 6 minority examples are required. |
| Hyperparameter | A setting chosen outside the model's fitting process, such as tree count or depth. |
| `n_estimators` | Number of trees; the selected forests use 100. |
| `max_depth` | Maximum allowed depth of each tree; selected value 10. |
| `min_samples_leaf` | Minimum training samples in a leaf; selected value 1. |
| Bootstrap | Sampling training rows with replacement for individual trees; enabled here. |
| Gini impurity | A measure of class mixture used to choose tree splits in this implementation. |
| `max_features="sqrt"` | Limits the candidate features considered at each split according to the square-root setting. |
| Training / fitting | Learning from the training examples. Uploading a CSV or receipt does not train the models. |
| Inference | Applying already fitted models and preprocessing to inputs. |
| Fraud score | The class-1 result from `predict_proba`, in the range 0 to 1. It is a model output, not an established real-world fraud probability. |
| Decision threshold / cutoff | The score boundary for a binary prediction; currently 0.43 for both models. |
| Calibration | Whether predicted probabilities agree with observed event frequencies in a relevant population. Real GCash calibration has not been established. |
| Model bundle | Saved models plus the preprocessing, feature schema, settings, and evidence needed to load them consistently. |

Ordinary SMOTE can produce fractional values in encoded category indicators.
Integritree preserves those numeric values rather than rounding them or pretending
that each synthetic vector is a realistic transaction receipt. SMOTE does not
create new independent evidence of actual fraud.

### Score, class, and risk band are different

The application displays `100 * score`. The binary decision uses the unrounded
score: `score >= 0.43` means Predicted Fraud, including equality.

| Displayed score range | Communication band |
| --- | --- |
| 0 to less than 20 | Minimal Risk |
| 20 to less than 40 | Low Risk |
| 40 to less than 60 | Moderate Risk |
| 60 to less than 80 | High Risk |
| 80 through 100 | Critical Risk |

**Teaching example:** a score of 0.45 displays as 45.00 and falls in Moderate
Risk, while its binary prediction is Fraud because 0.45 >= 0.43. There is no
contradiction: the band communicates score magnitude; the selected cutoff
determines the binary class.

## 3. Evaluation terms and formulas

Fraud is the positive class. A prediction alone cannot tell us whether it is correct;
we need an independent actual label.

| Actual label | Predicted legitimate | Predicted fraud |
| --- | --- | --- |
| Legitimate | True negative (TN): correctly cleared | False positive (FP): false alarm |
| Fraud | False negative (FN): missed fraud | True positive (TP): detected fraud |

This is the **confusion matrix**. The backend stores it as `[[TN, FP], [FN, TP]]`,
with actual labels on rows and predictions on columns.

| Metric | Formula / meaning | Question it answers |
| --- | --- | --- |
| Precision | `TP / (TP + FP)` | Of flagged transactions, how many were labeled fraud? |
| Recall / sensitivity | `TP / (TP + FN)` | Of labeled fraud, how much did the model detect? |
| F1 score | `2TP / (2TP + FP + FN)` | How do precision and recall balance? |
| Accuracy | `(TP + TN) / N` | What fraction of all predictions was correct? |
| MCC | `(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))` | How strongly do predictions and labels agree, considering all four cells? |
| PR curve | Precision versus recall as a score cutoff changes. | What trade-offs are available across cutoffs? |
| PR-AUC (Average Precision / AP) | Recall-increment-weighted precision across score thresholds. | How well does the score ranking retrieve fraud over different cutoffs? |

AP is the project's chosen PR-AUC convention; it is not trapezoidal interpolation.
It requires continuous scores and labels, not just one confusion matrix. AP does
not change merely because the displayed binary cutoff changes while scores stay
the same. The `pr_auc` result key specifically represents Average Precision.

MCC ranges from -1 to +1 when defined: positive values indicate agreement,
negative values indicate inverse association, and 0 indicates no measured
correlation. It is a coefficient, not an accuracy percentage.

**Teaching example, not a result:** suppose 100 records contain 10 fraud, with
TP=6, FN=4, FP=3, and TN=87. Precision is 6/9 = 66.67%; recall is 6/10 = 60%;
F1 is 12/19 = 63.16%; accuracy is 93%. The high accuracy should not distract from
the four missed fraud records. AP cannot be calculated from these counts alone.

### Undefined values

The tool reports unavailable values with a reason rather than disguising them as
ordinary zeros. Examples: no predicted positives makes precision unavailable;
no actual positives makes recall and AP unavailable; a zero MCC denominator makes
MCC unavailable. Count-based F1 can still be a valid zero if its denominator is
positive. If every actual label is positive, AP is 1 under the implemented convention.

### Comparing two models

**Signed symmetric percentage difference** is descriptive:

```text
B = benchmark RF metric
S = RF-SMOTE metric
Signed difference (%) = 100 * (S - B) / ((S + B) / 2)
```

Positive means RF-SMOTE has the higher metric; negative means RF has the higher
metric. Both zero or undefined inputs yield N/A. If either MCC is negative, the
comparison uses `S - B` in coefficient units instead. This calculation is not a
significance test and is not the same as percentage change relative to RF alone.

**McNemar's test** compares the two models' paired classification errors on the
same records. Let `b` count RF-correct/RF-SMOTE-incorrect cases and `c` count
RF-incorrect/RF-SMOTE-correct cases. These are **discordant pairs**.

The primary statistic is `(abs(b-c)-1)^2 / (b+c)` with one chi-square degree of
freedom. For 1–24 discordances, an exact two-sided binomial test is supplementary.
For zero discordances, the implementation reports p=1 by convention and no statistic.

- **Null hypothesis:** equal classification error rates under the test's assumptions.
- **Alpha / significance level:** 0.05, the declared comparison boundary.
- **p-value:** how unusual the observed disagreement imbalance would be under
  the null and test assumptions. It is not the probability that the null is true.
- **Decision:** p < 0.05 means reject the null; otherwise fail to reject it.
- **Direction:** `c > b` means RF-SMOTE made fewer observed errors; `b > c` means RF did.

McNemar does not separately establish significance for precision, recall, F1, MCC,
or AP. A nonsignificant result does not prove equivalence.

## 4. Explainability terms

| Term | Meaning |
| --- | --- |
| SHAP | Shapley Additive Explanations: feature contributions explaining a model output relative to a reference baseline. |
| TreeSHAP | A SHAP method for tree models, used here to explain fraud probability. |
| Interventional explanation | The configured feature-perturbation approach using a background dataset. The word does not make the result a causal claim. |
| Background sample | The reference inputs used to define the explanation baseline: 200 uniformly sampled original-training records, without replacement, seed 42. |
| Base value | The model's reference output for the SHAP explanation, not its classification threshold. |
| SHAP contribution | A feature's signed movement from the baseline toward the explained output. Positive increases the fraud score; negative decreases it. |
| Additivity / reconstruction | `base value + sum(contributions)` should reproduce the model score; the checked tolerance is 1e-6. |
| Waterfall chart | A visualization of the baseline and signed contributions leading to the final score. |
| Top positive contributor | The strongest risk-increasing contribution above the 1e-9 tolerance; it can exist even for a legitimate prediction. |
| Local explanation | Explanation of one transaction. |
| Global summary | Summary over multiple explanation targets; the report policy uses a separate uniform sample of 1,000 evaluation records. |
| Explanation coverage | Which records actually have completed explanations. It may be smaller than the full evaluated population. |

**Teaching example:** baseline 0.10 plus contributions 0.25, 0.15, and -0.05
gives score 0.45. A negative contribution lowered this model's score; it does not
prove the feature prevents fraud in reality. Actual Integritree explanations
contain the eleven model features.

The researcher interface requests SHAP for displayed/opened records. It does not
automatically explain an entire large CSV. The receipt flow attempts explanations
for both models after prediction; failure is reported and can be retried.

## 5. Receipt and software terms

| Term | Meaning in the system |
| --- | --- |
| OCR | Optical Character Recognition: reads text from the uploaded image. |
| RapidOCR | Provisionally selected local extraction engine; runs in a separate Python environment/process. |
| Parser | Interprets OCR text as candidate fields such as amount, reference, and time. |
| Layout recognition | Checks multiple expected screen anchors and supported workflow conditions before confirmation. |
| Candidate field | An extracted suggestion; missing/incorrect OCR must be completed or corrected by the user. |
| Confirmation | The user accepts editable fields through Proceed; required inputs are validated. |
| Principal | Transaction amount excluding fees, expressed numerically in PHP for the demonstration. |
| Mapping v2 | `gcash_confirmed_v2`: transforms confirmed amount, time, category, and Client/Merchant roles into the eleven model features. |
| Revision | A version of the confirmed receipt inputs. Editing and reconfirming retires previous derived results. |
| Frontend | React interface running in the browser. |
| Backend | Python services, API, model loading, processing, and temporary storage. |
| API / endpoint | The request interface connecting browser and backend, such as `POST /api/v1/research/analyses`. |
| Route | A URL mapping; React routes select pages, while FastAPI routes select server handlers. |
| Service | Backend code that coordinates a complete workflow beyond a single HTTP request. |
| Schema / contract | Explicit definitions and validation rules for data structure and values. |
| HTTP / JSON | Request-response transport / structured data representation used by the API. Upload bodies themselves contain CSV or image bytes. |
| Asynchronous job | Work accepted now and completed later. HTTP 202 does not mean the analysis is already finished. |
| Polling | Repeatedly requesting job status until it changes or completes. |
| WebSocket presence | A persistent receipt-session connection used to detect active browser connections and schedule cleanup. |
| Session token | An opaque value tying temporary application data to the browser session. It is not a user-account login. |
| `sessionStorage` | Browser storage that preserves tokens/drafts across refresh within a tab session. |
| SQLite | Local database used for researcher records, queries, metrics, and export processing. |
| Parquet | Column-oriented file format used for prepared data and research evidence. |
| Artifact / report | Saved model/preprocessing/evidence files / research outputs. These differ from temporary browser uploads. |
| Lockfile | Records dependency versions; separate from a runtime lock that prevents competing processes from writing the same workspace. |
| Regression test | Checks that a change has not broken established behavior. |
| Integration / end-to-end test | Checks collaborating components / a user flow across the browser and backend. |
| Smoke test | A limited check that an important flow runs successfully; not an accuracy study. |

## 6. Practice questions

1. Why are PaySim synthetic records and SMOTE synthetic vectors different concepts?
2. Why must `isFraud` stay outside the feature matrix?
3. What is the difference between a fraud score, its risk band, and its predicted class?
4. Why can accuracy be misleading with very rare fraud?
5. What additional information does AP need beyond a confusion matrix?
6. What does McNemar test, and what does it not test?
7. Can a legitimate prediction have a positive SHAP contributor? Explain why.
8. Why does reading a receipt correctly fail to prove real-world fraud accuracy?

Implementation references: [features](../backend/src/integritree/ml/features.py),
[inference](../backend/src/integritree/ml/inference.py),
[evaluation](../backend/src/integritree/ml/evaluation.py),
[SHAP](../backend/src/integritree/ml/explainability.py),
[receipt mapping](RECEIPT_MAPPING.md), and [researcher workflow](RESEARCHER_WORKFLOW.md).
