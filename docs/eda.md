# Exploratory Data Analysis (EDA)

## 1. Overview

This document summarizes the exploratory analysis of the Iranian Churn Dataset. The goal was to understand the distribution of customer churn, identify patterns in customer behavior, and examine relationships between the available features and churn.

The dataset contains 3,150 records and 14 columns. It has no missing values.

The target variable, `Churn`, is imbalanced:
- Retained customers (`0`): 2,655 (84.29%)
- Churned customers (`1`): 495 (15.71%)

Because churned customers represent a smaller portion of the dataset, accuracy alone is not sufficient for evaluating predictive models.

## 2. Customer Behavior

The following table compares the median values of selected customer behavior features between retained and churned customers.

| Feature | Retained | Churned |
|---|---:|---:|
| Seconds of Use | 3,530 | 1,182 |
| Frequency of use | 63 | 25 |
| Frequency of SMS | 25 | 9 |
| Distinct Called Numbers | 23 | 10 |
| Customer Value | 268.07 | 96.84 |

Across these features, churned customers generally have lower median usage and customer value than retained customers.

This suggests that reduced engagement with the service is associated with a higher likelihood of churn.

These differences describe associations in the dataset and do not establish that lower usage directly causes churn.

## 3. Categorical Feature Analysis

### 3.1 Complaints

| Complains | Churn rate |
|---|---:|
| 0 | 10.14% |
| 1 | 82.99% |

Customers who submitted complaints have a substantially higher churn rate than customers who did not.

This makes complaints one of the clearest categorical indicators of churn in the dataset.

### 3.2 Status

| Status | Churn rate |
|---|---:|
| 1 | 5.28% |
| 2 | 47.31% |

Status category 2 has a considerably higher churn rate than category 1.

The exact operational definition of `Status` is not fully documented, so its relationship with churn should be interpreted carefully.

The dataset documentation indicates that predictor information is aggregated from the first nine months and churn is measured at the end of month 12. The documented timing does not indicate that `Status` was measured after the target outcome.

### 3.3 Tariff Plan

| Tariff Plan | Churn rate |
|---|---:|
| 1 | 16.83% |
| 2 | 2.45% |

Customers on tariff plan 1 have a higher churn rate than customers on tariff plan 2.

However, the second tariff category represents a much smaller portion of the dataset, so the difference should be interpreted with that distribution in mind.

## 4. Age Analysis

The dataset includes both `Age` and `Age Group`.

The two variables map one-to-one in the dataset, meaning that `Age Group` is a categorical representation of the information already contained in `Age`.

The age-group churn rates were:

| Age Group | Churn rate |
|---|---:|
| 1 | 0.00% |
| 2 | 17.74% |
| 3 | 16.14% |
| 4 | 20.00% |
| 5 | 1.18% |

Age Group 4 has the highest observed churn rate, while Groups 1 and 5 have very low churn rates.

The extreme differences should be considered alongside the number of observations in each group. In particular, the oldest and youngest groups contain fewer records than the middle groups.

For modeling, `Age` was retained and `Age Group` was excluded because the two variables are redundant.

## 5. Correlation Analysis

Several features show strong correlations with one another.

| Feature pair | Correlation |
|---|---:|
| Age Group and Age | 0.961 |
| Seconds of Use and Frequency of use | 0.946 |
| Frequency of SMS and Customer Value | 0.925 |

The strong relationship between Age and Age Group confirms that they provide redundant information.

The usage-related features and Customer Value also share substantial information.

These relationships do not automatically mean that one of the correlated features must be removed. Tree-based models can handle correlated predictors, although correlated features can make individual feature importance values harder to interpret.

`Age Group` was excluded from modeling, while the other features were retained.

## 6. Distribution and Skewness

Several numeric features have positively skewed distributions.

| Feature | Skewness |
|---|---:|
| Frequency of SMS | 1.97 |
| Customer Value | 1.43 |
| Seconds of Use | 1.32 |
| Frequency of use | 1.14 |
| Distinct Called Numbers | 1.03 |

The distributions contain a concentration of lower values and a smaller number of higher observations.

The skew is especially noticeable among churned customers:
- Frequency of SMS: 4.88
- Customer Value: 2.76

These distributions help explain why medians were useful for comparing customer behavior instead of relying only on means.

## 7. Key Findings

The exploratory analysis highlighted several patterns:

1. **Complaints are strongly associated with churn.** Customers who submitted complaints have a much higher churn rate.
2. **Status is strongly associated with churn.** Category 2 has a considerably higher churn rate than category 1.
3. **Lower usage is associated with churn.** Churned customers have lower median usage across several service-related features.
4. **Customer Value differs between groups.** Churned customers have a lower median Customer Value.
5. **Some features contain overlapping information.** Age and Age Group are redundant, while several usage-related features are strongly correlated.
6. **The target is imbalanced.** Churned customers account for only 15.71% of the observations.

## 8. Modeling Considerations

The EDA findings informed the preparation and modeling decisions:

- `Age Group` was excluded, while `Age` was retained.
- Other correlated features were retained rather than removed solely because of correlation.
- All records, including duplicates, conflicting-label records, and zero-usage records, were preserved.
- The class imbalance was considered during model training and evaluation.
- Precision, recall, F1-score, ROC-AUC, and PR-AUC were used alongside accuracy to evaluate model performance.

The final modeling feature set contains 12 predictors, excluding `Age Group` and the target variable `Churn`.

## 9. Limitations

The analysis is based on the information available in the dataset. Several variable definitions remain incomplete, including the exact operational meaning of `Status` and the calculation of `Customer Value`.

The observed relationships should be interpreted as associations rather than causal explanations of customer churn.

For the detailed data-quality decisions, see [Data Dictionary and Initial Data Audit](data_dictionary.md).

For model performance and interpretation, see [Model Evaluation](model_evaluation.md).