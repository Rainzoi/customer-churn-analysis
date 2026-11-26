# Data Dictionary & Initial Data Audit

## 1. Project Overview

This project uses the **Iranian Churn Dataset** from the UCI Machine Learning Repository to explore customer churn and build prediction models.

The dataset includes customer information, service usage, complaints, and churn outcomes. Before starting the analysis, I looked through the data to understand its structure, check for possible issues, and decide how to handle them.

**Dataset:** [UCI Iranian Churn Dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)

**DOI:** [https://doi.org/10.24432/C5JW3Z](https://doi.org/10.24432/C5JW3Z)

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

**Local file:** `data/raw/Customer Churn.csv`

The original CSV is kept unchanged. Any data preparation is done separately.

## 2. Dataset Overview

| Property | Value |
|---|---:|
| Rows | 3,150 |
| Columns | 14 |
| Integer columns | 13 |
| Float columns | 1 |
| Missing values | 0 |
| Additional exact duplicate occurrences | 300 |
| Churned records | 495 |
| Retained records | 2,655 |

The target variable is `Churn`:

| Value | Meaning | Count | Percentage |
|---|---|---:|---:|
| 0 | Retained | 2,655 | 84.29% |
| 1 | Churned | 495 | 15.71% |

There are considerably more retained records than churned records, so the target variable is imbalanced.

## 3. Data Dictionary

The original column names are kept, including their spacing.

| Column | Description |
|---|---|
| `Call  Failure` | Number of call failures |
| `Complains` | Whether the customer submitted a complaint (0 or 1) |
| `Subscription  Length` | Length of the customer's subscription |
| `Charge  Amount` | Customer charge category |
| `Seconds of Use` | Total service usage measured in seconds |
| `Frequency of use` | Frequency of service usage |
| `Frequency of SMS` | Frequency of SMS usage |
| `Distinct Called Numbers` | Number of different numbers called |
| `Age Group` | Encoded customer age category |
| `Tariff Plan` | Customer tariff plan |
| `Status` | Customer status category |
| `Age` | Customer age |
| `Customer Value` | Calculated customer value |
| `Churn` | Indicates whether the customer churned |

### Categorical Variables

**Complains**
- `0`: No complaint
- `1`: Complaint

**Age Group**
- `1`: Youngest age category
- `2`: Second age category
- `3`: Middle age category
- `4`: Fourth age category
- `5`: Oldest age category

**Tariff Plan**
- `1`: Pay as you go
- `2`: Contractual

**Status**
- `1`: Active
- `2`: Non-active

The exact meaning of `Status` is not fully explained in the dataset documentation.

**Churn**
- `0`: Customer retained
- `1`: Customer churned

### Charge Amount

The dataset documentation describes `Charge Amount` as having values from 0 to 9. However, the downloaded CSV contains seven records with a value of 10.

I kept these records because there wasn't enough information to determine whether they were incorrect.

## 4. Initial Data Audit

### 4.1 Missing Values

There are no missing values in the dataset, so no missing-value handling was needed.

### 4.2 Duplicate Records

The initial audit found 300 additional exact duplicate occurrences. In total, 465 rows belong to groups of repeated records.

Since the dataset doesn't include a customer ID, it's impossible to know whether these records belong to the same customers or different customers with identical information.

**Decision:** Keep all records. Removing them could mean losing valid observations, especially since there's no way to confirm that they represent duplicate customers.

### 4.3 Conflicting Churn Labels

I found 14 combinations of identical predictor values that have different churn labels. These combinations affect 64 records.

- 11 combinations involve records with zero usage across all four usage-related features.
- 3 combinations involve records with some recorded usage.

In other words, two records can have the same recorded predictor values but different churn outcomes.

There's not enough information to determine why this happens or which label is correct.

**Decision:** Keep all original records and labels. I didn't remove records or change any churn labels.

### 4.4 Zero-Usage Records

A total of 132 records have zero values across all four usage-related features:

- `Seconds of Use`
- `Frequency of use`
- `Frequency of SMS`
- `Distinct Called Numbers`

| Churn outcome | Count |
|---|---:|
| Retained | 63 |
| Churned | 69 |
| Total | 132 |

The churn rate among these records is 52.27%, compared with 14.12% for records with some recorded usage.

**Decision:** Keep these records. Zero usage appears to be an important customer behavior pattern and may help the models identify customers who are more likely to churn.

### 4.5 Status

The `Status` variable has two categories:

| Status | Count | Churn rate |
|---|---:|---:|
| 1 | 2,368 | 5.28% |
| 2 | 782 | 47.31% |

Customers in category 2 have a much higher churn rate.

The UCI documentation says that predictor information comes from the first nine months, while churn is measured at the end of month 12. Based on this timing, there is no indication that `Status` was recorded after the churn outcome.

However, its exact meaning is still unclear.

**Decision:** Keep `Status` as a modeling feature.

### 4.6 Age and Age Group

`Age` and `Age Group` match consistently throughout the dataset:

| Age | Age Group | Record count |
|---:|---:|---:|
| 15 | 1 | 123 |
| 25 | 2 | 1,037 |
| 30 | 3 | 1,425 |
| 45 | 4 | 395 |
| 55 | 5 | 170 |

Since both variables contain the same information, there's no need to use both in the model.

**Decision:** Keep `Age` and leave `Age Group` out of the modeling features. The original CSV remains unchanged.

## 5. Final Data Handling Decisions

| Issue | Decision |
|---|---|
| Missing values | No action needed |
| Exact duplicates | Keep all records |
| Conflicting churn labels | Keep original records and labels |
| Zero-usage records | Keep |
| `Charge Amount` = 10 | Keep all seven records |
| Age and Age Group | Keep `Age`; exclude `Age Group` from modeling |
| Status | Keep as a predictor |
| Original CSV | Leave unchanged |

These decisions were made before modeling so that the original data could be preserved without making assumptions about records that might be valid.

## 6. Remaining Limitations

There are still a few things about the dataset that aren't fully clear:

- There is no customer ID, so repeated records can't be linked to individual customers.
- The reason for conflicting churn labels is unknown.
- The seven `Charge Amount` values of 10 aren't explained in the documentation.
- The exact meaning of `Status` is unclear.
- The formula used to calculate `Customer Value` isn't provided.

These are worth keeping in mind when looking at the analysis and model results.

## 7. Source

Hosseini, S. M. S., et al. *Iranian Churn Dataset*. UCI Machine Learning Repository.

[https://doi.org/10.24432/C5JW3Z](https://doi.org/10.24432/C5JW3Z)

License: Creative Commons Attribution 4.0 International (CC BY 4.0).