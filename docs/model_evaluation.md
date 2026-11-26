# Model Evaluation

## 1. Overview

This document summarizes the training, evaluation, and interpretation of four machine learning models developed to predict customer churn.

The models were trained using the Iranian Churn Dataset:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

The objective was to compare their ability to identify customers who churned while also considering the number of retained customers incorrectly classified as churned.

Random Forest was selected as the primary model based on its overall performance across the evaluation metrics.

## 2. Data Preparation and Train/Test Split

The final modeling dataset contains 12 predictor features. `Age Group` was excluded because it provides redundant information already represented by `Age`.

The target variable is `Churn`, where:
- `0` represents retained customers.
- `1` represents churned customers.

### Train/Test Split

A stratified group-based split was used to reduce the possibility of identical predictor combinations appearing in both the training and test sets.

The split used `StratifiedGroupKFold` with five folds, shuffling enabled, and a random state of 42. One fold was held out for testing.

| Dataset | Records |
|---|---:|
| Training | 2,517 |
| Testing | 633 |
| Total | 3,150 |

The churn distribution remained similar across both sets.

| Dataset | Retained | Churned |
|---|---:|---:|
| Training | 84.31% | 15.69% |
| Testing | 84.20% | 15.80% |

No identical predictor groups were shared between the training and test sets.

### Class Imbalance

The training data contains substantially more retained customers than churned customers.

To account for this imbalance, the positive-class weight was calculated using the training data:

`scale_pos_weight = 5.3722`

This value represents the ratio of retained to churned observations in the training set and was used for the applicable boosting model configurations.

## 3. Model Performance Comparison

The models were evaluated using the default classification threshold.

| Metric | Logistic Regression | Random Forest | XGBoost | LightGBM |
|---|---:|---:|---:|---:|
| Accuracy | 84.36% | **95.89%** | 95.10% | 95.58% |
| Precision | 50.28% | **82.46%** | 78.05% | 80.00% |
| Recall | 90.00% | 94.00% | **96.00%** | **96.00%** |
| F1-score | 64.52% | **87.85%** | 86.10% | 87.27% |
| ROC-AUC | 94.34% | **99.32%** | 99.10% | 99.13% |
| PR-AUC | 79.38% | **96.17%** | 95.30% | 95.43% |

### Interpretation

**Logistic Regression**

Logistic Regression achieved 84.36% accuracy and 90% recall. However, its precision was considerably lower than the other models at 50.28%.

This means it identified many churned customers but also incorrectly classified a relatively large number of retained customers as churned.

**Random Forest**

Random Forest achieved the highest accuracy, precision, F1-score, ROC-AUC, and PR-AUC.

It also achieved 94% recall, identifying most of the churned customers while producing fewer false positives than the other models.

**XGBoost**

XGBoost achieved the highest recall alongside LightGBM at 96%.

It identified more churned customers than Random Forest but produced more false positives and had slightly lower overall performance across the other metrics.

**LightGBM**

LightGBM also achieved 96% recall and performed similarly to XGBoost.

Its precision and F1-score were slightly higher than XGBoost, but it did not outperform Random Forest overall.

## 4. Confusion Matrix Analysis

The confusion matrices provide a closer look at how each model classified retained and churned customers.

| Model | True Negatives | False Positives | False Negatives | True Positives |
|---|---:|---:|---:|---:|
| Logistic Regression | 444 | 89 | 10 | 90 |
| Random Forest | 513 | 20 | 6 | 94 |
| XGBoost | 506 | 27 | 4 | 96 |
| LightGBM | 509 | 24 | 4 | 96 |

Where:
- **True Negative:** A retained customer correctly classified as retained.
- **False Positive:** A retained customer incorrectly classified as churned.
- **False Negative:** A churned customer incorrectly classified as retained.
- **True Positive:** A churned customer correctly classified as churned.

### Business Interpretation

A false negative represents a churned customer the model failed to identify. A false positive represents a retained customer incorrectly flagged as likely to churn.

If the business priority is identifying as many potential churners as possible, XGBoost and LightGBM have an advantage because they produced fewer false negatives.

If the priority is balancing churn detection with fewer unnecessary retention interventions, Random Forest performed better.

Random Forest produced 20 false positives and 6 false negatives, compared with 27 false positives and 4 false negatives for XGBoost.

## 5. Random Forest Feature Importance

Random Forest was selected as the primary model for further interpretation.

Its built-in feature importance scores were:

| Feature | Importance |
|---|---:|
| Status | 0.1637 |
| Frequency of use | 0.1307 |
| Customer Value | 0.1278 |
| Seconds of Use | 0.1234 |
| Complains | 0.1121 |
| Subscription Length | 0.0991 |
| Distinct Called Numbers | 0.0688 |
| Frequency of SMS | 0.0665 |
| Call Failure | 0.0442 |
| Age | 0.0403 |
| Charge Amount | 0.0218 |
| Tariff Plan | 0.0016 |

`Status` had the highest importance score, followed by Frequency of use, Customer Value, Seconds of Use, and Complains.

These values represent the relative contribution of features to the model's tree-based splits. They should not be interpreted as the percentage of churn explained by each feature.

Impurity-based importance can favor features with many possible values, and correlated features can share importance across one another.

For these reasons, feature importance was supplemented with SHAP analysis.

## 6. SHAP Model Interpretation

SHAP was used to examine how individual features influenced Random Forest predictions.

Unlike the built-in feature importance scores, SHAP provides information about both the magnitude and direction of feature contributions to model predictions.

### 6.1 Global Feature Importance

The SHAP global importance results identified the following features as particularly influential:

- Status
- Complains
- Frequency of use
- Customer Value
- Seconds of Use
- Subscription Length

These findings broadly support the patterns observed during exploratory data analysis.

### 6.2 SHAP Summary Findings

The SHAP summary plot showed several consistent patterns:

- Status category 2 generally pushes predictions toward churn, while category 1 pushes predictions away from churn.
- Customers with complaints generally have positive SHAP values, pushing predictions toward churn.
- Higher service usage and customer value generally push predictions away from churn.
- Lower usage and lower customer value generally contribute more positively toward churn predictions.

These patterns describe how the trained model uses the available features. They do not establish causal relationships.

### 6.3 Feature Dependence Findings

The dependence plots provided additional detail about individual feature relationships with model predictions.

**Complains**

Customers with a complaint value of 1 generally had positive SHAP values, while customers with a value of 0 generally had negative SHAP values.

**Status**

Status category 2 generally pushed predictions toward churn, while category 1 pushed predictions away from churn.

**Frequency of Use**

Lower usage values were generally associated with positive SHAP values. As usage increased, SHAP values tended to decrease and eventually plateau.

**Customer Value**

The pattern was similar to Frequency of Use. Lower customer value generally contributed toward churn predictions, while higher values tended to push predictions away from churn.

**Subscription Length**

The relationship was nonlinear. SHAP values varied across subscription lengths rather than following a consistent increasing or decreasing pattern.

**Call Failure**

SHAP values generally increased as call failures increased, indicating a stronger contribution toward churn predictions at higher values.

**Charge Amount**

Higher charge categories generally had more negative SHAP values, although the overall contribution was relatively small.

**Seconds of Use and Frequency of SMS**

Lower usage values generally pushed predictions toward churn. Higher values tended to contribute away from churn, with the effect becoming less pronounced at higher usage levels.

**Distinct Called Numbers**

The feature generally showed a modest relationship with predictions. Lower values tended to contribute slightly toward churn, while higher values generally contributed away from churn.

**Tariff Plan**

SHAP values were concentrated near zero, suggesting that this feature had a relatively small contribution to individual predictions.

**Age**

Age had a less consistent relationship with model predictions than features such as Status and Complains. Its SHAP values varied across age ranges, indicating a nonlinear relationship.

### 6.4 SHAP Limitations

SHAP explains how the model uses features to produce predictions. It does not prove that a feature causes a customer to churn.

Correlated features can also influence one another's explanations, making individual contributions more difficult to interpret independently.

## 7. Model Selection

Random Forest was selected as the primary model because it achieved the strongest overall results across the evaluation metrics.

It had:
- The highest accuracy at 95.89%.
- The highest precision at 82.46%.
- The highest F1-score at 87.85%.
- The highest ROC-AUC at 99.32%.
- The highest PR-AUC at 96.17%.
- Only 20 false positives and 6 false negatives.

However, model selection depends on the business objective.

If minimizing missed churn cases is the highest priority, XGBoost and LightGBM may be preferable because they achieved 96% recall.

For this project, Random Forest provides the strongest overall balance between identifying churned customers and limiting false positives.

## 8. Limitations and Further Improvements

Although the models achieved strong results, several limitations remain.

**Single train/test split**

The reported results come from one held-out test split. Additional validation, such as repeated group-based cross-validation, would help determine whether performance is consistent across different data partitions.

**Threshold selection**

All models were evaluated using the default classification threshold. Adjusting the threshold could change the balance between precision and recall.

A threshold should ultimately reflect the cost of missed churn cases compared with unnecessary retention interventions.

**Dataset limitations**

The dataset does not contain a customer identifier, and some identical predictor combinations have conflicting churn labels. The exact operational definitions of certain features are also unclear.

**Model interpretation**

Feature importance and SHAP describe model behavior rather than causal relationships. Further business context would be needed before using these findings to design retention strategies.

## 9. Conclusion

Four classification models were trained and evaluated to predict customer churn.

Random Forest achieved the strongest overall performance, with 95.89% accuracy, 82.46% precision, 94% recall, and an F1-score of 87.85%.

The results suggest that Status, Complains, customer usage, and Customer Value are important factors in the model's predictions.

SHAP analysis provided additional insight into how these features influenced the model's decisions.

Overall, the project demonstrates a complete analytical workflow, from initial data auditing and exploratory analysis to predictive modeling and model interpretation. The results provide a useful starting point for identifying potential churn risks, while further validation would be necessary before applying the models in a real-world setting.