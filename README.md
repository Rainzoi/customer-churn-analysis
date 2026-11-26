# Customer Churn Analysis & Prediction

## Project Overview

This project analyzes customer churn using the **Iranian Churn Dataset** from the UCI Machine Learning Repository.

The goal is to understand which customer behaviors are associated with churn and build machine learning models that can identify customers who are more likely to leave.

The project follows a data analysis workflow that includes data auditing, exploratory data analysis, data preparation, model training, evaluation, and model interpretation.

## Business Question

Can customer behavior and account information be used to identify customers who are likely to churn, and which factors are most strongly associated with that decision?

## Dataset

The project uses the [Iranian Churn Dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset) from the UCI Machine Learning Repository.

- **Records:** 3,150
- **Columns:** 14
- **Target variable:** `Churn`
- **Churn distribution:** 15.71% churned and 84.29% retained
- **Missing values:** None

The dataset contains customer account information, service usage, complaints, and churn outcomes. According to the dataset documentation, predictor information is aggregated from the first nine months, while churn is measured at the end of month 12.

The original CSV is stored locally at `data/raw/Customer Churn.csv` and is not included in this repository.

## Tools & Technologies

- **Python** — data analysis and modeling
- **pandas & NumPy** — data manipulation
- **scikit-learn** — data preparation, model training, and evaluation
- **XGBoost & LightGBM** — gradient boosting models
- **SHAP** — model interpretation
- **Matplotlib & Seaborn** — data visualization

## Project Workflow

1. **Data Audit:** Reviewed dataset structure, data types, missing values, duplicates, and potential inconsistencies.
2. **Data Preparation:** Prepared modeling features and created a stratified group-based train/test split.
3. **Exploratory Data Analysis:** Examined churn distribution, customer behavior, categorical features, correlations, and feature distributions.
4. **Model Training:** Trained Logistic Regression, Random Forest, XGBoost, and LightGBM models.
5. **Model Evaluation:** Compared model performance using accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC.
6. **Model Interpretation:** Used feature importance and SHAP to examine which features influenced model predictions.

## Model Results

Random Forest was selected as the primary model based on its overall performance across the evaluation metrics.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 84.36% | 50.28% | 90.00% | 64.52% |
| Random Forest | 95.89% | 82.46% | 94.00% | 87.85% |
| XGBoost | 95.10% | 78.05% | 96.00% | 86.10% |
| LightGBM | 95.58% | 80.00% | 96.00% | 87.27% |

Random Forest achieved the strongest overall balance of performance, while XGBoost and LightGBM identified slightly more churn cases.

See [Model Evaluation](docs/model_evaluation.md) for the full comparison and interpretation.

## Project Documentation

- [Data Dictionary & Initial Audit](docs/data_dictionary.md)
- [Exploratory Data Analysis](docs/eda.md)
- [Model Evaluation](docs/model_evaluation.md)

## Running the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd churn
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Download the Iranian Churn Dataset from the UCI Machine Learning Repository and place the CSV file at:

```text
data/raw/Customer Churn.csv
```

### 5. Run the scripts

Run the scripts from the project root directory:

```powershell
python src/data_audit.py
python src/data_preperation.py
python src/model_training.py
```

## Key Findings

- The dataset has a churn rate of 15.71%, resulting in an imbalanced target.
- Customer complaints and account status show strong associations with churn.
- Lower service usage and lower customer value are generally associated with churn.
- Random Forest produced the strongest overall evaluation results among the four models tested.
- SHAP analysis provided additional insight into how individual features influenced model predictions.

## Limitations

- The dataset does not contain a customer identifier.
- Some records have identical predictor values but different churn outcomes.
- The exact operational definitions of certain variables, including `Status` and `Customer Value`, are not fully documented.
- Model performance is based on one held-out test split and should be validated further before real-world deployment.
- Feature importance and SHAP explain model behavior; they do not establish causation.

## Data Source & License

Hosseini, S. M. S., & colleagues. *Iranian Churn Dataset*. UCI Machine Learning Repository.

https://doi.org/10.24432/C5JW3Z

The dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. Please refer to the original source for attribution and licensing details.