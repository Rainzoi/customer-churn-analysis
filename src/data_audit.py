import pandas as pd


# Path to the original dataset
DATA_PATH = "data/raw/Customer Churn.csv"


df = pd.read_csv(DATA_PATH)


# ==================================================
# 1. DATASET OVERVIEW
# ==================================================


print("\n=== DATASET OVERVIEW ===")

print("\n--- First Five Rows ---")
print(df.head())

row_count, column_count = df.shape

print("\n--- Dataset Dimensions ---")
print(f"Rows: {row_count}")
print(f"Columns: {column_count}")

print("\n--- Summary Statistics ---")
print(df.describe().to_string())

print("\n--- Dataset Information ---")
df.info()

print("\n--- Unique Values Per Column ---")
print(df.nunique().to_string())


# ==================================================
# 2. MISSING VALUES AND DATA TYPES
# ==================================================


print("\n=== DATA QUALITY: MISSING VALUES ===")

missing_count = df.isna().sum().sum()

print(f"Total missing values: {missing_count}")


# ==================================================
# 3. CHURN DISTRIBUTION AND CONFLICTING LABELS
# ==================================================


print("\n=== CHURN ANALYSIS ===")

churn_counts = df["Churn"].value_counts().sort_index()
churn_percentages = (df['Churn'].value_counts(normalize=True).sort_index()) * 100

churn_summary = pd.DataFrame({
    "Count": churn_counts,
    "Percentage": churn_percentages.round(2)
})

print("\n--- Churn Distribution ---")
print(churn_summary.to_string())


print("\n--- Identical Features With Conflicting Churn ---")

feature_columns = df.columns.drop("Churn").tolist()

churn_by_features = (
    df.groupby(feature_columns)["Churn"]
    .value_counts()
    .unstack()
)

conflicting_groups = churn_by_features[
    (churn_by_features > 0).sum(axis=1) > 1
]
print(
    "Conflicting feature combinations:",
    len(conflicting_groups)
)

print("\n--- Conflicting Churn Details ---")
print(conflicting_groups.to_string())


# ==================================================
# 4. ZERO-USAGE INVESTIGATION
# ==================================================


print("\n=== ZERO-USAGE ANALYSIS ===")

zero_usage_columns = [
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers"
]

zero_usage_mask = (df[zero_usage_columns] == 0).all(axis=1)
zero_usage_records = df[zero_usage_mask]

print("\n--- Zero-Usage Records ---")
print(f"Records with zero usage: {len(zero_usage_records)}")
print(f"Percentage of dataset: {zero_usage_mask.mean() * 100:.2f}%")

conflict_details = conflicting_groups.reset_index()

zero_usage_conflicts = conflict_details[
    conflict_details[zero_usage_columns].eq(0).all(axis=1)
]

print(
    "Conflicting feature combinations with zero usage:",
    len(zero_usage_conflicts)
)

print(zero_usage_conflicts.to_string(index=False))

print("\n--- CHURN RATE BY ZERO-USAGE STATUS ---")

usage_summary = (
    df.assign(Zero_Usage=zero_usage_mask)
    .groupby("Zero_Usage")["Churn"]
    .agg(
        Total_Records="count",
        Churned_Records="sum",
        Churn_Rate="mean"
    )
)

usage_summary["Churn_Rate"] = (
    usage_summary["Churn_Rate"] * 100
).round(2)

usage_summary = usage_summary.rename(
    columns={
        "Total_Records": "Total Records",
        "Churned_Records": "Churned Records",
        "Churn_Rate": "Churn Rate (%)"
    }
)

print(usage_summary.to_string())


print("\n--- ZERO USAGE BY STATUS AND CHURN ---")

zero_usage_status_summary = (
    zero_usage_records
    .groupby(["Status", "Churn"])
    .size()
    .unstack(fill_value=0)
)

print(zero_usage_status_summary.to_string())


# ==================================================
# 5. CHURN RATE BY STATUS: FULL DATASET
# ==================================================


print("\n=== CHURN RATE BY STATUS ===")

status_churn_summary = (
    df.groupby("Status")["Churn"]
    .agg(
        Total_Records="count",
        Churned_Records="sum",
        Churn_Rate="mean"
    )
)

status_churn_summary["Churn_Rate"] = (
    status_churn_summary["Churn_Rate"] * 100
).round(2)

status_churn_summary = status_churn_summary.rename(
    columns={
        "Total_Records": "Total Records",
        "Churned_Records": "Churned Records",
        "Churn_Rate": "Churn Rate (%)"
    }
)

print(status_churn_summary.to_string())


# ==================================================
# 6. AGE AND CHARGE AMOUNT INVESTIGATIONS
# ==================================================


print("\n=== AGE INSPECTION ===")

age_group_summary = (
    df.groupby(["Age", "Age Group"], dropna=False)
    .size()
    .reset_index(name="Customer Count")
    .sort_values(["Age Group", "Age"])
)

print("\n--- Age and Age Group Combinations ---")
print(age_group_summary.to_string(index=False))


print("\n=== CHARGE AMOUNT INSPECTION ===")

print("\n--- Charge Amount Frequencies ---")
print(df["Charge  Amount"].value_counts().sort_index().to_string())

print("\n--- Rows With Charge Amount 10 ---")
charge_10_rows = df[df["Charge  Amount"] == 10]

print(f"Number of rows: {len(charge_10_rows)}")

print(charge_10_rows.to_string(index=False))


# ==================================================
# 7. CATEGORICAL VALUE DISTRIBUTIONS
# ==================================================


print("\n=== CATEGORICAL VALUE DISTRIBUTIONS ===")

categorical_columns = [
    "Complains",
    "Tariff Plan",
    "Status"
]

for column in categorical_columns:
    counts = df[column].value_counts().sort_index()
    percentages = (
        df[column].value_counts(normalize=True).sort_index() * 100
    )

    summary = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2)
    })

    print(f"\n--- {column} ---")
    print(summary.to_string())


# ==================================================
# 8. DUPLICATE RECORD INVESTIGATION
# ==================================================


print("\n=== DUPLICATE RECORD INSPECTION ===")

duplicate_occurrences = df[df.duplicated()]

duplicate_rows = df[df.duplicated(keep=False)]

print("\n--- Duplicate Summary ---")
print(f"Exact duplicate occurrences: {len(duplicate_occurrences)}")
print(f"Rows involved in duplicate groups: {len(duplicate_rows)}")

print("\n--- Duplicate Occurrences by Churn ---")

print(
    duplicate_occurrences.groupby("Churn", dropna=False)
    .size()
    .rename("Duplicate Occurrences")
    .to_string()
)

print("\n--- Rows in Duplicate Groups by Churn ---")

print(
    duplicate_rows.groupby("Churn", dropna=False)
    .size()
    .rename("Rows in Duplicate Groups")
    .to_string()
)

print("\n--- MOST FREQUENT EXACT DUPLICATE RECORDS ---")

duplicate_counts = (
    df.groupby(list(df.columns), dropna=False)
    .size()
    .reset_index(name="Occurrences")
)

duplicate_counts = (
    duplicate_counts[duplicate_counts["Occurrences"] > 1]
    .sort_values("Occurrences", ascending=False)
)

print(duplicate_counts.head(10).to_string(index=False))