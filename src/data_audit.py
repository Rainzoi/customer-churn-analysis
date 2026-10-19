import pandas as pd


# Path to the original dataset
DATA_PATH = "data/raw/Customer Churn.csv"


df = pd.read_csv(DATA_PATH)

print("-Data on the file-")
print(df.describe())

print("\n-Info on the dataset-")
print(df.info())

print("\n-Rows and columns-")
print(df.shape)

print("\n-Unique values-")
print(df.nunique())

print("\n-Churn distribution-")
print(df["Churn"].value_counts())
print(f"Churn percentage: {df["Churn"].value_counts(normalize=True)*100}")

print("\n-Same features, different churn outcomes-")
feature_cols = [col for col in df.columns if col != "Churn"]
conflicting_rows = (
    df.groupby(feature_cols)["Churn"]
    .nunique()
)
conflicting_groups = conflicting_rows[conflicting_rows > 1]

print(f"Feature combinations with conflicting churn outcomes: {len(conflicting_groups)}")

print("\n-Conflicting churn details")
conflicting_features = (
    df.groupby(feature_cols, dropna=False)["Churn"]
    .nunique()
)
conflicting_features = conflicting_features[conflicting_features > 1]

for features in conflicting_features.index:
    mask = pd.Series(True, index=df.index)
    
    for column, value in zip(feature_cols, features):
        mask &= df[column].eq(value)

    print("\nFeature combination:")
    print(dict(zip(feature_cols, features)))

    print("Churn outcome counts:")
    print(df.loc[mask, "Churn"].value_counts().sort_index())

print("\n-Duplicates in dataset-")
print(df.duplicated().sum())

print("\n-Analyze duplicated rows-")
duplicate_rows = df[df.duplicated(keep=False)]
print(f"Rows involved in duplication: {len(duplicate_rows)}")

print("\nDuplicate groups")
print(duplicate_rows.value_counts().head(10))

print("\n-Age values-")
print(sorted(df["Age"].unique()))

print("\n-Age group values-")
print(sorted(df["Age Group"].unique()))

print("\n-Age and age groups combinations-")
print(
    df.groupby(["Age", "Age Group"])
    .size()
    .reset_index(name="Customer Count")
    .sort_values(["Age Group", "Age"])
    .to_string(index=False)
)

print("\n-Charge amount values -")
print(sorted(df["Charge  Amount"].unique()))

print("\n-Rows with charge amount 10-")
print(df[df["Charge  Amount"] == 10].to_string(index=False))

print("\n-Charge amount frequency-")
print(df["Charge  Amount"].value_counts().sort_index())


print("\n-Categorical value counts-")
for column in ["Complains", "Tariff Plan", "Status", "Churn"]:
    print("\n")
    counts = df[column].value_counts().sort_index()
    
    percentages = (
        df[column]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    summary = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages
    })

    print(summary)