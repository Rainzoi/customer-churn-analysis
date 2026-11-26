from itertools import combinations

import numpy as np
import pandas as pd


DATA_PATH = "data/raw/Customer Churn.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    # ==================================================
    # 1. AGE AND AGE GROUP CONSISTENCY
    # ==================================================

    print("\n=== AGE AND AGE GROUP CONSISTENCY ===")

    groups_per_age = df.groupby("Age")["Age Group"].nunique()
    ages_per_group = df.groupby("Age Group")["Age"].nunique()

    print(
        "Ages associated with multiple age groups:",
        (groups_per_age > 1).sum()
    )

    print(
        "Age groups associated with multiple ages:",
        (ages_per_group > 1).sum()
    )

    # ==================================================
    # 2. IDENTICAL FEATURE COLUMNS
    # ==================================================

    print("\n=== IDENTICAL FEATURE COLUMNS ===")

    feature_columns = df.drop(columns=["Churn"]).columns.tolist()

    identical_columns = []

    for column_1, column_2 in combinations(feature_columns, 2):

        if df[column_1].equals(df[column_2]):
            identical_columns.append((column_1, column_2))

    if identical_columns:

        for column_1, column_2 in identical_columns:
            print(f"{column_1} and {column_2}")

    else:
        print("No identical feature columns found.")

    # ==================================================
    # 3. NUMERIC FEATURE CORRELATIONS
    # ==================================================

    print("\n=== NUMERIC FEATURE CORRELATIONS ===")

    correlation_matrix = df[feature_columns].corr(method="pearson")

    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1
        ).astype(bool)
    )

    strong_correlations = (
        upper_triangle
        .stack()
        .reset_index()
    )

    strong_correlations.columns = [
        "Feature 1",
        "Feature 2",
        "Correlation"
    ]

    strong_correlations = strong_correlations[
        strong_correlations["Correlation"].abs() >= 0.80
    ].sort_values(
        by="Correlation",
        key=abs,
        ascending=False
    )

    if strong_correlations.empty:
        print("No strong correlations found.")

    else:
        print(strong_correlations.to_string(index=False))

    # ==================================================
    # 4. CUSTOMER VALUE INVESTIGATION
    # ==================================================

    print("\n=== CUSTOMER VALUE INVESTIGATION ===")

    customer_value_features = [
        "Seconds of Use",
        "Frequency of use",
        "Frequency of SMS",
        "Charge  Amount"
    ]

    customer_value_data = df[
        ["Customer Value"] + customer_value_features
    ]

    print("\n--- PEARSON CORRELATIONS ---")

    print(
        customer_value_data.corr(method="pearson")["Customer Value"]
        .drop("Customer Value")
        .sort_values(ascending=False)
    )

    print("\n--- SPEARMAN CORRELATIONS ---")

    print(
        customer_value_data.corr(method="spearman")["Customer Value"]
        .drop("Customer Value")
        .sort_values(ascending=False)
    )

    print("\n--- CUSTOMER VALUE BY CHARGE AMOUNT ---")

    customer_value_by_charge = (
        df.groupby("Charge  Amount")["Customer Value"]
        .agg(
            Count="size",
            Mean="mean",
            Median="median"
        )
    )

    print(customer_value_by_charge)


if __name__ == "__main__":
    main()