import pandas as pd


DATA_PATH = "data/raw/Customer Churn.csv"

ZERO_USAGE_COLUMNS = [
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers"
]


def main():

    df = pd.read_csv(DATA_PATH)

    feature_columns = df.columns.drop("Churn").tolist()

    # ==================================================
    # 1. IDENTIFY CONFLICTING FEATURE COMBINATIONS
    # ==================================================

    conflict_summary = (
        df.groupby(feature_columns)["Churn"]
        .agg(
            Total_Records="size",
            Churn_0=lambda values: (values == 0).sum(),
            Churn_1=lambda values: (values == 1).sum()
        )
        .reset_index()
    )

    conflict_summary = conflict_summary[
        (conflict_summary["Churn_0"] > 0) &
        (conflict_summary["Churn_1"] > 0)
    ].copy()

    # ==================================================
    # 2. CLASSIFY ZERO-USAGE CONFLICTS
    # ==================================================

    conflict_summary["Zero_Usage"] = (
        conflict_summary[ZERO_USAGE_COLUMNS] == 0
    ).all(axis=1)

    conflict_summary = conflict_summary.sort_values(
        by="Total_Records",
        ascending=False
    )

    # ==================================================
    # 3. SUMMARIZE FINDINGS
    # ==================================================

    print("\n=== CONFLICTING LABEL INVESTIGATION ===")

    print(f"Conflicting feature combinations: {len(conflict_summary)}")

    print(
        "Records represented by conflicting combinations:",
        conflict_summary["Total_Records"].sum()
    )

    print(
        "Conflicting combinations with zero usage:",
        conflict_summary["Zero_Usage"].sum()
    )

    print(
        "Conflicting combinations with any recorded usage:",
        (~conflict_summary["Zero_Usage"]).sum()
    )

    print("\n--- CONFLICTING GROUP DETAILS ---")

    display_columns = [
        "Subscription  Length",
        "Status",
        "Customer Value",
        "Seconds of Use",
        "Frequency of use",
        "Frequency of SMS",
        "Distinct Called Numbers",
        "Churn_0",
        "Churn_1",
        "Total_Records",
        "Zero_Usage"
    ]

    print(
        conflict_summary[display_columns]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()