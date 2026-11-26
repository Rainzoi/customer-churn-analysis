import pandas as pd


# Path to the original dataset
DATA_PATH = "data/raw/Customer Churn.csv"


def main():

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
    # 2. CATEGORICAL FEATURE DISTRIBUTIONS
    # ==================================================

    categorical_features = [
        "Complains",
        "Tariff Plan",
        "Status"
    ]

    print("\n=== CATEGORICAL FEATURE DISTRIBUTIONS ===")

    for feature in categorical_features:
        print(f"\n--- {feature} ---")
        print(df[feature].value_counts().sort_index())

    # ==================================================
    # 3. CHARGE AMOUNT DATA QUALITY CHECK
    # ==================================================

    print("\n=== CHARGE AMOUNT DATA QUALITY ===")

    unexpected_charge_values = df[df["Charge  Amount"] == 10]

    print(
        "Records with Charge Amount = 10:",
        len(unexpected_charge_values)
    )


if __name__ == "__main__":
    main()


