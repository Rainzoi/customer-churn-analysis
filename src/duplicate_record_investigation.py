import pandas as pd


DATA_PATH = "data/raw/Customer Churn.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    # ==================================================
    # 1. DUPLICATE RECORD SUMMARY
    # ==================================================

    print("\n=== DUPLICATE RECORD SUMMARY ===")

    exact_duplicates = df.duplicated().sum()
    duplicate_group_rows = df.duplicated(keep=False).sum()

    print(f"Total records: {len(df)}")
    print(f"Exact duplicate occurrences: {exact_duplicates}")
    print(f"Rows belonging to duplicate groups: {duplicate_group_rows}")

    # ==================================================
    # 2. EXACT DUPLICATE OCCURRENCES BY CHURN
    # ==================================================

    print("\n=== EXACT DUPLICATE OCCURRENCES BY CHURN ===")

    duplicate_occurrences = (
        df.loc[df.duplicated(), "Churn"]
        .value_counts()
        .sort_index()
    )

    print(duplicate_occurrences)

    # ==================================================
    # 3. ROWS IN DUPLICATE GROUPS BY CHURN
    # ==================================================

    print("\n=== ROWS IN DUPLICATE GROUPS BY CHURN ===")

    duplicate_group_distribution = (
        df.loc[df.duplicated(keep=False), "Churn"]
        .value_counts()
        .sort_index()
    )

    print(duplicate_group_distribution)

    # ==================================================
    # 4. FREQUENCY OF REPEATED RECORD GROUPS
    # ==================================================

    print("\n=== FREQUENCY OF REPEATED RECORD GROUPS ===")

    record_group_sizes = (
        df.groupby(df.columns.tolist())
        .size()
    )

    group_frequency = (
        record_group_sizes.value_counts()
        .sort_index()
    )

    print("Number of record groups by group size:")
    print(group_frequency)


if __name__ == "__main__":
    main()