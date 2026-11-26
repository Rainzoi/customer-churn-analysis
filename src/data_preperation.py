import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# Path to the original dataset
DATA_PATH = "data/raw/Customer Churn.csv"


def prepare_data():

    df = pd.read_csv(DATA_PATH)

    # ==================================================
    # 1. SEPARATE FEATURES AND TARGET
    # ==================================================

    X = df.drop(columns=["Churn", "Age Group"])
    y = df["Churn"]

    # ==================================================
    # 2. CREATE GROUP IDENTIFIERS
    # ==================================================

    # Rows with identical predictor values receive the same
    # group identifier. This keeps identical feature
    # combinations together during the split.

    groups = pd.MultiIndex.from_frame(X).factorize()[0]

    # ==================================================
    # 3. CREATE GROUPED TRAIN/TEST SPLIT
    # ==================================================

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    # ==================================================
    # 4. CALCULATE CLASS WEIGHTING
    # ==================================================

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scale_pos_weight,
        groups,
        train_idx,
        test_idx
    )


def main():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scale_pos_weight,
        groups,
        train_idx,
        test_idx
    ) = prepare_data()

    # ==================================================
    # 5. REVIEW THE SPLIT
    # ==================================================

    print("\n=== SPLIT OVERVIEW ===")

    print(f"Original rows: {len(X_train) + len(X_test)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    print(f"\nNumber of features: {X_train.shape[1]}")
    print("Features:")
    print(X_train.columns.tolist())

    # ==================================================
    # 6. CHECK CHURN DISTRIBUTION
    # ==================================================

    print("\n=== TRAINING CHURN DISTRIBUTION (%) ===")

    print(
        y_train.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\n=== TESTING CHURN DISTRIBUTION (%) ===")

    print(
        y_test.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    # ==================================================
    # 7. CHECK FOR GROUP OVERLAP
    # ==================================================

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    shared_groups = train_groups.intersection(test_groups)

    print("\n=== GROUP SPLIT CHECK ===")
    print(f"Shared groups between training and testing: {len(shared_groups)}")

    # ==================================================
    # 8. DISPLAY CLASS IMBALANCE WEIGHTING
    # ==================================================

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    print("\n=== CLASS IMBALANCE WEIGHTING ===")
    print(f"Retained customers: {negative_count}")
    print(f"Churned customers: {positive_count}")
    print(f"Scale positive weight: {scale_pos_weight:.4f}")


if __name__ == "__main__":
    main()


