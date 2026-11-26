import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = "data/raw/Customer Churn.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    # ==================================================
    # 1. OVERALL CHURN DISTRIBUTION
    # ==================================================

    print("\n--- Overall Churn Distribution ---")

    churn_counts = df["Churn"].value_counts().sort_index()
    churn_percentages = (
        df["Churn"].value_counts(normalize=True).sort_index() * 100
    )

    churn_summary = pd.DataFrame({
        "Count": churn_counts,
        "Percentage": churn_percentages
    })

    churn_summary.index = ["Retained", "Churned"]

    print(churn_summary)

    # ==================================================
    # 2. CUSTOMER FEATURE STATISTICS BY CHURN
    # ==================================================

    numeric_features = [
        "Call  Failure",
        "Subscription  Length",
        "Seconds of Use",
        "Frequency of use",
        "Frequency of SMS",
        "Distinct Called Numbers",
        "Customer Value"
    ]

    print("\n--- Customer Feature Statistics by Churn ---")

    feature_statistics = (
        df.groupby("Churn")[numeric_features]
        .agg(["min", "median", "mean", "max", "std"])
        .transpose()
    )

    print(feature_statistics)

    # ==================================================
    # 3. CHURN RATES BY CATEGORICAL FEATURES
    # ==================================================

    categorical_features = [
        "Complains",
        "Tariff Plan",
        "Status"
    ]

    print("\n--- Churn Rates by Categorical Features ---")

    for feature in categorical_features:

        churn_rates = (
            df.groupby(feature)["Churn"]
            .mean()
            .mul(100)
        )

        print(f"\n--- {feature} ---")
        print(churn_rates.round(2))

    # ==================================================
    # 4. CHURN RATES BY AGE GROUP
    # ==================================================

    print("\n--- Churn Rates by Age Group ---")

    age_group_churn = (
        df.groupby("Age Group")["Churn"]
        .agg(
            Customer_Count="size",
            Churn_Rate=lambda values: values.mean() * 100
        )
    )

    print(age_group_churn.round(2))

    # ==================================================
    # 5. CUSTOMER USAGE BOXPLOTS
    # ==================================================

    usage_features = [
        "Seconds of Use",
        "Frequency of use",
        "Frequency of SMS",
        "Distinct Called Numbers",
        "Customer Value"
    ]

    print("\n--- Customer Usage Boxplots ---")

    for feature in usage_features:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x="Churn",
            y=feature
        )

        plt.title(f"{feature} by Churn Status")
        plt.xlabel("Customer Status")
        plt.ylabel(feature)
        plt.tight_layout()
        plt.show()

    # ==================================================
    # 6. FEATURE SKEWNESS BY CHURN
    # ==================================================

    print("\n--- Feature Skewness by Churn ---")

    skewness = (
        df.groupby("Churn")[usage_features]
        .skew()
        .transpose()
    )

    skewness.columns = ["Retained", "Churned"]

    print(skewness.round(2))

    # ==================================================
    # 7. CUSTOMER USAGE HISTOGRAMS
    # ==================================================

    print("\n--- Customer Usage Histograms ---")

    for feature in usage_features:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=feature,
            hue="Churn",
            stat="density",
            common_norm=False,
            bins=30,
            element="step"
        )

        plt.title(f"{feature} Distribution by Churn Status")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()