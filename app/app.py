
import pandas as pd

from flask import Flask, render_template, request

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import predict_churn


app = Flask(__name__)

DATA_PATH = "data/raw/Customer Churn.csv"
df = pd.read_csv(DATA_PATH)


@app.route("/")
def overview():

    total_customers = len(df)

    churn_counts = df["Churn"].value_counts()

    retained_count = int(churn_counts.get(0, 0))
    churned_count = int(churn_counts.get(1, 0))

    retained_rate = round(retained_count / total_customers * 100, 2)
    churn_rate = round(churned_count / total_customers * 100, 2)

    return render_template(
        "overview.html",
        total_customers=total_customers,
        retained_count=retained_count,
        churned_count=churned_count,
        retained_rate=retained_rate,
        churn_rate=churn_rate
    )


@app.route("/analysis")
def analysis():

    analysis_df = df.copy()

    # Churn rate by complaints
    complaint_stats = (
        analysis_df.groupby("Complains")["Churn"]
        .agg(["count", "mean"])
        .reset_index()
    )
    complaint_stats["churn_rate"] = complaint_stats["mean"] * 100
    complaint_stats = complaint_stats.to_dict("records")

    # Churn rate by status
    status_stats = (
        analysis_df.groupby("Status")["Churn"]
        .agg(["count", "mean"])
        .reset_index()
    )
    status_stats["churn_rate"] = status_stats["mean"] * 100
    status_stats = status_stats.to_dict("records")

    # Churn rate by tariff plan
    tariff_stats = (
        analysis_df.groupby("Tariff Plan")["Churn"]
        .agg(["count", "mean"])
        .reset_index()
    )
    tariff_stats["churn_rate"] = tariff_stats["mean"] * 100
    tariff_stats = tariff_stats.to_dict("records")

    # Churn rate by subscription length
    analysis_df["Subscription Range"] = pd.cut(
        analysis_df["Subscription  Length"],
        bins=[0, 12, 24, 36, 48, 60],
        labels=[
            "1–12 months",
            "13–24 months",
            "25–36 months",
            "37–48 months",
            "49–60 months"
        ]
    )

    subscription_stats = (
        analysis_df.groupby("Subscription Range", observed=True)["Churn"]
        .agg(["count", "mean"])
        .reset_index()
    )
    subscription_stats["churn_rate"] = subscription_stats["mean"] * 100
    subscription_stats = subscription_stats.to_dict("records")

    # Churn rate by customer usage
    analysis_df["Usage Range"] = pd.cut(
        analysis_df["Seconds of Use"],
        bins=[-1, 3000, 6000, float("inf")],
        labels=[
            "0–3,000 seconds",
            "3,001-6,000 seconds",
            "6,001+ seconds"
        ]
    )

    usage_stats = (
        analysis_df.groupby("Usage Range", observed=True)["Churn"]
        .agg(["count", "mean"])
        .reset_index()
    )
    usage_stats["churn_rate"] = usage_stats["mean"] * 100
    usage_stats = usage_stats.to_dict("records")

    # Churn rate by SMS frequency
    analysis_df["SMS Range"] = pd.cut(
        analysis_df["Frequency of SMS"],
        bins=[-1, 25, 75, float("inf")],
        labels=[
            "0–25 messages",
            "26–75 messages",
            "76+ messages"
        ]
    )

    sms_stats = (
        analysis_df.groupby("SMS Range", observed=True)["Churn"]
        .agg(["count", "mean"])
        .reset_index()
    )
    sms_stats["churn_rate"] = sms_stats["mean"] * 100
    sms_stats = sms_stats.to_dict("records")

    # Churn rate by distinct called numbers
    analysis_df["Called Numbers Range"] = pd.cut(
        analysis_df["Distinct Called Numbers"],
        bins=[-1, 10, 25, float("inf")],
        labels=[
            "0–10 numbers",
            "11–25 numbers",
            "26+ numbers"
        ]
    )

    called_numbers_stats = (
        analysis_df.groupby("Called Numbers Range", observed=True)["Churn"]
        .agg(["count", "mean"])
        .reset_index()
    )
    called_numbers_stats["churn_rate"] = (
        called_numbers_stats["mean"] * 100
    )
    called_numbers_stats = called_numbers_stats.to_dict("records")

    return render_template(
        "analysis.html",
        complaint_stats=complaint_stats,
        status_stats=status_stats,
        tariff_stats=tariff_stats,
        subscription_stats=subscription_stats,
        usage_stats=usage_stats,
        sms_stats=sms_stats,
        called_numbers_stats=called_numbers_stats
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():

    result = None

    if request.method == "POST":
        customer_data = {
            "Call  Failure": float(request.form["Call  Failure"]),
            "Complains": int(request.form["Complains"]),
            "Subscription  Length": float(request.form["Subscription  Length"]),
            "Charge  Amount": float(request.form["Charge  Amount"]),
            "Seconds of Use": float(request.form["Seconds of Use"]),
            "Frequency of use": float(request.form["Frequency of use"]),
            "Frequency of SMS": float(request.form["Frequency of SMS"]),
            "Distinct Called Numbers": float(request.form["Distinct Called Numbers"]),
            "Tariff Plan": int(request.form["Tariff Plan"]),
            "Status": int(request.form["Status"]),
            "Age": float(request.form["Age"]),
            "Customer Value": float(request.form["Customer Value"])
        }

        result = predict_churn(customer_data)

    return render_template("predict.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
