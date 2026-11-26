import joblib
import pandas as pd

MODEL_PATH = "models/random_forest_model.joblib"


def predict_churn(customer_data):

    model = joblib.load(MODEL_PATH)

    required_features = list(model.feature_names_in_)

    customer_df = pd.DataFrame([customer_data])

    missing_features = set(required_features) - set(customer_df.columns)
    extra_features = set(customer_df.columns) - set(required_features)

    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    if extra_features:
        raise ValueError(f"Unexpected features: {extra_features}")

    customer_df = customer_df[required_features]

    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]

    result = {
        "Prediction": "Churn" if prediction == 1 else "Retained",
        "Churn Probability": round(float(probability), 4)
    }

    return result