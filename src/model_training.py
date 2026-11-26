import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

from data_preperation import prepare_data


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC": average_precision_score(y_test, probabilities)
    }

    return predictions, probabilities, results


def display_results(model_name, results):

    print(f"\n=== {model_name.upper()} RESULTS ===")

    for metric, value in results.items():
        print(f"{metric + ':':<11} {value:.4f}")


def main():

    # ==================================================
    # 1. LOAD PREPARED DATA
    # ==================================================

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
    # 2. BUILD LOGISTIC REGRESSION PIPELINE
    # ==================================================

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    logistic_model.fit(X_train, y_train)

    y_pred, y_probability, logistic_results = evaluate_model(
        logistic_model, X_test, y_test
    )

    display_results("Logistic Regression", logistic_results)

    # ==================================================
    # 3. TRAIN RANDOM FOREST
    # ==================================================

    random_forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    random_forest.fit(X_train, y_train)

        # Save trained Random Forest model
    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        random_forest,
        "models/random_forest_model.joblib"
    )

    print("\nRandom Forest model saved successfully.")

    rf_pred, rf_probability, rf_results = evaluate_model(
        random_forest, X_test, y_test
    )

    display_results("Random Forest", rf_results)

    # ==================================================
    # 4. TRAIN XGBOOST
    # ==================================================

    xgboost_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    xgboost_model.fit(X_train, y_train)

    xgb_pred, xgb_probability, xgb_results = evaluate_model(
        xgboost_model, X_test, y_test
    )

    display_results("XGBoost", xgb_results)

    # ==================================================
    # 5. TRAIN LIGHTGBM
    # ==================================================

    lightgbm_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    lightgbm_model.fit(X_train, y_train)

    lgbm_pred, lgbm_probability, lgbm_results = evaluate_model(
        lightgbm_model, X_test, y_test
    )

    display_results("LightGBM", lgbm_results)

    # ==================================================
    # 6. CONFUSION MATRIX COMPARISON
    # ==================================================

    models_predictions = {
        "Logistic Regression": y_pred,
        "Random Forest": rf_pred,
        "XGBoost": xgb_pred,
        "LightGBM": lgbm_pred
    }

    confusion_results = []

    for model_name, predictions in models_predictions.items():

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        confusion_results.append({
            "Model": model_name,
            "True Negatives": tn,
            "False Positives": fp,
            "False Negatives": fn,
            "True Positives": tp
        })

    confusion_df = pd.DataFrame(confusion_results)

    print("\n=== CONFUSION MATRIX COMPARISON ===")
    print(confusion_df.to_string(index=False))

    # ==================================================
    # 7. RANDOM FOREST FEATURE IMPORTANCE
    # ==================================================

    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": random_forest.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")
    print(feature_importance.to_string(index=False))

    # ==================================================
    # 8. SHAP FEATURE EXPLANATIONS
    # ==================================================

    explainer = shap.TreeExplainer(random_forest)

    shap_values = explainer(X_test)

    if shap_values.values.ndim == 3:
        churn_shap_values = shap_values[:, :, 1]
    else:
        churn_shap_values = shap_values

    print("\n=== SHAP GLOBAL FEATURE IMPORTANCE ===")
    shap.plots.bar(churn_shap_values)

    print("\n=== SHAP SUMMARY PLOT ===")
    shap.plots.beeswarm(churn_shap_values)

    # ==================================================
    # 9. SHAP DEPENDENCE PLOTS
    # ==================================================

    print("\n=== SHAP DEPENDENCE: STATUS ===")

    shap.plots.scatter(
        churn_shap_values[:, "Status"],
        x_jitter=0.2
    )

    print("\n=== SHAP DEPENDENCE: COMPLAINS ===")

    shap.plots.scatter(
        churn_shap_values[:, "Complains"],
        x_jitter=0.2
    )


if __name__ == "__main__":
    main()