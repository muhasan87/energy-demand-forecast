import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


def evaluate():
    """
    Loads the best model (XGBoost), evaluates it in detail,
    and exports predictions for the dashboard.
    """

    # ── 1. Load data ──────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv("data/processed/features.csv", parse_dates=["timestamp"])

    X = df.drop(columns=["timestamp", "energy_demand"])
    y = df["energy_demand"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    test_timestamps = df["timestamp"].iloc[X_test.index]

    # ── 2. Load best model ────────────────────────────────
    print("Loading XGBoost model...")
    with open("src/models/saved/xgboost.pkl", "rb") as f:
        model = pickle.load(f)

    # ── 3. Make predictions ───────────────────────────────
    predictions = model.predict(X_test)

    # ── 4. Calculate metrics ──────────────────────────────
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

    print(f"\nModel Performance:")
    print(f"  RMSE: {rmse:.2f} MW")
    print(f"  MAE:  {mae:.2f} MW")
    print(f"  MAPE: {mape:.2f}%")

    # ── 5. Feature importance ─────────────────────────────
    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(f"\nFeature Importance:")
    print(importance_df)

    # ── 6. Export predictions for dashboard ───────────────
    results_df = pd.DataFrame(
        {
            "timestamp": test_timestamps.values,
            "actual": y_test.values,
            "predicted": predictions,
            "error": np.abs(y_test.values - predictions),
        }
    )

    os.makedirs("data/processed", exist_ok=True)
    results_df.to_csv("data/processed/predictions.csv", index=False)
    print(f"\nSaved predictions to data/processed/predictions.csv")

    # ── 7. Plot predicted vs actual ───────────────────────
    plt.figure(figsize=(14, 5))
    plt.plot(results_df["timestamp"], results_df["actual"], label="Actual", alpha=0.7)
    plt.plot(
        results_df["timestamp"], results_df["predicted"], label="Predicted", alpha=0.7
    )
    plt.title("Actual vs Predicted Energy Demand")
    plt.xlabel("Date")
    plt.ylabel("Energy Demand (MW)")
    plt.legend()
    plt.tight_layout()

    os.makedirs("screenshots", exist_ok=True)
    plt.savefig("screenshots/predicted_vs_actual.png")
    print("Saved plot to screenshots/predicted_vs_actual.png")
    plt.show()

    return results_df, importance_df


if __name__ == "__main__":
    evaluate()
