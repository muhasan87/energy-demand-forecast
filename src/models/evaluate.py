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
    residuals = y_test.values - predictions

    # ── 4. Calculate metrics ──────────────────────────────
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs(residuals / y_test.values)) * 100

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

    # ── 6. Export predictions ─────────────────────────────
    results_df = pd.DataFrame(
        {
            "timestamp": test_timestamps.values,
            "actual": y_test.values,
            "predicted": predictions,
            "error": np.abs(residuals),
            "residual": residuals,
            "hour": pd.DatetimeIndex(test_timestamps.values).hour,
            "month": pd.DatetimeIndex(test_timestamps.values).month,
        }
    )

    os.makedirs("data/processed", exist_ok=True)
    results_df.to_csv("data/processed/predictions.csv", index=False)
    print(f"\nSaved predictions to data/processed/predictions.csv")

    os.makedirs("screenshots", exist_ok=True)

    # ── 7. Plot 1 — Actual vs Predicted ───────────────────
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
    plt.savefig("screenshots/01_predicted_vs_actual.png")
    plt.close()
    print("Saved plot 1 — predicted vs actual")

    # ── 8. Plot 2 — Residual plot over time ───────────────
    plt.figure(figsize=(14, 5))
    plt.plot(results_df["timestamp"], results_df["residual"], alpha=0.6, color="coral")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.title("Residuals Over Time (Actual - Predicted)")
    plt.xlabel("Date")
    plt.ylabel("Residual (MW)")
    plt.tight_layout()
    plt.savefig("screenshots/02_residuals_over_time.png")
    plt.close()
    print("Saved plot 2 — residuals over time")

    # ── 9. Plot 3 — Error by hour of day ──────────────────
    hourly_error = results_df.groupby("hour")["error"].mean()
    plt.figure(figsize=(10, 5))
    plt.bar(hourly_error.index, hourly_error.values, color="steelblue")
    plt.title("Average Prediction Error by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Absolute Error (MW)")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig("screenshots/03_error_by_hour.png")
    plt.close()
    print("Saved plot 3 — error by hour")

    # ── 10. Plot 4 — Error by month ───────────────────────
    monthly_error = results_df.groupby("month")["error"].mean()
    plt.figure(figsize=(10, 5))
    plt.bar(monthly_error.index, monthly_error.values, color="darkorange")
    plt.title("Average Prediction Error by Month")
    plt.xlabel("Month")
    plt.ylabel("Mean Absolute Error (MW)")
    plt.xticks(monthly_error.index)
    plt.tight_layout()
    plt.savefig("screenshots/04_error_by_month.png")
    plt.close()
    print("Saved plot 4 — error by month")

    # ── 11. Plot 5 — Scatter predicted vs actual ──────────
    plt.figure(figsize=(7, 7))
    plt.scatter(
        results_df["actual"], results_df["predicted"], alpha=0.3, s=5, color="purple"
    )
    min_val = min(results_df["actual"].min(), results_df["predicted"].min())
    max_val = max(results_df["actual"].max(), results_df["predicted"].max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linewidth=1,
        label="Perfect prediction",
    )
    plt.title("Predicted vs Actual Scatter Plot")
    plt.xlabel("Actual Demand (MW)")
    plt.ylabel("Predicted Demand (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("screenshots/05_scatter_predicted_vs_actual.png")
    plt.close()
    print("Saved plot 5 — scatter plot")

    # ── 12. Plot 6 — Feature importance ───────────────────
    plt.figure(figsize=(10, 5))
    plt.barh(importance_df["feature"], importance_df["importance"], color="teal")
    plt.title("Feature Importance (XGBoost)")
    plt.xlabel("Importance Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("screenshots/06_feature_importance.png")
    plt.close()
    print("Saved plot 6 — feature importance")

    print("\nAll plots saved to screenshots/")
    return results_df, importance_df


if __name__ == "__main__":
    evaluate()
