import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def train_models():
    # ── 1. Load data ──────────────────────────────────────
    print("Loading features...")
    df = pd.read_csv("data/processed/features.csv", parse_dates=["timestamp"])

    # ── 2. Split into features (X) and target (y) ─────────
    X = df.drop(columns=["timestamp", "energy_demand"])
    y = df["energy_demand"]

    print(f"Features: {X.columns.tolist()}")
    print(f"Rows: {len(X)}")

    # ── 3. Split into train and test sets ─────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")

    # ── 4. Define models ──────────────────────────────────
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        ),
    }

    # ── 5. Train and evaluate each model ──────────────────
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print(f"{name} RMSE: {rmse:.2f} MW")
        results[name] = rmse
        trained_models[name] = model

    # ── 6. Pick the best model ────────────────────────────
    best_name = min(results, key=results.get)
    print(f"\nBest model: {best_name} (RMSE: {results[best_name]:.2f} MW)")

    # ── 7. Save all models ────────────────────────────────
    os.makedirs("src/models/saved", exist_ok=True)

    for name, model in trained_models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        path = f"src/models/saved/{filename}"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved {name} → {path}")

    # ── 8. Save results summary ───────────────────────────
    results_df = pd.DataFrame(
        list(results.items()), columns=["model", "rmse"]
    ).sort_values("rmse")

    results_df.to_csv("src/models/saved/results.csv", index=False)
    print("\nModel comparison:")
    print(results_df)

    return trained_models, results


if __name__ == "__main__":
    train_models()
