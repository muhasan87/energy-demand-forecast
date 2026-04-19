import pandas as pd
import numpy as np
import os
import holidays


def build_features():
    """
    Merges weather and energy datasets, adds lag features,
    and engineers flags for holidays, seasons, and demand patterns.
    """

    # ── 1. Load data ──────────────────────────────────────
    print("Loading data...")
    weather = pd.read_csv("data/raw/weather.csv", parse_dates=["timestamp"])
    energy = pd.read_csv("data/raw/energy.csv", parse_dates=["timestamp"])

    print(f"Weather rows: {len(weather)}")
    print(f"Energy rows:  {len(energy)}")

    # ── 2. Merge on timestamp ─────────────────────────────
    print("Merging on timestamp...")
    df = pd.merge(weather, energy, on="timestamp", how="inner")
    df = df.drop(columns=["region"])
    print(f"Merged rows: {len(df)}")

    # ── 3. Public holiday flag ────────────────────────────
    print("Adding public holiday flag...")
    nsw_holidays = holidays.Australia(state="NSW", years=2025)
    df["is_holiday"] = df["timestamp"].dt.date.apply(
        lambda d: 1 if d in nsw_holidays else 0
    )

    # ── 4. Extreme heat flag ──────────────────────────────
    df["is_extreme_heat"] = (df["apparent_temp"] >= 35).astype(int)

    # ── 5. Season flag ────────────────────────────────────
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Summer
        elif month in [3, 4, 5]:
            return 1  # Autumn
        elif month in [6, 7, 8]:
            return 2  # Winter
        else:
            return 3  # Spring

    df["season"] = df["month"].apply(get_season)

    # ── 6. Business day flag ──────────────────────────────
    df["is_business_day"] = ((df["is_weekend"] == 0) & (df["is_holiday"] == 0)).astype(
        int
    )

    # ── 7. Lag features ───────────────────────────────────
    print("Adding lag features...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["demand_1hr_ago"] = df["energy_demand"].shift(1)
    df["demand_24hr_ago"] = df["energy_demand"].shift(24)
    df["demand_168hr_ago"] = df["energy_demand"].shift(168)
    df["temp_1hr_ago"] = df["temperature"].shift(1)
    df["temp_change"] = df["temperature"] - df["temp_1hr_ago"]

    # ── 8. Drop nulls created by lag features ─────────────
    df = df.dropna()
    print(f"Rows after adding lags and dropping nulls: {len(df)}")

    # ── 9. Save ───────────────────────────────────────────
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/features.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(df.head(3))
    print(f"\nFeatures: {df.columns.tolist()}")

    return df


if __name__ == "__main__":
    build_features()
