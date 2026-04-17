import pandas as pd
import os


def build_features():
    """
    Merges weather and energy datasets on timestamp,
    producing one clean table ready for ML training.
    """
    print("Loading data...")
    weather = pd.read_csv("data/raw/weather.csv", parse_dates=["timestamp"])
    energy = pd.read_csv("data/raw/energy.csv", parse_dates=["timestamp"])

    print(f"Weather rows: {len(weather)}")
    print(f"Energy rows:  {len(energy)}")

    print("Merging on timestamp...")
    df = pd.merge(weather, energy, on="timestamp", how="inner")
    print(f"Merged rows: {len(df)}")

    df = df.drop(columns=["region"])

    df = df.dropna()
    print(f"Rows after dropping nulls: {len(df)}")

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/features.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(df.head(5))

    return df


if __name__ == "__main__":
    df = build_features()
