import pandas as pd
import os
import glob


def fetch_energy():
    """
    Reads all 12 monthly AEMO CSV files, combines them,
    and resamples from 5-minute intervals to hourly.
    """
    folder = "data/raw/AEMO"
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    if not files:
        print("No AEMO CSV files found in data/raw/AEMO/")
        return None

    print(f"Found {len(files)} files — combining...")

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined: {len(df)} rows")

    df = df.rename(
        columns={
            "SETTLEMENTDATE": "timestamp",
            "TOTALDEMAND": "energy_demand",
            "REGION": "region",
        }
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[["timestamp", "energy_demand", "region"]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = df.set_index("timestamp")
    df = df.resample("h")["energy_demand"].mean().reset_index()
    df["region"] = "NSW1"

    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/energy.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

    return df


if __name__ == "__main__":
    df = fetch_energy()
    print(df.head(10))
