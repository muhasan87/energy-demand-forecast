import pandas as pd
import os
import glob
import requests
from datetime import datetime


def download_aemo_csvs(start_year=2025, start_month=1, end_year=None, end_month=None):
    """
    Automatically downloads AEMO price and demand CSVs
    for NSW1 for the specified date range.
    """
    if end_year is None:
        end_year = datetime.today().year
    if end_month is None:
        end_month = datetime.today().month

    os.makedirs("data/raw/AEMO", exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}

    year = start_year
    month = start_month

    while (year, month) <= (end_year, end_month):
        filename = f"energy_{year}_{month:02d}.csv"
        filepath = f"data/raw/AEMO/{filename}"

        if os.path.exists(filepath):
            print(f"Already exists — skipping {filename}")
        else:
            url = f"https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{year}{month:02d}_NSW1.csv"
            print(f"Downloading {filename}...")
            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    with open(filepath, "w") as f:
                        f.write(response.text)
                    print(f"Saved {filename}")
                else:
                    print(f"Failed {filename} — status {response.status_code}")
            except Exception as e:
                print(f"Error {filename}: {e}")

        if month == 12:
            month = 1
            year += 1
        else:
            month += 1


def fetch_energy():
    """
    Downloads all AEMO CSVs, combines them,
    and resamples to hourly.
    """
    # Auto download any missing months
    download_aemo_csvs(start_year=2025, start_month=1)

    folder = "data/raw/AEMO"
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    if not files:
        print("No AEMO CSV files found.")
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
