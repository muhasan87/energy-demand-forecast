import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()


def fetch_weather(latitude=-33.8688, longitude=151.2093, days_back=90):
    """
    Fetch hourly historical weather data from Open-Meteo.
    Default location: Sydney, Australia.
    """
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cloud_cover",
            "apparent_temperature",
        ],
        "timezone": "Australia/Sydney",
    }

    print(f"Fetching weather data from {start_date} to {end_date}...")
    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    hourly = data["hourly"]

    df = pd.DataFrame(
        {
            "timestamp": hourly["time"],
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "wind_speed": hourly["wind_speed_10m"],
            "cloud_cover": hourly["cloud_cover"],
            "apparent_temp": hourly["apparent_temperature"],
        }
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_peak_hour"] = df["hour"].isin(range(7, 22)).astype(int)

    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/weather.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

    return df


if __name__ == "__main__":
    df = fetch_weather()
    print(df.head())
