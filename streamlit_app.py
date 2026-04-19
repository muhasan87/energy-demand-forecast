import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import holidays

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="NSW Energy Demand Forecaster", page_icon="⚡", layout="wide"
)

# ── Sidebar width + button styling ────────────────────────
st.markdown(
    """
<style>
    [data-testid="stSidebar"] {
        min-width: 320px;
        max-width: 320px;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 18px !important;
        padding: 8px 0px !important;
    }
    [data-testid="stSidebar"] .stRadio div {
        gap: 12px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── API URL ───────────────────────────────────────────────
API_URL = "http://54.91.10.150:8000"


# ── Helper functions ──────────────────────────────────────
def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Summer
    elif month in [3, 4, 5]:
        return 1  # Autumn
    elif month in [6, 7, 8]:
        return 2  # Winter
    else:
        return 3  # Spring


def is_nsw_holiday(date):
    nsw_holidays = holidays.Australia(state="NSW", years=date.year)
    return 1 if date.date() in nsw_holidays else 0


def fetch_live_aemo():
    """Fetch current NSW demand and look up historical values."""
    try:
        url = "https://visualisations.aemo.com.au/aemo/apps/api/report/ELEC_NEM_SUMMARY"
        response = requests.get(url, timeout=10)
        data = response.json()
        records = [r for r in data["ELEC_NEM_SUMMARY"] if r["REGIONID"] == "NSW1"]

        if not records:
            return None

        latest = records[0]
        current_demand = latest["TOTALDEMAND"]
        current_time = pd.to_datetime(latest["SETTLEMENTDATE"])

        # Load historical energy data
        df = pd.read_csv("data/raw/energy.csv", parse_dates=["timestamp"])

        # Round current time to nearest hour
        current_hour = current_time.floor("h")

        # Same hour yesterday
        yesterday_hour = current_hour - pd.Timedelta(hours=24)
        yesterday_row = df[df["timestamp"] == yesterday_hour]
        demand_24h = (
            float(yesterday_row["energy_demand"].values[0])
            if len(yesterday_row) > 0
            else current_demand
        )

        # Same hour last week
        lastweek_hour = current_hour - pd.Timedelta(hours=168)
        lastweek_row = df[df["timestamp"] == lastweek_hour]
        demand_168h = (
            float(lastweek_row["energy_demand"].values[0])
            if len(lastweek_row) > 0
            else current_demand
        )

        return {
            "demand": current_demand,
            "demand_24h": demand_24h,
            "demand_168h": demand_168h,
            "timestamp": latest["SETTLEMENTDATE"],
            "yesterday_hour": str(yesterday_hour),
            "lastweek_hour": str(lastweek_hour),
        }

    except Exception as e:
        print(f"AEMO fetch error: {e}")
    return None


# ── Sidebar navigation ────────────────────────────────────
st.sidebar.markdown("# ⚡ NSW Energy Forecaster")
st.sidebar.markdown(
    "*Predicting hourly electricity demand for New South Wales, Australia*"
)
st.sidebar.divider()
st.sidebar.markdown("### Navigate")
page = st.sidebar.radio(
    "", ["📊 Dashboard", "⚡ Demand Simulator", "📅 24hr Forecast", "📈 Model Journey"]
)
st.sidebar.divider()
st.sidebar.markdown("**Model:** XGBoost")
st.sidebar.markdown("**MAPE:** 2.79%")
st.sidebar.markdown("**Data:** AEMO + Open-Meteo 2025")
st.sidebar.divider()
st.sidebar.markdown("*Built by Muhammad Halidh*")
st.sidebar.markdown(
    "[GitHub](https://github.com/muhasan87/energy-demand-forecast) | [API Docs](http://54.91.10.150:8000/docs)"
)

# ════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("NSW Energy Demand — Model Performance Dashboard")
    st.markdown("*XGBoost model trained on 2025 AEMO data*")

    try:
        df = pd.read_csv("data/processed/predictions.csv", parse_dates=["timestamp"])

        # ── Key metrics ───────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("RMSE", "259.37 MW")
        col1.caption(
            "Average prediction error, weighted toward larger mistakes. Industry standard for hourly energy forecasting is 300–500 MW. Our model beats this benchmark."
        )

        col2.metric("MAE", "180.16 MW")
        col2.caption(
            "Average absolute difference between predicted and actual demand across all test hours."
        )

        col3.metric("MAPE", "2.79%")
        col3.caption(
            "Predictions are within 2.79% of actual demand on average. Under 5% is considered strong performance for hourly energy forecasting."
        )

        col4.metric("Test rows", "1,719")
        col4.caption(
            "Hours of real NSW electricity data the model was evaluated on — none of which were seen during training."
        )

        st.divider()

        # ── Actual vs Predicted ───────────────────────────
        st.subheader("Actual vs Predicted Energy Demand")
        st.markdown(
            "Compares what the model predicted against real NSW electricity demand across the test period (October–December 2025). The closer the two lines, the better the model performed."
        )
        fig1 = px.line(
            df,
            x="timestamp",
            y=["actual", "predicted"],
            labels={"value": "Energy Demand (MW)", "timestamp": "Date"},
            color_discrete_map={"actual": "#1f77b4", "predicted": "#ff7f0e"},
        )
        st.plotly_chart(fig1, use_container_width=True)

        # ── Error by hour ─────────────────────────────────
        st.subheader("Average Prediction Error by Hour of Day")
        st.markdown(
            "Shows which hours of the day the model struggles with most. Afternoon peak hours (12pm–3pm) have the highest error as demand is most volatile during this period due to simultaneous commercial, industrial and residential activity."
        )
        hourly_error = df.groupby("hour")["error"].mean().reset_index()
        fig2 = px.bar(
            hourly_error,
            x="hour",
            y="error",
            labels={"error": "Mean Absolute Error (MW)", "hour": "Hour of Day"},
            color="error",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Error by month ────────────────────────────────
        st.subheader("Average Prediction Error by Month")
        st.markdown(
            "December shows the highest error as NSW transitions into summer — demand patterns shift significantly as temperatures rise and aircon usage increases. The model had limited summer training data for this period."
        )
        monthly_error = df.groupby("month")["error"].mean().reset_index()
        fig3 = px.bar(
            monthly_error,
            x="month",
            y="error",
            labels={"error": "Mean Absolute Error (MW)", "month": "Month"},
            color="error",
            color_continuous_scale="Oranges",
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ── Scatter plot ──────────────────────────────────
        st.subheader("Predicted vs Actual Scatter Plot")
        st.markdown(
            "Each dot represents one hour of prediction. Dots sitting on the red line indicate a perfect prediction. The tighter the cluster around the line, the more accurate the model — with some spread at extreme high and low demand values."
        )
        fig4 = px.scatter(
            df,
            x="actual",
            y="predicted",
            opacity=0.4,
            labels={
                "actual": "Actual Demand (MW)",
                "predicted": "Predicted Demand (MW)",
            },
        )
        fig4.add_shape(
            type="line",
            x0=df["actual"].min(),
            y0=df["actual"].min(),
            x1=df["actual"].max(),
            y1=df["actual"].max(),
            line=dict(color="red", width=1),
        )
        st.plotly_chart(fig4, use_container_width=True)

    except FileNotFoundError:
        st.error("predictions.csv not found — run evaluate.py first.")

# ════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTION TOOL
# ════════════════════════════════════════════════════════════
elif page == "⚡ Demand Simulator":
    st.title("NSW Energy Demand Predictor")
    st.markdown(
        "Adjust the weather conditions and time below to simulate any scenario — the model will predict what NSW electricity demand would be under those exact conditions. Live grid data is automatically fetched from AEMO to provide real context."
    )

    # ── Live data banner at top ───────────────────────────
    live = fetch_live_aemo()
    if live:
        readable_time = pd.to_datetime(live["timestamp"]).strftime("%I:%M %p, %d %b %Y")
        demand_1hr_ago = live["demand"]
        demand_24hr_ago = live["demand_24h"]
        demand_168hr_ago = live["demand_168h"]
        st.success(f"⚡ Live NSW demand: {demand_1hr_ago:,.0f} MW")
        st.caption(f"Last updated: {readable_time} AEST · Updates every 5 minutes")
    else:
        st.warning("Could not fetch live data — using default values")
        demand_1hr_ago = 7000.0
        demand_24hr_ago = 7000.0
        demand_168hr_ago = 7000.0

    st.divider()

    # ── Two columns with padding ──────────────────────────
    col1, spacer, col2 = st.columns([5, 1, 5])

    with col1:
        st.markdown("### 🌤 Weather Conditions")
        st.markdown("")
        temperature = st.slider("Temperature (°C)", -5.0, 45.0, 22.0)
        apparent_temp = st.slider("Apparent Temperature (°C)", -5.0, 50.0, 22.0)
        with st.expander("⚙️ Advanced weather inputs"):
            humidity = st.slider("Humidity (%)", 0, 100, 60)
            wind_speed = st.slider("Wind Speed (km/h)", 0.0, 100.0, 15.0)
            cloud_cover = st.slider("Cloud Cover (%)", 0, 100, 30)

    with col2:
        st.markdown("### 🕐 Time & Context")
        st.markdown("")
        date = st.date_input("Date", datetime.today())
        hour = st.slider("Hour of Day", 0, 23, datetime.now().hour)

    st.divider()

    # ── Auto calculate flags ──────────────────────────────
    dt = datetime.combine(date, datetime.min.time()).replace(hour=hour)
    day_of_week = dt.weekday()
    month = dt.month
    is_weekend = 1 if day_of_week >= 5 else 0
    is_peak_hour = 1 if 7 <= hour <= 22 else 0
    is_holiday = is_nsw_holiday(dt)
    is_extreme_heat = 1 if apparent_temp >= 35 else 0
    season = get_season(month)
    is_business_day = 1 if is_weekend == 0 and is_holiday == 0 else 0
    temp_1hr_ago = temperature - 0.5
    temp_change = 0.5

    # ── Auto calculated values ────────────────────────────
    st.subheader("Auto-calculated features")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Day of week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day_of_week]
    )
    c2.metric("Is weekend", "Yes" if is_weekend else "No")
    c3.metric("Is holiday", "Yes" if is_holiday else "No")
    c4.metric("Is business day", "Yes" if is_business_day else "No")
    c5.metric("Season", ["Summer", "Autumn", "Winter", "Spring"][season])

    st.divider()

    # ── Predict button ────────────────────────────────────
    col_btn, col_msg = st.columns([2, 5])
    with col_btn:
        predict_clicked = st.button("⚡ Predict Energy Demand", type="primary")
    with col_msg:
        st.caption(
            "⏱ First prediction may take 20–30 seconds while the API wakes up. Subsequent predictions are instant."
        )
    if predict_clicked:
        payload = {
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "cloud_cover": cloud_cover,
            "apparent_temp": apparent_temp,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "is_peak_hour": is_peak_hour,
            "is_holiday": is_holiday,
            "is_extreme_heat": is_extreme_heat,
            "season": season,
            "is_business_day": is_business_day,
            "demand_1hr_ago": demand_1hr_ago,
            "demand_24hr_ago": demand_24hr_ago,
            "demand_168hr_ago": demand_168hr_ago,
            "temp_1hr_ago": temp_1hr_ago,
            "temp_change": temp_change,
        }

        with st.spinner("Getting prediction from API..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                result = response.json()

                st.success("Prediction complete!")
                st.metric(
                    label="Predicted NSW Energy Demand",
                    value=f"{result['predicted_demand_mw']:,.0f} MW",
                )

                avg_demand = 7200
                diff = result["predicted_demand_mw"] - avg_demand
                if diff > 500:
                    st.warning(f"⚠️ High demand — {diff:,.0f} MW above average")
                elif diff < -500:
                    st.info(f"💙 Low demand — {abs(diff):,.0f} MW below average")
                else:
                    st.success("✅ Demand is within normal range")

            except Exception as e:
                st.error(f"API error: {e}")


# ════════════════════════════════════════════════════════════
# PAGE 3 — 24HR FORECAST
# ════════════════════════════════════════════════════════════
elif page == "📅 24hr Forecast":
    st.title("24-Hour NSW Energy Demand Forecast")
    st.markdown(
        "Automatically forecasts hourly NSW electricity demand using real weather forecasts from Open-Meteo and live grid data from AEMO."
    )
    st.info(
        "⏱ First load may take 20–30 seconds — the forecast makes 24 sequential API calls. Subsequent loads are faster."
    )

    with st.spinner("Fetching forecast data..."):
        # ── Fetch weather forecast ────────────────────────
        try:
            forecast_url = "https://api.open-meteo.com/v1/forecast"
            forecast_params = {
                "latitude": -33.8688,
                "longitude": 151.2093,
                "hourly": [
                    "temperature_2m",
                    "apparent_temperature",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "cloud_cover",
                ],
                "timezone": "Australia/Sydney",
                "forecast_days": 2,
            }
            forecast_response = requests.get(
                forecast_url, params=forecast_params, timeout=10
            )
            forecast_data = forecast_response.json()

            forecast_df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(forecast_data["hourly"]["time"]),
                    "temperature": forecast_data["hourly"]["temperature_2m"],
                    "apparent_temp": forecast_data["hourly"]["apparent_temperature"],
                    "humidity": forecast_data["hourly"]["relative_humidity_2m"],
                    "wind_speed": forecast_data["hourly"]["wind_speed_10m"],
                    "cloud_cover": forecast_data["hourly"]["cloud_cover"],
                }
            )

            # Filter to next 24 hours from now
            now = pd.Timestamp.now(tz="Australia/Sydney").tz_localize(None)
            forecast_df = (
                forecast_df[forecast_df["timestamp"] >= now.floor("h")]
                .head(24)
                .reset_index(drop=True)
            )

        except Exception as e:
            st.error(f"Could not fetch weather forecast: {e}")
            st.stop()

        # ── Fetch live AEMO data ──────────────────────────
        live = fetch_live_aemo()
        if live:
            current_demand = live["demand"]
            demand_24hr_ago = live["demand_24h"]
            demand_168hr_ago = live["demand_168h"]
            readable_time = pd.to_datetime(live["timestamp"]).strftime(
                "%I:%M %p, %d %b %Y"
            )
            st.success(
                f"⚡ Current NSW demand: {current_demand:,.0f} MW · Last updated: {readable_time} AEST"
            )
        else:
            current_demand = 7000.0
            demand_24hr_ago = 7000.0
            demand_168hr_ago = 7000.0
            st.warning("Could not fetch live AEMO data — using default values")

        # ── Run recursive forecast ────────────────────────
        nsw_holidays = holidays.Australia(state="NSW", years=2026)
        predictions = []
        prev_demand = current_demand

        for _, row in forecast_df.iterrows():
            hour = row["timestamp"].hour
            month = row["timestamp"].month
            day_of_week = row["timestamp"].weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            is_holiday = 1 if row["timestamp"].date() in nsw_holidays else 0
            is_peak_hour = 1 if 7 <= hour <= 22 else 0
            is_extreme_heat = 1 if row["apparent_temp"] >= 35 else 0
            is_business_day = 1 if is_weekend == 0 and is_holiday == 0 else 0
            season = get_season(month)
            temp_1hr_ago = row["temperature"] - 0.5
            temp_change = 0.5

            payload = {
                "temperature": row["temperature"],
                "humidity": row["humidity"],
                "wind_speed": row["wind_speed"],
                "cloud_cover": row["cloud_cover"],
                "apparent_temp": row["apparent_temp"],
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": is_weekend,
                "is_peak_hour": is_peak_hour,
                "is_holiday": is_holiday,
                "is_extreme_heat": is_extreme_heat,
                "season": season,
                "is_business_day": is_business_day,
                "demand_1hr_ago": prev_demand,
                "demand_24hr_ago": demand_24hr_ago,
                "demand_168hr_ago": demand_168hr_ago,
                "temp_1hr_ago": temp_1hr_ago,
                "temp_change": temp_change,
            }

            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                result = response.json()
                prediction = result["predicted_demand_mw"]
            except:
                prediction = prev_demand

            predictions.append(
                {
                    "time": row["timestamp"].strftime("%I:%M %p"),
                    "timestamp": row["timestamp"],
                    "temperature": row["temperature"],
                    "apparent_temp": row["apparent_temp"],
                    "predicted_mw": prediction,
                }
            )

            prev_demand = prediction

        results_df = pd.DataFrame(predictions)

        st.divider()

        # ── Summary metrics ───────────────────────────────
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Peak demand",
            f"{results_df['predicted_mw'].max():,.0f} MW",
            results_df.loc[results_df["predicted_mw"].idxmax(), "time"],
        )
        col2.metric(
            "Lowest demand",
            f"{results_df['predicted_mw'].min():,.0f} MW",
            results_df.loc[results_df["predicted_mw"].idxmin(), "time"],
        )
        col3.metric("Average demand", f"{results_df['predicted_mw'].mean():,.0f} MW")

        st.divider()

        # ── Forecast chart ────────────────────────────────
        st.subheader("Hourly Demand Forecast")
        st.markdown(
            "Predicted NSW electricity demand for the next 24 hours based on Open-Meteo weather forecasts."
        )

        fig = px.line(
            results_df,
            x="timestamp",
            y="predicted_mw",
            labels={"predicted_mw": "Predicted Demand (MW)", "timestamp": "Time"},
            color_discrete_sequence=["#ff7f0e"],
        )
        fig.update_traces(line_width=2.5)
        st.plotly_chart(fig, use_container_width=True)

        # ── Hourly table ──────────────────────────────────
        st.subheader("Hourly Breakdown")
        display_df = results_df[
            ["time", "temperature", "apparent_temp", "predicted_mw"]
        ].copy()
        display_df.columns = [
            "Time",
            "Temperature (°C)",
            "Feels Like (°C)",
            "Predicted Demand (MW)",
        ]
        display_df["Predicted Demand (MW)"] = (
            display_df["Predicted Demand (MW)"].round(0).astype(int)
        )
        display_df["Temperature (°C)"] = display_df["Temperature (°C)"].round(1)
        display_df["Feels Like (°C)"] = display_df["Feels Like (°C)"].round(1)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.caption(
            "⚠️ Forecast uses recursive predictions — each hour's predicted demand is used as input for the next hour. Weather data sourced from Open-Meteo. Grid context from AEMO."
        )
# ════════════════════════════════════════════════════════════
# PAGE 4 — MODEL JOURNEY
# ════════════════════════════════════════════════════════════
elif page == "📈 Model Journey":
    st.title("Model Development Journey")
    st.markdown(
        "This project went through three stages of improvement — from a basic baseline to a production-ready forecasting model."
    )

    st.divider()

    # ── Stage 1 ───────────────────────────────────────────
    st.subheader("Stage 1 — Baseline Model")
    st.markdown(
        "Started with a default XGBoost model using only weather and time features — no lag features, no holiday flags."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Test RMSE", "812 MW")
    col2.metric("MAPE", "8.06%")
    col3.metric("Overfit gap", "626 MW")
    col1.caption("Far above industry benchmark of 300–500 MW")
    col2.caption("Over 8% error — too high for reliable forecasting")
    col3.caption("Model was memorising training data, not learning patterns")

    st.image(
        "screenshots_round1/01_predicted_vs_actual.png",
        caption="Stage 1 — Large gaps between actual and predicted demand, especially in December.",
    )

    st.divider()

    # ── Stage 2 ───────────────────────────────────────────
    st.subheader("Stage 2 — Hyperparameter Tuning")
    st.markdown(
        "Reduced overfitting by constraining the model through regularisation — limiting tree depth, reducing learning rate, and subsampling."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Test RMSE", "763 MW", delta="-49 MW", delta_color="inverse")
    col2.metric("MAPE", "8.06%")
    col3.metric("Overfit gap", "411 MW", delta="-215 MW", delta_color="inverse")
    col1.caption("Modest improvement in test accuracy")
    col2.caption("MAPE unchanged — tuning alone was not enough")
    col3.caption("Overfitting gap reduced significantly through regularisation")

    st.divider()

    # ── Stage 3 ───────────────────────────────────────────
    st.subheader("Stage 3 — Feature Engineering")
    st.markdown(
        "The biggest improvement came from giving the model memory — adding lag features so it knows previous demand values, plus holiday flags, season, extreme heat and business day indicators."
    )

    st.markdown("**New features added:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
- `demand_1hr_ago` — demand from the previous hour
- `demand_24hr_ago` — same hour yesterday
- `demand_168hr_ago` — same hour last week
- `temp_1hr_ago` — temperature from previous hour
- `temp_change` — whether temperature is rising or falling
        """)
    with col2:
        st.markdown("""
- `is_holiday` — NSW public holiday flag
- `is_extreme_heat` — apparent temperature above 35°C
- `season` — summer, autumn, winter, spring
- `is_business_day` — combines weekend and holiday flags
        """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Test RMSE", "259 MW", delta="-504 MW", delta_color="inverse")
    col2.metric("MAPE", "2.79%", delta="-5.27%", delta_color="inverse")
    col3.metric("Overfit gap", "102 MW", delta="-524 MW", delta_color="inverse")
    col1.caption("Below industry benchmark — strong performance")
    col2.caption("Under 3% error — reliable hourly forecasting")
    col3.caption("Healthy gap — model generalises well to unseen data")

    st.image(
        "screenshots/01_predicted_vs_actual.png",
        caption="Stage 3 — Actual and predicted lines track closely throughout the test period.",
    )

    st.divider()

    # ── Summary table ─────────────────────────────────────
    st.subheader("Summary")
    summary = pd.DataFrame(
        {
            "Stage": [
                "Stage 1 — Baseline",
                "Stage 2 — Tuned",
                "Stage 3 — Feature Engineering",
            ],
            "Test RMSE (MW)": [812, 763, 259],
            "MAPE (%)": [8.06, 8.06, 2.79],
            "Overfit Gap (MW)": [626, 411, 102],
            "Features": [10, 10, 19],
        }
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)
