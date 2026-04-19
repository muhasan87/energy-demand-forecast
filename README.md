# Energy Demand Forecasting Pipeline 🔋

![Python](https://img.shields.io/badge/Python-3.12-blue)
![AWS](https://img.shields.io/badge/AWS-Lambda%20%7C%20S3-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![MAPE](https://img.shields.io/badge/MAPE-2.79%25-brightgreen)

> An end-to-end machine learning pipeline that predicts hourly electricity demand 
> for New South Wales, Australia using real weather data and historical energy consumption.

---

## What this project does

This pipeline ingests real hourly weather data from the Open-Meteo API and historical 
electricity demand data from AEMO (Australian Energy Market Operator), engineers 
meaningful features, trains and evaluates multiple ML models, and serves live predictions 
through an interactive web application — automated through AWS cloud infrastructure and 
visualised in a Tableau Public dashboard.

**The core question:** Given the weather conditions and time of day, how much 
electricity will NSW need in the next hour?

---

## Tech stack

| Layer | Technology |
|---|---|
| Data ingestion | Open-Meteo API, AEMO, AWS Lambda, EventBridge |
| Raw storage | AWS S3 |
| Feature engineering | Python, Pandas |
| ML models | Scikit-learn, XGBoost |
| Web app | Streamlit |
| REST API | FastAPI |
| Workflow modelling | Camunda BPMN |
| Version control | Git, GitHub |

🔗 **Live API:** https://energy-demand-forecast-hgmm.onrender.com/docs
