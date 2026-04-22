#!/bin/bash

# Update energy data from S3
echo "Starting energy data update - $(date)"

cd /home/ubuntu/energy-demand-forecast
source venv/bin/activate

# Sync new AEMO files from S3
aws s3 sync s3://energy-forecast-nsw-muhasan/aemo/ data/raw/AEMO/

# Rebuild energy.csv
python src/ingestion/fetch_energy.py

echo "Energy data update complete - $(date)"