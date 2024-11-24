import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sklearn
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from scipy.fft import fft
from dateutil.relativedelta import relativedelta
import os

#Define a function that offsets all repeating WaterYear values
def duplicate_years(df):
    # Create a set to keep track of seen values and a dictionary for offsets
    seen = set()
    offsets = {}
    append = 0

    # Increment duplicates
    for idx, year in enumerate(df['WaterYear']):
        if year in seen:
            # Increment the year using the current offset for this value
            offsets[year] += 1
            df.at[idx, 'WaterYear'] = int(year + offsets[year])
        else:
            # First occurrence, initialize offset
            seen.add(year)
            offsets[year] = 0

    df['WaterYear'] = df['WaterYear'].astype('int')
    return df

# Load data
data = pd.read_csv('./Data/final_data_temp.csv')

# Convert WaterYear to date time format and set as index
data["Date"] = pd.to_datetime(data["WaterYear"].astype(str) + "-10-01")
data.set_index("Date", inplace=True)
grouped_data = data.groupby("StationName ")

# Ensure the forecast directory exists
#output_dir = "../Station_Forecasts"
#os.makedirs(output_dir, exist_ok=True)
new_stations = []

# Loop through each unique station and create a model - SARIMAX
for station in list(grouped_data):
    station_name = station[0]
    station = station[1]
    # Resample Station data to ensure consistent frequency and fill any missing dates
    station = duplicate_years(station)
    station.index = pd.to_datetime(station["WaterYear"].astype(str) + "-10-01")
    try:
        station = station.resample("YS-OCT").asfreq()
    except:
        print(station.to_string())

    # Append new stations array for future analysis
    new_stations.append(station)

    # Train SARIMAX model with reduced complexity for sparse data
    model = SARIMAX(
        station["TotalPrecipitation_inches"],
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarima_model = model.fit(disp=False)

    # Predict future precipitation values
    predict_years = 80
    pred = sarima_model.get_forecast(steps=predict_years)
    pred_vals = pred.predicted_mean
    pred_ci = pred.conf_int()

    # Create a forecast index based on the last known date and desired frequency
    pred_index = pd.date_range(
        start=station.index[-1] + pd.DateOffset(years=1),
        periods=predict_years,
        freq="YS-OCT",
    )

    # Create forecast series and assign the forecast index
    forecast_series = pd.Series(pred_vals.values, index=pred_index)

    # Adjust the confidence interval index to match forecast index
    forecast_ci = pred_ci.set_index(pred_index)

    # Combine forecast series and confidence interval
    rainfall_forecast_series = pd.concat([forecast_series, forecast_ci], axis=1)

    # Save forecast data to a text file
    with open(f"./Station_Forecasts/{station_name}_forecast.txt", "w") as f:
        f.write(f"Precipitation Forecast for {station_name} Station:\n")
        for row in rainfall_forecast_series.itertuples():
            # Check if row.Index is a datetime object
            if isinstance(row.Index, pd.Timestamp):
                f.write(
                    f"{row.Index.strftime('%Y-%m-%d')}: Prediction: {row[1]:.2f} inches, "
                    f"Lower {row[2]:.2f} inches, Upper {row[3]:.2f} inches\n"
                )
            else:
                f.write(
                    f"{row.Index}: Prediction: {row[1]:.2f} inches, "
                    f"Lower {row[2]:.2f} inches, Upper {row[3]:.2f} inches\n"
                )

    print(
        f"Forecast for {station_name} Station saved to ../Station_Forecasts/{station_name}_forecast.txt"
    )

'''
Note: Certain collection locations have inaccurate or flat-out wrong SARIMAX Predictions:

Ball Mountain (negative values)
Boulder Creek (Very high values)
Butte Lake (one negative value)
Clover Valley (very high values)
Gazelle Mountain (very high values)
Hogsback Road (Very high values)
Lights Creek (very high values)
Little Last Change (very high values)
Medicine Lake (a few negative values)
Mount Hough (very high values)
Shaffer Mountain (negatve values)
Swain Mountain (negative values)
Three Mile Valley (very high values)

There are also some where the MLE failed to converge
'''