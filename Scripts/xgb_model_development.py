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
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
'''
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
# Creating lag features for time-series data
def create_lag_features(data, lag_steps=1):
    for i in range(1, lag_steps + 1):
        data[f'lag_{i}'] = data["TotalPrecipitation_inches"].shift(i)
    return data

# Creating rolling mean for time-series data
def create_rolling_mean(data, window_size=3):
    data['rolling_mean'] = data["TotalPrecipitation_inches"].rolling(window=window_size).mean()
    return data

# Applying Fourier transformation for capturing seasonality
def apply_fourier_transform(data):
    values = data['TotalPrecipitation_inches'].values
    fourier_transform = fft(values)
    data['fourier_transform'] = np.abs(fourier_transform)
    return data

errors_list = []
predictions_list = []
for station in list(grouped_data):
    station_maes = []
    station_rmses = []

    station_name = station[0]
    station = station[1]
    # Resample Station data to ensure consistent frequency and fill any missing dates
    station = duplicate_years(station)
    station = station.dropna()
    station.index = pd.to_datetime(station["WaterYear"].astype(str) + "-10-01")
    try:
        station = station.resample("YS-OCT").asfreq()
    except:
        print(station.to_string())

    # Applying lag feature creation to the dataset
    station = create_lag_features(station, lag_steps = 3)
    # Applying rolling mean to the dataset
    station = create_rolling_mean(station, window_size = 5)
    station = apply_fourier_transform(station)

    #X1 = station[['lag_1', 'lag_2', 'lag_3']]
    X2 = station['rolling_mean'] #We picked the rolling mean because precipitation data is susceptible to unnecessary trends (record high/low rainfall years)
    #X3 = station['fourier_transform']
    y = station['TotalPrecipitation_inches']

    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, shuffle=False)

    param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0]
    }
    grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    # Training the XGBoost model
    xgb_model = XGBRegressor(**best_params)
    xgb_model.fit(X_train, y_train)

    # Evaluating the XGBoost model on the testing set
    predictions = xgb_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    errors_list.append([mae, rmse])

    plt.plot(y_train.values, label='Training', color = 'green', alpha = 0.8)

    # Plot actual values
    plt.plot(y_test.values, label='Actual', color='blue', alpha=0.8)

    # Plot predicted values
    plt.plot(predictions, label='Prediction', color='orange', alpha=0.8)

    plt.xlabel('Time')
    plt.ylabel('Rainfall')
    plt.title('Rainfall Predictions')
    plt.legend()
    plt.show()

new_df = pd.DataFrame(errors_list, columns = ['mae', 'rmse'])
print(new_df)