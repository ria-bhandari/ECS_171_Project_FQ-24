import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

# Load data
data = pd.read_csv("../Data/final_data.csv")

# Convert WaterYear to date time format and set as index
data["Date"] = pd.to_datetime(data["WaterYear"].astype(str) + "-10-01")
data.set_index("Date", inplace=True)

# Remove duplicate index entries and sort
data = data[~data.index.duplicated(keep="first")].sort_index()

# Set frequency by resampling to ensure consistency, filling missing dates if needed
data = data.resample("YS-OCT").asfreq()

# Loop through each unique county and create a model
for county in data["County"].unique():

    # Filter data for the current county
    county_data = data[data["County"] == county]

    # Resample county data to ensure consistent frequency and fill any missing dates
    county_data = county_data.resample("YS-OCT").asfreq()

    # Train SARIMAX model with reduced complexity for sparse data
    model = SARIMAX(
        county_data["TotalPrecipitation_inches"],
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
        start=county_data.index[-1] + pd.DateOffset(years=1),
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
    with open(f"../County_Forecasts/{county}_forecast.txt", "w") as f:
        f.write(f"Precipitation Forecast for {county} County:\n")
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
        f"Forecast for {county} County saved to ../County_Forecasts/{county}_forecast.txt"
    )
