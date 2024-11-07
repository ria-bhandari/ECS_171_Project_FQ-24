import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

data = pd.read_csv("../Data/final_data.csv")

# Convert WaterYear to date time format
data["Date"] = pd.to_datetime(data["WaterYear"].astype(str) + "-10-01")
data.set_index("Date", inplace=True)

# Time series models for each county

# Get counties from the data
for county in data["County"]:

    county_data = data[data["County"] == county]

# Sort the index and drop any duplicate entries
data = data[~data.index.duplicated(keep="first")].sort_index()

# Set the frequency explicitly
# data = data.asfreq("AS-OCT")

# Train SARIMAX model

for county in data["County"]:

    model = SARIMAX(
        county_data["TotalPrecipitation_inches"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarima_model = model.fit(disp=False)

    # Predict future precipitation values
    predict_years = 20
    pred = sarima_model.get_forecast(steps=predict_years)
    pred_index = range(
        county_data["WaterYear"].iloc[-1] + 1,
        county_data["WaterYear"].iloc[-1] + 1 + predict_years,
    )
    pred_vals = pred.predicted_mean
    # Create a forecast index based on the DateTime index
    pred_index = pd.date_range(
        start=county_data.index[-1] + pd.DateOffset(years=1),
        periods=predict_years,
        freq="AS-OCT",
    )

    # Combine forecast values with the index
    rainfall_forecast_series = pd.Series(pred_vals.values, index=pred_index)

    with open(f"../County_Forecasts/{county}_forecast.txt", "w") as f:
        f.write(f"Precipitation Forecast for {county} County:\n")
        for date, value in rainfall_forecast_series.items():
            f.write(f"{date.strftime('%Y-%m-%d')}: {value:.2f} inches\n")

    print(
        f"Forecast for {county} County saved to county_forecasts/{county}_forecast.txt"
    )
