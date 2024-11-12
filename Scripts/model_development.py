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
    pred_ci = pred.conf_int()

    # Create a forecast index based on the DateTime index
    pred_index = pd.date_range(
        start=county_data.index[-1] + pd.DateOffset(years=1),
        periods=predict_years,
        freq="AS-OCT",
    )

    #Creates a pd series with the dates as the index
    forecast_series = pd.Series(pred_vals.values, index=pred_index)

    #Change the indicies of the CI dataframe to the dates
    forecast_ci = pred_ci.rename(index=dict(zip(range(603,623),pred_index)))

    #combine the forecast series and confidence interval
    rainfall_forecast_series = pd.concat([forecast_series, forecast_ci], axis=1)

    with open(f"../County_Forecasts/{county}_forecast.txt", "w") as f:
        f.write(f"Precipitation Forecast for {county} County:\n")
        for row in rainfall_forecast_series.itertuples():
            f.write(f"{row[0].strftime('%Y-%m-%d')}: Prediction: {row[1]:.2f} inches, Lower {row[2]:.2f} inches, Upper: {row[3]:.2f} inches\n")

    print(
        f"Forecast for {county} County saved to county_forecasts/{county}_forecast.txt"
    )
