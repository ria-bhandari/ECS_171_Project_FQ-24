import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load data
data = pd.read_csv("../Data/final_data.csv")

# Convert WaterYear to a datetime index
data["Date"] = pd.to_datetime(data["WaterYear"].astype(str) + "-10-01")
data.set_index("Date", inplace=True)

# Get unique counties
unique_counties = data["County"].unique()

def create_lagged_features(data, target_col, lags=3):
    """
    Create lagged features for time-series forecasting.
    """
    for lag in range(1, lags + 1):
        data[f"Lag_{lag}"] = data[target_col].shift(lag)
    return data.dropna()  # Drop rows with NaN values after shifting

def forecast_future(data, model, steps=10):
    """
    Forecast future values using the Random Forest model and lagged features.
    """
    future_preds = []
    # Start with the last row of data
    last_known = data.iloc[-1][["Lag_1", "Lag_2", "Lag_3"]].values

    for _ in range(steps):
        # Convert last_known to DataFrame with matching feature names
        input_data = pd.DataFrame([last_known], columns=["Lag_1", "Lag_2", "Lag_3"])
        next_pred = model.predict(input_data)[0]  # Predict next value
        future_preds.append(next_pred)
        # Update lagged values for the next prediction step
        last_known = np.roll(last_known, -1)
        last_known[-1] = next_pred

    return future_preds

# Iterate through each county
for county_name in unique_counties:
    print(f"\nProcessing County: {county_name}")
    
    # Filter data for the current county
    county_data = data[data["County"] == county_name].copy()

    # Create lagged features
    county_data = create_lagged_features(county_data, target_col="TotalPrecipitation_inches", lags=3)

    # Check if there is enough data after creating lagged features
    if county_data.empty or len(county_data) < 10:
        print(f"Not enough data for {county_name}. Skipping...")
        continue

    # Define features (lagged values) and target
    X = county_data[["Lag_1", "Lag_2", "Lag_3"]]
    y = county_data["TotalPrecipitation_inches"]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate model performance
    test_predictions = rf_model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, test_predictions)
    rmse = mean_squared_error(y_test, test_predictions, squared=False)
    r2 = r2_score(y_test, test_predictions)

    # Print metrics
    print(f"Model Performance Metrics for {county_name}:")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"  R² Score: {r2:.2f}")

    # Generate forecasts for the next 80 years
    forecast_years = 80
    future_forecast = forecast_future(county_data, rf_model, steps=forecast_years)

    # Create future dates
    future_dates = pd.date_range(
        start=county_data.index[-1] + pd.DateOffset(years=1),
        periods=forecast_years,
        freq="YS-OCT"
    )

    # Combine into a DataFrame
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Precipitation": future_forecast
    }).set_index("Date")

    # Define output file path
    output_path = f"../County_Forecasts/{county_name}_RFforecast.txt"

    # Save the forecast to a text file
    with open(output_path, "w") as f:
        f.write(f"Precipitation Forecast for {county_name} County:\n")
        for date, value in forecast_df["Predicted_Precipitation"].items():
            f.write(f"{date.strftime('%Y-%m-%d')}: {value:.2f} inches\n")

    print(f"Forecast for {county_name} County saved to {output_path}")
