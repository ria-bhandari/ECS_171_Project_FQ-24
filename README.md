# ECS_171_Project_FQ-24

This project aims to predict rainfall in California using three machine learning models: **XGBoost**, **SARIMA**, and **Random Forest**. 


## Project Workflow

1. **Data Preprocessing**:  
   - Clean and preprocess the input datasets (`final_data.csv` and `Rainfall_Dataset.csv`) using scripts in the `Scripts` directory.
   - Feature engineering and exploratory data analysis are conducted to prepare the data for modeling.

2. **Model Development**:  
   - Train and evaluate three models:
     - **XGBoost**: Developed and evaluated in `xgb_model.ipynb`.
     - **SARIMA**: Implemented in `time_series_model.py`.
     - **Random Forest**: Code integrated in the respective Python scripts (3 different files).

3. **Forecasting**:  
   - Generate rainfall forecasts at the **county level** (`County_Forecasts/`) and the **station level** (`Station_Forecasts/`).

4. **Visualization and Metrics**:  
   - Generate visualizations and calculate evaluation metrics in notebooks like `barplot_metrics.ipynb` and `metrics_df.ipynb`.

## Models Used

### XGBoost
- A powerful gradient boosting algorithm designed for structured data. Used to generate accurate forecasts at both county and station levels.

### SARIMA
- A statistical time series forecasting method that extends ARIMA by accounting for seasonality.

### Random Forest
- An ensemble learning method based on decision trees. Useful for making robust predictions on complex datasets.

## Installation and Usage

### Prerequisites
Ensure you have Python installed. Clone the repository and navigate to the project directory.

### Install Dependencies
Install required Python libraries using `requirements.txt`:

```bash
pip install -r requirements.txt
```
### Run the Project

1. **Preprocessing and Training**:
   - Use the Jupyter notebooks in the `Scripts/` directory to preprocess data and train the models.
2. **Forecasting**:
   - Run the `.py` files in the `Scripts/` directory to generate forecasts for counties and stations using different models.

### Results

The forecast results are saved in the `County_Forecasts/` and `Station_Forecasts/` directories as text files.







