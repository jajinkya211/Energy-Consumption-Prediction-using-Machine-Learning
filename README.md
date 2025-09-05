# Energy Consumption Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Scikit--learn%20%7C%20TensorFlow-orange.svg)

## Project Overview

This project provides a complete walkthrough of a time series forecasting problem, aiming to predict hourly energy consumption. Accurate demand forecasting is critical for energy providers to maintain grid stability, manage resources efficiently, and prevent outages. This repository demonstrates how to leverage historical data to build, train, and compare several machine learning and deep learning models for this purpose.

The analysis is performed on the **Dayton hourly energy consumption dataset**, which contains a single time series of energy usage in Megawatts (MW).

## Project Workflow

1.  **Exploratory Data Analysis (EDA):** The dataset is loaded and visualized to identify key characteristics, including long-term trends, yearly seasonality (peaks in summer/winter), and daily patterns.
2.  **Feature Engineering:** To prepare the data for modeling, new features are created from the timestamp, such as the hour, day of the week, month, and year. Lag features (consumption from previous hours and days) are also engineered to help the models learn from recent history.
3.  **Model Development:** Three distinct models are implemented to tackle the forecasting problem:
    * **Linear Regression:** A simple statistical model used to establish a performance baseline.
    * **Random Forest Regressor:** A powerful ensemble learning model that can capture complex non-linear relationships between features.
    * **Long Short-Term Memory (LSTM) Network:** A type of recurrent neural network (RNN) specifically designed to learn from sequential data, making it ideal for time series forecasting.
4.  **Model Evaluation:** The models are trained on a historical data segment and evaluated on a separate, more recent test set. Their performance is quantitatively compared using standard regression metrics like Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared ($R^2$).

## Technologies & Libraries

* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Time Series Analysis:** Statsmodels
* **Machine Learning:** Scikit-learn
* **Deep Learning:** TensorFlow (with Keras)
* **Environment:** Jupyter Notebook

## Results

Both the Random Forest and LSTM models demonstrated a strong ability to predict energy consumption, significantly outperforming the Linear Regression baseline. The R-squared value of 0.95 indicates that these models can explain 95% of the variance in the test data.

Feature importance analysis from the Random Forest model highlighted that the most influential predictors were the consumption from the previous hour (`lag_1`), the hour of the day, and the consumption from the same hour on the previous day (`lag_24`).

**Download the data:**
Obtain the "Hourly Energy Consumption" dataset from [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption). 
