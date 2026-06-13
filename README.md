# Cab Fare Prediction

Predicts cab fare amounts using machine learning based on pickup/drop-off locations, passenger count, and trip distance.

## Overview
This project builds a regression model to predict cab fare for a ride based on historical trip data, following the CRISP-DM methodology (Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation).

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## Dataset
- 16,067 training observations with 7 variables: fare_amount, pickup/dropoff coordinates, passenger_count, pickup_datetime
- 9,514 test observations (without fare_amount)

## Workflow
1. **Data Cleaning** - handled missing values using KNN imputation, removed the pickup_datetime column (incomplete time information)
2. **Outlier Treatment** - removed negative fares, unrealistic passenger counts, and out-of-range location coordinates
3. **Feature Engineering** - created a `distance` feature using the Haversine formula to calculate trip distance from pickup/dropoff coordinates
4. **Correlation Analysis** - confirmed distance as a strong predictor of fare amount
5. **Model Training** - compared Random Forest and Linear Regression
6. **Prediction** - generated fare predictions on the test dataset and exported results to CSV

## Results
| Model | MAE (%) | Accuracy (%) |
|---|---|---|
| Random Forest | 20.21 | 79.79 |
| Linear Regression | 28.17 | 71.83 |

Random Forest was selected as the final model due to higher accuracy and lower error.

## Key Files
- `cab_fare_prediction.ipynb` - main notebook with full pipeline
- `train.csv` - training dataset
- `test.csv` - test dataset
- `Predicted_Values.csv` - final fare predictions on test data

## Future Work
Exploring regularized regression techniques (e.g., Ridge Regression) to address potential overfitting with additional features.
