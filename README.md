# Seoul-Bike-Sharing-Demand-Prediction
Comparison of classical machine learning and deep learning models for bike demand prediction

### Team Members
Sara Shahzad

## Abstract
This project, using 2 classical machine learning models **(Ridge Regresion and Random Forest)** and a deep learning **Neural Networks** model, predicts the hourly count of rented bikes in Seoul using temporal and weather related features. Comprehensive evaluation demonstrates the comparative performance of classical versus deep learning approaches. Statistical tests confirm the significance of differences in model performance.

## Introduction
Bike sharing has become an essential component of urban transportation. Accurate prediction of bike demand can optimize resource allocation and improve service quality. This project aims to predict the number of rented bikes within Seoul; South Korea using a combination of temporal (date, day, month, weekday) and environmental features (temperature, humidity, wind speed, etc.) by applying both classical machine learning and deep learning models.

### Objectives
* Implement and tune classical ML models.
* Design and train deep neural networks.
* Compare model performance using multiple metrics.
* Conduct statistical significance testing for model comparison.

## Dataset Description
**Source:** https://www.kaggle.com/datasets/saurabhshahane/seoul-bike-sharing-demand-prediction

**Size:** 8760 rows, 12+ columns

**Features:** Date, Hour, Temperature, Humidity, Wind speed, Visibility, Dew point, Solar radiation, Rainfall, Snowfall

**Categorical features:** Seasons, Holiday, Functioning Day

**Target:** Rented Bike Count

**Preprocessing:**
Converted Date to day, month, weekday
One-hot encoded categorical variables
Scaled features for ML and deep learning models
Train/validation/test split: 70% / 15% / 15%

## Methodology
### Classical Machine Learning
**1. Ridge Regression**
Feature scaling applied

Hyperparameter tuning done using GridSearchCV (alpha values)

5-fold cross-validation for robust evaluation

**2. Random Forest Regressor**

Hyperparameter tuning done using RandomizedSearchCV

Parameters tuned: n_estimators, max_depth, min_samples_split, min_samples_leaf

### Deep Learning

Architecture: Fully connected feedforward neural network with 3 hidden layers (128, 64, 32 neurons)

Regularization: Dropout (0.3), BatchNormalization

Training: Adam optimizer, MSE loss, EarlyStopping, ReduceLROnPlateau

Validation: Used validation set for early stopping and learning rate adjustment

### Evaluation & Comparison

**Metrics:** RMSE, MAE, R²

**Statistical Significance Testing:** Paired t-test between models’ squared errors

## Results and Analysis

