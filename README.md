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

**Size:** 8760 rows, 14 columns

**Features:** Date, Hour, Temperature, Humidity, Wind speed, Visibility, Dew point, Solar radiation, Rainfall, Snowfall

**Categorical features:** Seasons, Holiday, Functioning Day

**Target:** Rented Bike Count

**Preprocessing:**
Converted Date to day, month, weekday
One-hot encoded categorical variables
Scaled features for ML and deep learning models

Data split:
|  | Training | Validation | Testing | 
|-------|------|-----|-----|
|Percentage|70%|15%|15%|


## Methodology
### Classical Machine Learning
**1. Ridge Regression**
Linear regression with l2 norm

Feature scaling applied

Hyperparameter tuning done using GridSearchCV (alpha values)

Regularization parameter alpha tuned using 5-fold cross-validation

**2. Random Forest Regressor**
Ensemble-based non-linear model

Hyperparameter tuning done using RandomizedSearchCV

Parameters tuned: n_estimators, max_depth, min_samples_split, min_samples_leaf

### Deep Learning

Architecture: Fully connected neural network with 3 hidden layers (128, 64, 32 neurons) all using ReLu axitivation

Regularization: Dropout (0.3), BatchNormalization

Training: Adam optimizer, MSE loss, EarlyStopping, ReduceLROnPlateau

Validation: Used validation set for early stopping and learning rate adjustment

### Evaluation & Comparison

**Metrics**
* Root Mean Squared Error (RMSE)

* Mean Absolute Error (MAE)

* Coefficient of Determination (R²)

**Statistical Significance Testing**
* Paired t-test applied on squared prediction errors

* Ensures fair comparison using the same test set

* Significance threshold: p < 0.05

## Results and Analysis
| Model | RMSE | MAE | R^2 |
|-------|------|-----|-----|
| Ridge Regression | 430.905322 | 324.345614 | 0.541045 |
| Random Forest | 175.238297 | 99.273702 | 0.924096 |
| Neural Network | 223.076010 | 134.429321 | 0.876998 |

Ridge Regression underperformed due to its linear nature, limiting its ability to capture complex non-linear relationships.

Random Forest achieved strong performance by modeling non-linear interactions between weather and temporal features.

The Deep Learning model achieved the much better results, benefiting from feature scaling and regularization.

Paired t-tests confirmed that performance differences between Ridge and non-linear models were statistically significant (p < 0.05).

### Visualization of Results

<img width="640" height="627" alt="histo1" src="https://github.com/user-attachments/assets/ebdd8dfd-803b-4dbc-91c9-84e7b9ff19f1" />

<img width="638" height="642" alt="histo2" src="https://github.com/user-attachments/assets/78c872d6-2490-4ed5-b777-85175864b0cd" />


<img width="677" height="510" alt="loss curve" src="https://github.com/user-attachments/assets/f2f6088e-35a0-44cd-8412-7a7dbaaa39b0" />

### Statistical Significance Testing

<img width="396" height="48" alt="stat cal" src="https://github.com/user-attachments/assets/2cbb827c-e628-40b1-bdef-d9af52a9d94e" />

These results indicate that performance differences are unlikely to be due to random chance, validating the comparative analysis.

### Business Impact Analysis

Accurate bike demand prediction has direct operational and economic benefits:

* Optimized Bike Redistribution: Reduces shortages and oversupply at stations, improving user satisfaction.

* Cost Reduction: Better forecasts lower transportation and maintenance costs.

* Scalable Decision Support: Non-linear models enable city planners to adapt to seasonal and weather-driven demand changes.

* Service Reliability: Improved availability during peak hours enhances public trust in bike-sharing systems.

## Conclusion
This project demonstrates that non-linear models, particularly Random Forests and Neural Networks, significantly outperform linear regression for bike demand prediction. Deep learning offers competitive performance, especially when proper regularization and training strategies are applied. The results highlight the importance of model selection based on data complexity in real world regression problems.

## Future Work
* Explore external data sources (events, traffic, holidays)
* Deploy the best model as a web-based prediction service

## References
Kaggle: Seoul Bike Sharing Demand Prediction Dataset: https://www.kaggle.com/datasets/saurabhshahane/seoul-bike-sharing-demand-prediction
