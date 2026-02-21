DEMAND FORECASTING IN RETAIL 

Final Year Project (FYP)

Student Name: Loo Wei Meng

Student ID: 1211105911

Programme: Bachelor of Information Technology (Honours) Business Intelligence and Analytics

University: Multimedia University (Melaka Campus)

Supervisor: Assoc. Prof. Ts. Dr. Lew Sook Ling

---------------------------------------------------------------------------------------------------------------------------------------
1. Project Overview

This repository contains the complete implementation of the Final Year Project titled: 

DEMAND FORECASTING IN RETAIL 

The objective of this project is to develop and compare traditional time series models and advanced machine learning models for retail demand forecasting.

Two publicly available datasets obtained from the Kaggle platform are used in this project: 
- Walmart Dataset
- Adidas Sales Dataset

The project includes:
- Data preprocessing and cleaning
- Feature engineering 
- Exploratory Data Analysis (EDA)
- Outlier detection and treatment (IQR + Winsorization)
- Traditional forecasting models (ARIMA, SARIMA)
- Machine learning models
- Ensemble models
- Model evaluation and comparison

All implementations are developed using Python.

------------------------------------------------------------------------------------------------------------------------------
2. Datasets

Go to the following Kaggle link to download the original dataset. The folder in GitHub also provides the original dataset for each dataset.
- Walmart Dataset

Link to Download: https://www.kaggle.com/datasets/yasserh/walmart-dataset

- Adidas Sales Dataset

Link to Download: https://www.kaggle.com/datasets/heemalichaudhari/adidas-sales-dataset

------------------------------------------------------------------------------------------------------------------------------
3. Tools & Environment

Required Software: 
- Python Version Used - Python 3.12.7 (packaged by Anaconda, Inc.)

Link to Download: https://www.python.org/downloads/

- Anaconda Navigator (Recommended)

Link to Download: https://www.anaconda.com/download

Execution platform:
- Jupyter Notebook

---------------------------------------------------------------------------------------------------------------------------------------
4. Required Python Libraries

Install using Anaconda Prompt or Command Prompt:
- numpy: For numerical computations and array handling.
- pandas: For data manipulation and CSV/Excel analysis.
- matplotlib & seaborn: For exploratory data analysis and result visualization.
- scipy: For statistical analysis and Winsorization.
- scikit-learn: For machine learning models (Linear, Ridge, RF) and evaluation metrics.
- statsmodels & pmdarima: For traditional time series models (ARIMA, SARIMA).
- xgboost & lightgbm: For advanced gradient boosting ensemble models.

---------------------------------------------------------------------------------------------------------------------------------------
5. Execution Instructions

Step 1 – Launch Environment
- Open Anaconda Navigator
- Launch Jupyter Notebook
- Navigate to the project folder containing the .ipynb files.

Step 2 – Run Notebooks Sequentially
For each dataset, the workflow is structured as follows:

1️. Data Formatting (Dataset 2: Adidas Sales Dataset only) 
- Convert Excel to CSV

2️. Preprocessing & EDA

Includes:
- Data cleaning
- Datetime conversion
- Temporal feature extraction (Year, Quarter, Month, Week, Day)
- Feature engineering:
-- Lag features
-- Rolling mean and rolling standard deviation
-- Seasonal classification
- Outlier detection using IQR
-  Winsorization treatment
- Categorical Variables Encoding
- Correlation analysis
- Save the preprocessed dataset

3️. Traditional Time Series Models
- Weekly aggregation
- Train-test split (chronological split)
- ARIMA
- SARIMA (auto parameter tuning)

Evaluation metrics:
- MSE
- RMSE
- MAE
- R²

Residual and forecast visualisation included.

4️. Machine Learning Models

Models implemented:
- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting Machine (GBM)
- XGBoost
- LightGBM

Evaluation metrics:
- MSE
- RMSE
- MAE
- R²

Residual and forecast visualisation included.

5. Ensemble Models:
- RF-XGBoost-LR (Stacking)
- RF-LightGBM (Voting)

Evaluation metrics:
- MSE
- RMSE
- MAE
- R²

Residual and forecast visualisation included.

6. Advanced techniques:
- TimeSeriesSplit cross-validation
- RandomizedSearchCV hyperparameter tuning
- Stacking Ensemble
- Voting Ensemble
- Train vs Test performance comparison
- Feature importance analysis

---------------------------------------------------------------------------------------------------------------------------------------
6. Reproducibility

To reproduce results:
- Install required software
- Install required libraries
- Place dataset files in the project folder
- Run notebooks sequentially

All outputs (tables, metrics, and visualisations) will be regenerated automatically.

---------------------------------------------------------------------------------------------------------------------------------------
7. Notes
- All datasets are publicly available.
- No external database configuration required.
- No private APIs or credentials required.
- All results are fully reproducible from the provided notebooks. 
