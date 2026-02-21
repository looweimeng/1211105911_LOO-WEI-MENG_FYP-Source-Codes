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
1. Walmart Dataset
2. Adidas Sales Dataset

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

---------------------------------------------------------------------------------------------------------------------------------------
2. Repository Structure
1211105911_LOO-WEI-MENG_FYP-Source-Codes/
│
├── Dataset 1 Walmart Dataset Finalised Python Codes/
│   ├── 1. Dataset 1 Walmart Dataset Preprocessing and EDA.ipynb
│   ├── 2. Dataset 1 Walmart Dataset Modelling (Traditional Models).ipynb
│   ├── 3. Dataset 1 Walmart Dataset Modelling (ML Models).ipynb
│   ├── Walmart.csv
│   └── walmart_preprocessed.csv
│
├── Dataset 2 Adidas Sales Dataset Finalised Python Codes/
│   ├── 1. Dataset 2 Adidas Sales Dataset Data Format.ipynb
│   ├── 2. Dataset 2 Adidas Sales Dataset Preprocessing and EDA.ipynb
│   ├── 3. Dataset 2 Adidas Sales Dataset Modelling (Traditional Models).ipynb
│   ├── 4. Dataset 2 Adidas Sales Dataset Modelling (ML Models).ipynb
│   ├── Adidas US Sales Dataset.xlsx
│   ├── Adidas.csv
│   └── adidas_preprocessed.csv
│
├── Readme.txt
|
└── README.md

Each dataset folder contains:
- Raw dataset
- Preprocessed dataset
- Preprocessing notebooks
- Traditional statistical modelling notebooks
- Machine learning modelling notebooks

---------------------------------------------------------------------------------------------------------------------------------------
3. Datasets
Go to the following Kaggle link to download the original dataset. The folder in GitHub also provides the original dataset for each dataset.
1. Walmart Dataset: https://www.kaggle.com/datasets/yasserh/walmart-dataset
2. Adidas Sales Dataset: https://www.kaggle.com/datasets/heemalichaudhari/adidas-sales-dataset

---------------------------------------------------------------------------------------------------------------------------------------
4. Tools & Environment
Required Software: 
- Python Version Used - Python 3.13.2
- Anaconda Navigator (Recommended)

Execution platform:
Jupyter Notebook

---------------------------------------------------------------------------------------------------------------------------------------
5. Required Python Libraries
Install using Anaconda Prompt or Command Prompt:
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels pmdarima xgboost lightgbm

---------------------------------------------------------------------------------------------------------------------------------------
6. Execution Instructions
Step 1 – Launch Environment
1. Open Anaconda Navigator
2. Launch Jupyter Notebook
3. Navigate to the project folder

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
  Lag features
	Rolling mean and rolling standard deviation
  Seasonal classification
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

5. Ensemble Models:
- RF-XGBoost-LR (Stacking)
- RF-LightGBM (Voting)

6. Advanced techniques:
- TimeSeriesSplit cross-validation
- RandomizedSearchCV hyperparameter tuning
- Stacking Ensemble
- Voting Ensemble
- Train vs Test performance comparison
- Feature importance analysis

---------------------------------------------------------------------------------------------------------------------------------------
7. Reproducibility
To reproduce results:
1. Install required software
2. Install required libraries
3. Place dataset files in the project folder
4. Run notebooks sequentially

All outputs (tables, metrics, and visualisations) will be regenerated automatically.

---------------------------------------------------------------------------------------------------------------------------------------
8. Notes
- All datasets are publicly available.
- No external database configuration required.
- No private APIs or credentials required.
- All results are fully reproducible from the provided notebooks. 
