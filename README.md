# Bengaluru House Price Prediction

An end-to-end machine learning project for predicting house prices in Bengaluru using **Python, pandas, NumPy, scikit-learn, Joblib, and Flask**.

The project covers exploratory data analysis, feature engineering, data preprocessing, model experimentation, Random Forest regression, evaluation, model serialization, and deployment through a web application.

---

## 📌 Project Overview

This project uses the Bengaluru House Data dataset to build a machine learning model for predicting residential property prices based on features such as:

- Location
- Area type
- Total square footage
- Number of bathrooms
- Number of balconies
- Number of bedrooms (BHK)

The project follows a complete machine learning workflow:

**Data → EDA → Feature Engineering → Preprocessing → Model Training → Evaluation → Model Serialization → Web Application**

---

## 🧠 Key Features

- Exploratory Data Analysis (EDA)
- Missing-value analysis
- Data cleaning and preprocessing
- Feature engineering
- Price-per-square-foot analysis
- BHK extraction from property-size information
- Outlier filtering using price-per-square-foot
- Log transformation of the target variable
- Numerical feature scaling
- Categorical feature encoding
- Regression model experimentation
- Random Forest regression
- Model evaluation using RMSE and R²
- Model serialization using Joblib
- Flask-based prediction application

---

## 📊 Dataset

The project uses the **Bengaluru House Data** dataset.

The original dataset contains:

- **13,320 observations**
- **9 columns**

### Main Features

| Feature | Description |
|---|---|
| `area_type` | Type/category of the property area |
| `availability` | Property availability information |
| `location` | Location of the property |
| `size` | Property size information |
| `society` | Society/project information |
| `total_sqft` | Total area in square feet |
| `bath` | Number of bathrooms |
| `balcony` | Number of balconies |
| `price` | Property price |

---

## 🔎 Exploratory Data Analysis

The notebook explores the dataset to understand its structure, distributions, missing values, and relationships between variables.

The analysis includes:

- Dataset structure and dimensions
- Missing-value analysis
- Price distribution
- Log-price distribution
- Feature relationships
- Correlation analysis
- Price-per-square-foot analysis
- Examination of property sizes and BHK values

---

## ⚙️ Feature Engineering

### Price per Square Foot

A price-per-square-foot feature is calculated as:

```text
price_per_sqft = (price × 100000) / total_sqft
