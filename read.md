# 🏠 Explainable House Price Prediction using CatBoost and SHAP

## Project Overview

This project predicts house prices using the CatBoost Regressor algorithm and explains the model's predictions using SHAP (SHapley Additive Explanations).

A Flask web application allows users to enter house details, receive a predicted house price, and understand which features influenced the prediction.

The project combines **Machine Learning, Explainable AI (XAI), and Web Application Development**.

---

## 🎯 Objectives

- Predict house prices using a machine learning regression model.
- Build a user-friendly Flask web application.
- Explain individual predictions using SHAP.
- Identify the most influential features affecting house prices.
- Provide both global and local model explanations.

---

## ✨ Features

- 🏠 House Price Prediction
- 🤖 CatBoost Regression Model
- 🧠 SHAP Explainable AI
- 📊 SHAP Summary Plot
- 📈 SHAP Feature Importance Plot
- 📉 SHAP Local Waterfall Plot
- 💡 Human-readable AI Explanation
- 🌐 Flask Web Application
- ✅ SHAP Prediction Verification

---

## 🛠️ Technologies Used

- Python
- CatBoost
- SHAP
- Flask
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Dataset

The project uses a House Price Dataset containing information about
residential properties.

The dataset was cleaned and preprocessed before training the machine
learning model. The cleaned dataset used for this project is included
in this repository.

### Dataset Features

- Area
- Bedrooms
- Bathrooms
- Floors
- Balcony
- Facing
- Area Type

### Target Variable

- House Price

## Dataset Source

The original dataset was obtained from Kaggle.

A cleaned and preprocessed version of the dataset used for training
is included in this repository as `house_price_prediction.csv`.

---
## 🤖 Machine Learning Model

The project uses a **CatBoost Regressor** for house price prediction.

CatBoost is a gradient boosting algorithm that is particularly useful for datasets containing both numerical and categorical features.

The trained model is saved as:

```text
catboost_model.pkl
Additional model resources:
- `feature_names.pkl` — stores the feature names used by the model.
- `background_data.pkl` — background/reference data used for SHAP explainability.

## Evaluation Metrics:
RMSE: 1.847353534410137
R2 Score: 0.6245846313057173
MAE: 1.3057545052877413
## Author

**Yusra Fateen**  
BE Artificial Intelligence & Machine Learning