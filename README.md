# Salary-Growth-Predictor
Analyzing the Impact of Inflation on Salaries Across Industries

## 📌 Overview

In an era of rising inflation and economic uncertainty, salary growth often struggles to keep pace with the cost of living. This project presents a robust, data-driven solution that **predicts future salary trends across industries**, incorporating both **job-specific variables** and **macroeconomic indicators**.
It blends **machine learning** with **economic forecasting** to deliver real, actionable insights into wage dynamics.

---

## 🎯 Problem Statement

Despite ongoing inflation, salary adjustments do not always follow suit—leading to decreased purchasing power and widening compensation gaps across industries.

**Objective:**
Predict future salary growth across industries based on:

* Job-specific features (experience, role, location, etc.)
* Macroeconomic indicators (inflation, GDP, unemployment rate, etc.)

---

## 📊 Data Sources

### 🔹 Synthetic Salary Dataset

* Simulates 1000+ employee profiles using BASE\_START\_SALARY and experience-based growth
* Adjusts for categorical factors like negotiation power, hot skills, and location
* Includes random noise for realism

### 🔹 Economic Data

* **USA**: Live data fetched via [FRED API](https://fred.stlouisfed.org/) for CPI, Fed Rate, GDP, and Unemployment
* **India**: Historical economic indicators (GDP, CPI) integrated via custom fetch functions

---

## 🧠 Why This Project Stands Out

* **Real vs Nominal Salary Analysis**: Adjusts for inflation to reflect true purchasing power
* **Comparative Global Perspective**: Analyzes the interplay of Indian and U.S. macroeconomic trends
* **Contextualized Salary Forecasting**: Goes beyond historical salary data by embedding economic reality into model features
* **Scalable & Reproducible**: All components are modular, well-documented, and ready for deployment

---

## 🧪 Exploratory Data Analysis

Key Questions Explored:

* How do salaries vary by role, industry, and region?
* What is the correlation between salary growth and experience?
* Are macroeconomic indicators like inflation and GDP influencing wages?

---

## 🛠️ Model Architecture

**Models Used:**

* 🔸 Random Forest
* 🔸 XGBoost
* 🔸 Gradient Boosting
* 🔸 Lasso Regression (best performing)

**Training Pipeline:**

* Log transformation of target salaries for scale handling
* One-hot encoding for categorical features
* Standardization of numeric inputs
* TimeSeriesSplit for time-aware cross-validation
* Hyperparameter tuning with GridSearchCV

**Evaluation Metrics:**

* Mean Absolute Error (MAE)
* R² Score

---

## 📈 Economic Forecasting

* Predicts inflation, GDP, and unemployment using Linear Regression
* Forecasted data merged with job-specific input to generate **future salary predictions**
* Handles missing future data gracefully by fallback mechanisms

---

## 🧾 Output

* Trained model: `salary_prediction_model_enhanced.joblib`
* Supporting datasets: `auxiliary_data_enhanced.joblib`
* Predictions exported to `.csv` for quick review

---

## 🧠 Key Takeaways

### ✅ Strengths:

* Integrated job + macroeconomic features
* Time-aware evaluation and forecasting
* High accuracy with model interpretability
* Extendable to other geographies or sectors

### ⚠️ Limitations:

* Simplified economic forecasting via linear models
* Sensitive to data quality and overfitting in ensemble models


