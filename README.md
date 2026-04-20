# Ford Car Price Prediction

A practical machine learning project to estimate used Ford car prices from structured car attributes.

This project is built as a learning-first notebook workflow, starting with data exploration and ending with linear regression modeling using different encoding strategies.

## Project Goal

Predict the car price using available features such as:
- model
- year
- transmission
- mileage
- fuelType
- tax
- mpg
- engineSize

## Project Files

- Ford_Car_Price.ipynb: end-to-end EDA, preprocessing, encoding, scaling, training, and evaluation
- ford.csv: dataset used for training and analysis

## Workflow Summary

### 1) Data Loading and Basic Checks

- Loaded dataset with pandas
- Checked shape and column types
- Reviewed category distributions (fuel type, transmission, model)
- Checked missing values and duplicate rows

### 2) Exploratory Data Analysis (EDA)

Used multiple plots to understand the data:
- Histogram + KDE for price distribution
- Correlation heatmap for numeric features
- Boxplots for price vs categorical features (year, transmission, fuelType, model)
- Scatter/regression plots for price relationships (mileage, tax)
- Pairplot for quick multi-feature relationship view

### 3) Feature/Target Split

- Features (X): all columns except price
- Target (y): price

### 4) Encoding Approaches Compared

Two strategies were tested:

1. One-hot encoding (with drop_first=True)
2. Label encoding (for learning and comparison)

### 5) Scaling

- Standardization was applied to numeric input features
- Target variable (price) was kept in original units

### 6) Model Training

- Model used: Linear Regression (scikit-learn)
- Train-test split: 80-20

## Results

### One-Hot Encoded Features + Linear Regression

- R2 score: 0.8458422267
- Adjusted R2 score: 0.8454982205

### Label Encoded Features + Linear Regression

- R2 score: 0.7365884289

## Interpretation

- The one-hot encoded version performs better than the label-encoded version.
- This is expected because model, transmission, and fuelType are nominal categories, and one-hot encoding avoids introducing artificial order.
- The close values of R2 and adjusted R2 indicate that the feature set is useful and not heavily inflated by irrelevant predictors.

## Key Learnings

- For nominal categorical variables, one-hot encoding is generally a better default for linear models.
- Always split data before fitting transformations in strict production pipelines to avoid leakage.
- R2 explains variance captured, while MAE/RMSE should be used to understand error in real price units.

## How to Run

1. Open Ford_Car_Price.ipynb
2. Run cells top to bottom
3. Review EDA plots and model metrics

## Next Improvements

- Add MAE and RMSE comparison for both encoding methods
- Try regularized models (Ridge, Lasso)
- Add tree-based models (Random Forest, XGBoost) for benchmark comparison
- Add cross-validation for more stable performance estimates

---

This project focuses on building strong ML fundamentals through hands-on experimentation and metric-driven comparison.