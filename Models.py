# Load necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pygam import LinearGAM, s
from xgboost import XGBRegressor

np.random.seed(420)  # For reproducibility


# ===============================================================
# Section 1: Data Loading and Preparation
# ===============================================================
print("--- 1. Loading and Preparing Data ---")

# Load the dataset from the .xlsx file
file_path = "data/concrete_data.xlsx"
try:
    concrete_df = pd.read_excel(f"{file_path}", engine='openpyxl')
except FileNotFoundError:
    print("Error: File not found. Please ensure the dataset is in the correct path.")
    exit()

# Assign cleaner column names
concrete_df.columns = ["cement", "slag", "ash", "water",
                       "superplasticizer", "coarse_agg", "fine_agg",
                       "age", "strength"]

# Display the first few rows and data types
print("Data Head:")
print(concrete_df.head())
print("\nData Info:")
concrete_df.info()

# Check for missing values
if (concrete_df.isnull().sum().sum() == 0):
    print("\nData is complete. No missing values found.\n")
else:
    print("\nWarning: Missing values detected.\n")


# ===============================================================
# Section 2: Exploratory Data Analysis (EDA) & Plot Generation
# ===============================================================
print("--- 2. Performing EDA and Generating Plots ---")

# 2.1 Correlation Matrix Heatmap (for internal analysis)
plt.figure(figsize=(10, 8))
corr_matrix = concrete_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix of Concrete Ingredients")
plt.xticks(rotation=30)
plt.savefig('plots/correlation_heatmap_py.png', dpi=150)
plt.close()
print("Saved 'correlation_heatmap_py.png' to working directory.\n")

# 2.2 Scatter Plot
plt.figure(figsize=(8, 6))
sns.regplot(x="age", y="strength", data=concrete_df, lowess=True,
            scatter_kws={'alpha': 0.4, 'color': 'steelblue'},
            line_kws={'color': 'red', 'linewidth': 2})
plt.title("Strength vs. Age Shows Strong Non-Linearity")
plt.xlabel("Age (days)")
plt.ylabel("Compressive Strength (MPa)")
plt.grid(True)
plt.savefig('plots/strength_vs_age_py.png', dpi=150)
plt.close()
print("Saved 'strength_vs_age_py.png' to working directory.\n")


# ===============================================================
# Section 3: Data Splitting
# ===============================================================
print("--- 3. Splitting Data into Training (80%) and Test (20%) Sets ---")
X = concrete_df.drop(columns="strength", axis=1)
y = concrete_df["strength"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)
print(f"Training set size: {X_train.shape[0]} observations")
print(f"Test set size: {X_test.shape[0]} observations\n")


# ===============================================================
# Section 4: Model Training and Evaluation
# ===============================================================

# --- 4.1 Model 1: Multiple Linear Regression (MLR - Baseline) ---
print("--- Training Model 1: Multiple Linear Regression ---")
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

# Evaluate MLR on test data
pred_mlr = mlr_model.predict(X_test)
rmse_mlr = np.sqrt(mean_squared_error(y_test, pred_mlr))
r2_mlr = r2_score(y_test, pred_mlr)
print(f"Linear Model Test RMSE: {rmse_mlr:.2f} MPa\n")


# --- 4.2 Model 2: Lasso Regression ---
print("--- Training Model 2: Lasso Regression ---")

# LassoCV automatically finds the best alpha (lambda) using cross-validation
lasso_cv = LassoCV(cv=10, random_state=420)
lasso_cv.fit(X_train, y_train)
print(f"Best Alpha (Lambda) from 10-fold CV: {lasso_cv.alpha_:.3f}")

# Evaluate Lasso on test data
pred_lasso = lasso_cv.predict(X_test)
rmse_lasso = np.sqrt(mean_squared_error(y_test, pred_lasso))
r2_lasso = r2_score(y_test, pred_lasso)
print(f"Lasso Model Test RMSE: {rmse_lasso:.2f} MPa\n")


# --- 4.3 Model 3: Generalized Additive Model (GAM) ---
print("--- Training Model 3: Generalized Additive Model ---")

# Fit a GAM with smoothing splines for each predictor
gam_model = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7))
gam_model.fit(X_train, y_train)
print("GAM Model Summary:")
gam_model.summary()

# Generate and save the component plots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()
for i, term in enumerate(gam_model.terms):
    if term.isintercept:
        continue
    XX = gam_model.generate_X_grid(term=i)
    pdep, confi = gam_model.partial_dependence(term=i, X=XX, width=0.95)
    axes[i].plot(XX[:, term.feature], pdep)
    axes[i].plot(XX[:, term.feature], confi, c='r', ls='--')
    axes[i].set_title(X_train.columns[term.feature])
    axes[i].set_xlabel(X_train.columns[term.feature])
    axes[i].set_ylabel("Partial Effect")
    axes[i].grid(True)

plt.suptitle("Partial Effects from GAM Model", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/gam_plots_py.png', dpi=150)
plt.close()
print("Saved 'gam_plots_py.png' to working directory.\n")

# Evaluate GAM on test data
pred_gam = gam_model.predict(X_test)
rmse_gam = np.sqrt(mean_squared_error(y_test, pred_gam))
r2_gam = r2_score(y_test, pred_gam)
print(f"GAM Model Test RMSE: {rmse_gam:.2f} MPa\n")


# --- 4.4 Model 4: Random Forest ---
print("--- Training Model 4: Random Forest ---")

# Get default mtry (max_features) as sqrt of number of features
n_features = X_train.shape[1]
mtry = int(np.sqrt(n_features))
print(f"Using mtry (max_features) = {mtry}")

rf_model = RandomForestRegressor(n_estimators=500, max_features=mtry, random_state=420,
                                 oob_score=True)
rf_model.fit(X_train, y_train)
print(f"Random Forest OOB Score: {rf_model.oob_score_:.3f}")

# Visualize RF Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)
features = X_train.columns

plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importances")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.savefig('plots/rf_importance_plot_py.png', dpi=150)
plt.close()
print("Saved 'rf_importance_plot_py.png' to working directory.\n")

# Evaluate RF on test data
pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
r2_rf = r2_score(y_test, pred_rf)
print(f"RF Model Test RMSE: {rmse_rf:.2f} MPa\n")


# --- 4.5 Model 5: Extreme Gradient Boosting (XGBoost) ---
print("--- Training Model 5: Extreme Gradient Boosting ---")
xgb_model = XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=420, n_jobs=-1)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Evaluate XGBoost on test data
pred_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
r2_xgb = r2_score(y_test, pred_xgb)
print(f"XGBoost Model Test RMSE: {rmse_xgb:.2f} MPa\n")


# ===============================================================
# Section 5: Final Results Summary
# ===============================================================
print("--- FINAL RESULTS ---")
results_summary = pd.DataFrame({
    "Model": ["Multiple Linear Regression (MLR)", "Lasso Regression",
              "Generalized Additive Model (GAM)", "Random Forest",
              "Extreme Gradient Boosting (XGBoost)"],
    "Test RMSE": [rmse_mlr, rmse_lasso, rmse_gam, rmse_rf, rmse_xgb],
    "Test R^2": [r2_mlr, r2_lasso, r2_gam, r2_rf, r2_xgb]
})
results_summary['Test RMSE'] = results_summary['Test RMSE'].round(2)
results_summary['Test R^2'] = results_summary['Test R^2'].round(3)
print(results_summary)
