# Load necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import line
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pygam import LinearGAM, s, f


# ===============================================================
# Section 1: Data Loading and Preparation
# ===============================================================
print("--- 1. Loading and Preparing Data ---")

# Load the dataset from the .xlsx file
try:
    concrete_df = pd.read_excel('data/Concrete_Data.xlsx', engine='openpyxl')
except FileNotFoundError:
    print("File not found. Please ensure the dataset is in the correct path.")
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} observations")
print(f"Test set size: {X_test.shape[0]} observations\n")


# ===============================================================
# Section 4: Model Training and Evaluation
# ===============================================================
# TODO: add all models from R script.


# TODO: update README as well!
# TODO: add Deep Learning model for comparison?
