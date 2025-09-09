# predictive-modeling-concrete

A project focused on predicting compressive strength of concrete based on its core ingredients and age, with a goal of exploring various regression techniques to identify which model provides the most accurate predictions and best interpretability.

This analysis uses [Concrete Compressive Strength dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) from UCI Machine Learning Repository.


## Project Goal

1.  **Build and compare** several regression models to predict concrete compressive strength.
2.  **Evaluate** models based on their predictive accuracy using Root Mean Squared Error (RMSE).
3.  **Interpret** results to understand non-linear effects of different ingredients on concrete strength.
4.  **Determine** best model(s) for pure prediction as well as for interpretability.


## Methodology

1.  **Data Exploration (EDA):** Begin by loading data, cleaning column names, and visualizing key relationships, such as non-linear trend between `age` and `strength`.
2.  **Data Splitting:** Split dataset into an 80% training set and a 20% testing set to ensure a robust evaluation of model performance.
3.  **Model Training:** Four different regression models trained on training data:
    * **Multiple Linear Regression (MLR):** A baseline model assuming linear relationships.
    * **Lasso Regression:** A regularized model for automatic feature selection.
    * **Generalized Additive Model (GAM):** A flexible model designed to capture non-linear relationships using smoothing splines.
    * **Random Forest:** An ensemble model used to maximize predictive accuracy and capture complex interactions between variables.
    * **Extreme Gradient Boosting (XGBoost):** A gradient boosting ensemble model, often a top performer on tabular data.
4.  **Evaluation:** All models evaluated on unseen test set, and their performance compared using **Root Mean Squared Error (RMSE)**.


## Visualizations Generated

* **`correlation_heatmap`**: A correlation plot showing correlations between each variable.
* **`strength_vs_age`**: A scatter plot showing a strong, non-linear relationship between age of concrete and its final compressive strength.
* **`gam_plots`**: Partial dependence plots from GAM, illustrating individual effect of each ingredient on strength.
* **`rf_importance_plot`**: A feature importance plot from Random Forest model, ranking predictors by their contribution to predictive accuracy.

### Key Findings:

* **Best for Prediction:** **XGBoost** achieved lowest RMSE, making it most accurate for predictive modeling. It also offered an excellent balance of speed and model size compared to **Random Forest**.
* **Best for Interpretation:** **GAM** provided best balance of performance and interpretability, with an excellent RMSE and clear partial dependence plots to explain non-linear effects of each ingredient.


## Getting Started

To replicate this analysis, clone repository and run R and/or Python script.

### Prerequisites

Ensure R and RStudio are installed. Install all required R packages:
```R
install.packages(c("dplyr", "readxl", "ggplot2", "corrplot", "caret", "glmnet", "mgcv", "randomForest"))
```

### Usage

```bash
git clone https://github.com/sharmabhay/predictive-modeling-concrete.git
cd predictive-modeling-concrete
pip install -r requirements.txt
```
