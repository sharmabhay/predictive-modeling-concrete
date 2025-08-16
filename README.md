# predictive-modeling-concrete

A project focused on predicting the compressive strength of concrete based on its core ingredients and age, with the goal of exploring various regression techniques to identify which model provides the most accurate predictions and best interpretability.

This analysis uses the popular [Concrete Compressive Strength dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) from the UCI Machine Learning Repository.


## Project Goal

1.  **Build and compare** several regression models to predict concrete compressive strength.
2.  **Evaluate** the models based on their predictive accuracy using Root Mean Squared Error (RMSE).
3.  **Interpret** the results to understand the non-linear effects of different ingredients on concrete strength.
4.  **Determine** the best model(s) for pure prediction as well as for interpretability.


## Methodology

1.  **Data Exploration (EDA):** Begin by loading data, cleaning column names, and visualizing key relationships, most notably the non-linear trend between `age` and `strength`.
2.  **Data Splitting:** Split dataset into an 80% training set and a 20% testing set to ensure a robust evaluation of model performance.
3.  **Model Training:** Four different regression models trained on the training data:
    * **Multiple Linear Regression (MLR):** A baseline model assuming linear relationships.
    * **Lasso Regression:** A regularized model for automatic feature selection.
    * **Generalized Additive Model (GAM):** A flexible model designed to capture non-linear relationships using smoothing splines.
    * **Random Forest:** An ensemble model used to maximize predictive accuracy and capture complex interactions between variables.
4.  **Evaluation:** All models evaluated on the unseen test set, and their performance compared using **Root Mean Squared Error (RMSE)**.


## Visualizations Generated

* **`correlation_heatmap.png`**: A correlation plot showing the correlations between each variable.
* **`strength_vs_age.png`**: A scatter plot showing the strong, non-linear relationship between the age of the concrete and its final compressive strength.
* **`gam_plots.png`**: Partial dependence plots from the GAM, illustrating the individual effect of each ingredient on strength.
* **`rf_importance_plot.png`**: A feature importance plot from the Random Forest model, ranking the predictors by their contribution to predictive accuracy.

### Key Findings:

* **Best for Prediction:** **Random Forest** achieved the lowest RMSE. Its ability to model complex interactions makes it the most accurate predictor.
* **Best for Interpretation:** **Generalized Additive Model (GAM)** provided the best balance of performance and interpretability, with an excellent RMSE and clear partial dependence plots to explain the non-linear effects of each ingredient.


## Getting Started

To replicate this analysis, clone repository and run R script.

### Prerequisites

Ensure R and RStudio are installed. Install all required R packages:
```R
install.packages(c("dplyr", "readxl", "ggplot2", "corrplot", "caret", "glmnet", "mgcv", "randomForest"))
```

### Usage

```bash
git clone https://github.com/sharmabhay/predictive-modeling-concrete.git
cd predictive-modeling-concrete
```
