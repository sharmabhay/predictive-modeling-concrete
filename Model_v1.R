setwd("H:/Documents - Copy/GitHub/predictive-modeling-concrete/")

# Load necessary libraries
library(readr)          # For reading the data file efficiently
library(dplyr)          # For data manipulation
library(ggplot2)        # For high-quality visualizations
library(corrplot)       # For visualizing the correlation matrix
library(caret)          # For data splitting
library(glmnet)         # For Lasso regression
library(gam)            # For Generalized Additive Models


# ===============================================================
# Section 1: Data Loading and Preparation
# ===============================================================
cat("--- 1. Loading and Preparing Data ---\n")

# Load the dataset from the .xlsx file
file_path <- "data/Concrete_Data.xlsx"
if (!file.exists(file_path)) {
  stop("Error: 'Concrete_Data.xlsx' not found. Please place it in your working directory.")
}
concrete_data <- read_excel(file_path)

# Assign cleaner column names
colnames(concrete_data) <- c("cement", "slag", "ash", "water", "superplasticizer", 
                             "coarse_agg", "fine_agg", "age", "strength")

# Confirm data is loaded correctly and check for missing values
cat("Data Summary:\n")
print(summary(concrete_data))
if(sum(is.na(concrete_data)) == 0) {
  cat("\nData is complete. No missing values found.\n\n")
} else {
  cat("\nWarning: Missing values detected.\n\n")
}


# ===============================================================
# Section 2: Exploratory Data Analysis (EDA) & Plot Generation
# ===============================================================
cat("--- 2. Performing EDA and Generating Plots ---\n")

# 2.1 Correlation Matrix Heatmap (for internal analysis)
cor_matrix <- cor(concrete_data)
png("correlation_heatmap.png", width = 800, height = 800, res = 100)
corrplot(cor_matrix, method = "color", type = "upper", order = "hclust",
         addCoef.col = "black", tl.col = "black", tl.srt = 45, diag = FALSE)
dev.off()
cat("Saved 'correlation_heatmap.png' to working directory.\n")

# 2.2 Scatter Plot
strength_age_plot <- ggplot(concrete_data, aes(x = age, y = strength)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  geom_smooth(method = "loess", color = "red", se = TRUE, linewidth = 1.2) +
  labs(title = "Concrete Strength Increases Non-Linearly with Age",
       x = "Age (days)",
       y = "Compressive Strength (MPa)") +
  theme_bw(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("strength_vs_age.png", plot = strength_age_plot, width = 7, height = 5, dpi = 150)
cat("Saved 'strength_vs_age.png' to working directory.\n\n")


# ===============================================================
# Section 3: Data Splitting
# ===============================================================
cat("--- 3. Splitting Data into Training (80%) and Test (20%) Sets ---\n")
trainIndex <- createDataPartition(concrete_data$strength, p = .8, list = FALSE, times = 1)
train_data <- concrete_data[trainIndex, ]
test_data  <- concrete_data[-trainIndex, ]
cat(paste("Training set size:", nrow(train_data), "observations\n"))
cat(paste("Test set size:", nrow(test_data), "observations\n\n"))


# ===============================================================
# Section 4: Model Training and Evaluation
# ===============================================================

# Define RMSE function for evaluation
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# --- 4.1 Model 1: Multiple Linear Regression (Baseline) ---
cat("--- Training Model 1: Multiple Linear Regression ---\n")
lm_model <- lm(strength ~ ., data = train_data)
pred_lm <- predict(lm_model, newdata = test_data)
rmse_lm <- rmse(test_data$strength, pred_lm)
cat(paste("Linear Model Test RMSE:", round(rmse_lm, 2), "MPa\n\n"))


# --- 4.2 Model 2: Lasso Regression ---
cat("--- Training Model 2: Lasso Regression ---\n")
# Prepare data matrices required by glmnet
x_train <- model.matrix(strength ~ . - 1, data = train_data)
y_train <- train_data$strength
x_test <- model.matrix(strength ~ . - 1, data = test_data)
y_test <- test_data$strength

# Use 10-fold CV to find the optimal lambda
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1) # alpha=1 specifies Lasso
best_lambda <- cv_lasso$lambda.min
cat(paste("Best Lambda from 10-fold CV:", round(best_lambda, 4), "\n"))

# Display the coefficients selected by Lasso
lasso_coef <- coef(cv_lasso, s = best_lambda)
cat("\nLasso Coefficients at Optimal Lambda:\n")
print(lasso_coef)

# Evaluate on test data
pred_lasso <- predict(cv_lasso, s = best_lambda, newx = x_test)
rmse_lasso <- rmse(y_test, pred_lasso)
cat(paste("\nLasso Model Test RMSE:", round(rmse_lasso, 2), "MPa\n\n"))


# --- 4.3 Model 3: Generalized Additive Model (GAM) ---
cat("--- Training Model 3: Generalized Additive Model ---\n")
# Fit a GAM with smoothing splines for each predictor
gam_model <- gam(strength ~ s(cement, 4) + s(slag, 4) + s(ash, 4) + 
                   s(water, 4) + s(superplasticizer, 4) + s(coarse_agg, 4) +
                   s(fine_agg, 4) + s(age, 4), data = train_data)

# Print GAM summary to see significance of non-linear terms
cat("GAM Model Summary:\n")
print(summary(gam_model))

# Generate and save the component plots
png("gam_plots.png", width = 1200, height = 700, res = 120)
par(mfrow=c(2,4), mar=c(4.5, 4.5, 2, 1), oma=c(0,0,2,0))
plot(gam_model, se = TRUE, col = "darkgreen", rug=TRUE, cex.lab=1.2)
mtext("Partial Effects from GAM Model", outer = TRUE, cex = 1.5)
dev.off()
cat("\nSaved 'gam_plots.png' to working directory.\n")

# Evaluate GAM on test data
pred_gam <- predict(gam_model, newdata = test_data)
rmse_gam <- rmse(test_data$strength, pred_gam)
cat(paste("\nGAM Model Test RMSE:", round(rmse_gam, 2), "MPa\n\n"))


# ===============================================================
# Section 5: Final Results Summary
# ===============================================================
cat("--- FINAL RESULTS ---\n")
results_summary <- data.frame(
  Model = c("Multiple Linear Regression", "Lasso Regression", "Generalized Additive Model"),
  Test_RMSE = c(round(rmse_lm, 2), round(rmse_lasso, 2), round(rmse_gam, 2))
)
print(results_summary)