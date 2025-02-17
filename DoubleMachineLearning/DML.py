import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 10

# Covariates X
X = np.random.normal(size=(n_samples, n_features))

# Treatment T (partially dependent on X)
T = np.random.normal(size=n_samples) + X[:, 0] + 0.5 * X[:, 1]

# Outcome Y (dependent on T and X)
theta = 2.0  # True causal effect
Y = theta * T + X[:, 0] + 0.5 * X[:, 1] + np.random.normal(size=n_samples)

# Split data into training and test sets
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.5, random_state=42)

# Step 1: Fit nuisance models using Lasso Regression
model_T = Lasso(alpha=0.1)  # Model for T ~ X
model_T.fit(X_train, T_train)
T_residual = T_test - model_T.predict(X_test)  # Residuals for T

model_Y = Lasso(alpha=0.1)  # Model for Y ~ X
model_Y.fit(X_train, Y_train)
Y_residual = Y_test - model_Y.predict(X_test)  # Residuals for Y

# Step 2: Fit causal model (regress Y_residual on T_residual)
theta_estimator = LinearRegression()
theta_estimator.fit(T_residual.reshape(-1, 1), Y_residual)

# Estimated causal effect
estimated_theta = theta_estimator.coef_[0]
print(f"Estimated causal effect: {estimated_theta}")

# Reference: Double Machine Learning by Chernozhukov et al. (2018)
# Paper: https://doi.org/10.1111/ectj.12097
