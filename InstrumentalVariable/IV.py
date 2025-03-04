import numpy as np

# Generate synthetic data with confounding
np.random.seed(42)
n = 2000  # Number of samples

# Instrument Z (exogenous)
Z = np.random.normal(0, 1, n)

# Unobserved confounder U
U = np.random.normal(0, 1, n)

# Treatment X (endogenous: affected by Z and U)
X = 0.5 * Z + U + np.random.normal(0, 0.5, n)

# Outcome Y (true causal effect = 3.0)
Y = 3.0 * X + U + np.random.normal(0, 1, n)

# Add constant term for intercept
Z_const = np.column_stack([np.ones(n), Z])  # Stage 1 design matrix
X_const = np.column_stack([np.ones(n), X])  # OLS design matrix

# OLS regression (biased)
ols_coef = np.linalg.inv(X_const.T @ X_const) @ X_const.T @ Y
ols_effect = ols_coef[1]  # Treatment effect estimate

# 2SLS implementation
# Stage 1: Regress X on Z
stage1_coef = np.linalg.inv(Z_const.T @ Z_const) @ Z_const.T @ X
X_hat = Z_const @ stage1_coef  # Predicted X

# Stage 2: Regress Y on predicted X
X_hat_const = np.column_stack([np.ones(n), X_hat])
tsls_coef = np.linalg.inv(X_hat_const.T @ X_hat_const) @ X_hat_const.T @ Y
iv_effect = tsls_coef[1]  # Causal effect estimate

# Print results
print(f"True causal effect: 3.0")
print(f"OLS estimate (biased): {ols_effect:.3f}")
print(f"2SLS estimate: {iv_effect:.3f}")
