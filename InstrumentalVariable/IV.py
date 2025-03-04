import numpy as np

def generate_iv_data(n=1000):
    """Generate synthetic data with:
    - Z: Valid instrument
    - X: Endogenous treatment
    - Y: Outcome variable
    - True causal effect: 3.0"""
    np.random.seed(42)
    Z = np.random.normal(size=n)          # Instrument
    U = np.random.normal(size=n)           # Unobserved confounder
    X = 0.5*Z + U + np.random.normal(0, 0.1, n)  # Endogenous treatment
    Y = 3.0*X + U + np.random.normal(0, 0.1, n)  # Outcome
    return X.reshape(-1,1), Y, Z.reshape(-1,1)

def iv_2sls(X, Y, Z):
    """Two-stage least squares (2SLS) implementation"""
    # Stage 1: Regress X on Z
    Z1 = np.hstack([np.ones((Z.shape[0],1)), Z])
    beta1 = np.linalg.inv(Z1.T @ Z1) @ Z1.T @ X
    X_hat = Z1 @ beta1
    
    # Stage 2: Regress Y on predicted X
    X_hat1 = np.hstack([np.ones((X_hat.shape[0],1)), X_hat])
    beta2 = np.linalg.inv(X_hat1.T @ X_hat1) @ X_hat1.T @ Y
    return beta2[1].item()

def ols(X, Y):
    """Ordinary least squares for comparison"""
    X1 = np.hstack([np.ones((X.shape[0],1)), X])
    beta = np.linalg.inv(X1.T @ X1) @ X1.T @ Y
    return beta[1].item()

if __name__ == '__main__':
  # Example usage
  X, Y, Z = generate_iv_data()
  print(f"True causal effect: 3.0")
  print(f"OLS estimate: {ols(X, Y):.3f} (biased)")
  print(f"2SLS estimate: {iv_2sls(X, Y, Z):.3f} (unbiased)")
