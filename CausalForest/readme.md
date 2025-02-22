# Causal Forest for Heterogeneous Treatment Effect Estimation

A Python implementation of Causal Forest, designed to estimate conditional average treatment effects (CATE) from observational data.  
**Key Feature**: Non-parametric estimation of treatment effect heterogeneity without external ML library dependencies.



## Background
Traditional causal inference methods (e.g., linear regression, propensity score matching) often assume homogeneous treatment effects or rely on strong parametric assumptions. **Causal Forest** extends Random Forest to causal inference by:  
1. Optimizing splits for treatment effect heterogeneity  
2. Combining multiple causal trees to reduce variance  
3. Handling high-dimensional confounders non-linearly  



## Problem Formulation
Under the **Potential Outcomes Framework**, we define:  
- Individual Treatment Effect:  $$\tau_i = Y_i(1) - Y_i(0) $$
- Conditional Average Treatment Effect (CATE):  $$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$$ 

**Assumptions**:  
1. Unconfoundedness: $W \perp(Y(1), Y(0)) \mid X$
2. Overlap: $0 < P(W=1 \mid X) < 1$

---

## Algorithm Overview

### 1. Single Causal Tree Construction
**Splitting Criterion**: Maximize treatment effect variance between child nodes  
$$\Delta = (\hat{\tau}_L - \hat{\tau}_R)^2$$
where $\hat{\tau}_L, \hat{\tau}_R$ are effect estimates in left/right nodes.

**Steps**:  
1. At each node, search features/thresholds to maximize $ \Delta $
2. Recursively split until:  
   - Node samples < `min_samples_leaf`  
   - Reaches `max_depth`  
3. Estimate leaf effects via:  
   $$
   \hat{\tau}_{\text{leaf}} = \frac{1}{n_1}\sum Y_{W=1} - \frac{1}{n_0}\sum Y_{W=0}
   $$

### 2. Forest Ensemble
1. **Bootstrap Aggregation**: Grow \( B \) trees on resampled datasets  
2. **Prediction**: Average effects across all trees  
   $$
   \hat{\tau}(x) = \frac{1}{B}\sum_{b=1}^B \hat{\tau}_b(x)
   $$

---

## Usage Example
```python
from causal_forest import CausalForest

# Initialize model
cf = CausalForest(n_trees=100, min_samples_leaf=20, max_depth=5)

# Train on observational data
# X: covariates, W: treatment (0/1), Y: outcome
cf.fit(X, W, Y)

# Predict CATE for new samples
tau_hat = cf.predict_effect(X_test)
