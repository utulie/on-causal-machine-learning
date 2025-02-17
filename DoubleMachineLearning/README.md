# Double Machine Learning (DML)

## Description

In high-dimensional data, traditional methods for estimating causal effects can be biased due to unobserved confounders. The causal effect of a treatment $T$ on an outcome $Y$ can be expressed as:

$$
Y_i = m(X_i) + \theta T_i + \epsilon_i
$$

where:
- $Y_i$ is the outcome for observation $i$
- $T_i$ is the treatment indicator
- $m(X_i)$ represents the effect of control variables \( X \)
- $\epsilon_i$ is the error term

The challenge arises when $m(X)$ is unknown and may be high-dimensional, leading to biased estimates of $\theta$ if not properly controlled.

## Solution Steps


   - Use machine learning to predict the outcome variable $Y$ and treatment variable $T$ based on control variables $X$:
   
   $$
   \hat{m}(X) = \text{argmin}_m \mathbb{E}[(Y - m(X))^2]
   $$
   
   $$
   \hat{g}(X) = \text{argmin}_g \mathbb{E}[(T - g(X))^2]
   $$


   - Remove the estimated nuisance effects:
   
   $$
   \tilde{Y} = Y - \hat{m}(X)
   $$
   
   $$
   \tilde{T} = T - \hat{g}(X)
   $$


   - Calculate the average treatment effect via linear regression
   


## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1), C1-C68.


