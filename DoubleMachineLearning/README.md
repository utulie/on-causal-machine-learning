# Double Machine Learning (DML)

## Problem Description

In high-dimensional data, traditional methods for estimating causal effects can be biased due to unobserved confounders. The causal effect of a treatment \( T \) on an outcome \( Y \) can be expressed as:

$$
Y_i = m(X_i) + \theta T_i + \epsilon_i
$$

where:
- \( Y_i \) is the outcome for observation \( i \)
- \( T_i \) is the treatment indicator
- \( m(X_i) \) represents the effect of control variables \( X \)
- \( \epsilon_i \) is the error term

The challenge arises when \( m(X) \) is unknown and may be high-dimensional, leading to biased estimates of \( \theta \) if not properly controlled.

## Solution Steps

1. **Estimate Nuisance Parameters**:
   - Use machine learning to predict the outcome variable \( Y \) and treatment variable \( T \) based on control variables \( X \):
   
   $$
   \hat{m}(X) = \text{argmin}_m \, \mathbb{E}[(Y - m(X))^2]
   $$
   
   $$
   \hat{g}(X) = \text{argmin}_g \, \mathbb{E}[(T - g(X))^2]
   $$

2. **Compute Adjusted Variables**:
   - Remove the estimated nuisance effects:
   
   $$
   \tilde{Y} = Y - \hat{m}(X)
   $$
   
   $$
   \tilde{T} = T - \hat{g}(X)
   $$

3. **Estimate Causal Effects**:
   - Calculate the average treatment effect:
   
   $$
   \hat{\theta} = \frac{1}{n} \sum_{i=1}^n \tilde{Y}_i \tilde{T}_i
   $$

## References

- Chernozhukov, V., Chetverikov, D., & Kato, K. (2018). "Double Machine Learning for Treatment and Causal Parameters." *The Annals of Statistics*, 46(2), 1-30. [Link to paper].

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
