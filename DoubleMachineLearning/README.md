# Double Machine Learning (DML)

## Introduction

Double Machine Learning (DML) is a framework for estimating causal parameters in high-dimensional settings, combining machine learning with econometric techniques. It addresses the challenge of confounding variables by employing machine learning models to control for them.

The DML estimator is formulated as follows:

1. Estimate the nuisance parameters:

   \[
   \hat{m}(X) = \text{argmin}_m \, \mathbb{E}[(Y - m(X))^2]
   \]

   \[
   \hat{g}(X) = \text{argmin}_g \, \mathbb{E}[(T - g(X))^2]
   \]

2. Compute the adjusted outcome and treatment:

   \[
   \tilde{Y} = Y - \hat{m}(X)
   \]

   \[
   \tilde{T} = T - \hat{g}(X)
   \]

3. Estimate the causal effect:

   \[
   \hat{\theta} = \frac{1}{n} \sum_{i=1}^n \tilde{Y}_i \tilde{T}_i
   \]

## References

- Chernozhukov, V., Chetverikov, D., & Kato, K. (2018). "Double Machine Learning for Treatment and Causal Parameters." *The Annals of Statistics*, 46(2), 1-30. [Link to paper](https://projecteuclid.org/euclid.aos/1532034204).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
