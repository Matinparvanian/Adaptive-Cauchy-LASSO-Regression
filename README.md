<img width="1536" height="1024" alt="ACLR_Method" src="https://github.com/user-attachments/assets/597c99d1-79e8-4277-873c-75deefffebb9" />

# Adaptive Cauchy LASSO Regression

This repository contains the paper:

**Robust Variable Selection via Adaptive Cauchy LASSO: Guarantees and Initialization for High-Dimensional Heavy-Tailed Data**

The paper introduces **Adaptive Cauchy LASSO Regression (ACLR)**, a robust sparse regression framework designed for variable selection and prediction in the presence of heavy-tailed noise, outliers, and correlated predictors. Classical sparse regression methods can become unstable under contamination and may produce unreliable feature selection results, particularly in high-dimensional settings where the number of variables may exceed the sample size.

ACLR combines a Cauchy-based loss function with adaptive $\ell_1$ penalization to reduce the influence of extreme observations while encouraging sparse solutions. The method further incorporates a Ridge–Huber initialization strategy to stabilize the adaptive weights used in the penalization step. This combination improves robustness and false-positive control while maintaining competitive prediction performance.

The proposed framework is evaluated through extensive simulation studies under both low-dimensional and high-dimensional settings, including correlated covariate structures and scenarios where $p \gg n$. ACLR is compared with several benchmark methods, including:

- Cauchy LASSO Regression (CLR)
- LASSO
- Quantile LASSO (QL)
- Nolan–Ojeda–Revah Regression (NOR)

Performance is assessed using false-positive rate (FPR), false-negative rate (FNR), and mean absolute error (MAE).

## Authors

Matin Parvanian and Mina Aminghafari  
Department of Mathematics and Statistics  
University of Calgary
