.libPaths("~/R/4.3.1")

library(MASS)
library(glmnet)
library(quantreg)
library(ggplot2)
library(dplyr)
library(tidyr)

set.seed(123)

############################
# 1) global settings
############################

# Example 3 from the paper:
# beta = (4, 0.5, -2, 3, -0.75, 0, 0, 0, 0, 0)^T
# This example studies the effect of different-valued coefficients.

p <- 10
beta_true <- c(4, 0.5, -2, 3, -0.75, rep(0, 5))

related_idx <- 1:5
unrelated_idx <- 6:10

n_train_values <- c(50, 100, 200)
rho_values <- c(0, 0.5, 0.75)
n_test <- 30

# Number of replications for final simulation.
# For paper-style final results, use 500 or 1000.
n_rep <- 500
report_every <- 100

# Small pilot repetitions used only for ACLR hyperparameter tuning.
# Increase to 50 or 100 for a more stable tuning decision.
tuning_n_rep <- 30

# Selection threshold. A slightly lower threshold helps avoid counting
# small but real estimated active coefficients as false negatives.
coef_tol <- 0.25

eps_num <- 1e-8

# CLR and QL lambda grids
lambda_grid_clr <- seq(0.01, 3, by = 0.01)
lambda_grid_ql  <- exp(seq(log(0.01), log(20), length.out = 25))

# ACLR hyperparameter grid.
# The selected grid row will define lambda_grid_aclr, adaptive_gamma,
# adaptive_eps, and huber_k for the final simulation.
aclr_grid <- expand.grid(
  adaptive_gamma = c(0.5, 0.6, 0.7, 0.8),
  adaptive_eps   = c(0.01, 0.05, 0.10),
  huber_k        = c(1.345, 1.5, 2.0),
  lambda_min     = c(0.0005),
  lambda_max     = c(3, 5),
  lambda_len     = c(35),
  stringsAsFactors = FALSE
)

make_lambda_grid_aclr <- function(lambda_min, lambda_max, lambda_len) {
  exp(seq(log(lambda_min), log(lambda_max), length.out = lambda_len))
}

# Default placeholder values; overwritten after grid tuning.
lambda_grid_aclr <- make_lambda_grid_aclr(0.0005, 3, 35)
adaptive_gamma <- 0.5
adaptive_eps   <- 0.10
huber_k        <- 1.5

# Huber-ridge initializer settings for the ACLR version.
# NULL means lambda_init = 1 / n inside the initializer.
lambda_init_huber_ridge <- NULL

############################
# 2) covariance + data generation
############################

make_gamma <- function(p, rho) {
  outer(1:p, 1:p, function(i, j) rho^abs(i - j))
}

generate_example3_data <- function(n_train, n_test, p, rho, beta_true) {
  Gamma <- make_gamma(p, rho)
  
  X_train <- MASS::mvrnorm(n = n_train, mu = rep(0, p), Sigma = Gamma)
  X_test  <- MASS::mvrnorm(n = n_test,  mu = rep(0, p), Sigma = Gamma)
  
  eps_train <- rcauchy(n_train, location = 0, scale = 1)
  eps_test  <- rcauchy(n_test,  location = 0, scale = 1)
  
  y_train <- as.numeric(X_train %*% beta_true + eps_train)
  y_test  <- as.numeric(X_test  %*% beta_true + eps_test)
  
  # match GitHub style: center predictors and response
  X_train <- scale(X_train, center = TRUE, scale = FALSE)
  X_test  <- scale(X_test,  center = TRUE, scale = FALSE)
  y_train <- as.numeric(scale(y_train, center = TRUE, scale = FALSE))
  y_test  <- as.numeric(scale(y_test,  center = TRUE, scale = FALSE))
  
  list(
    X_train = X_train,
    y_train = y_train,
    X_test  = X_test,
    y_test  = y_test
  )
}

############################
# 3) CLR 
############################

clr_objective_github <- function(beta, X, y, lambda) {
  r <- as.numeric(y - X %*% beta)
  sum(log(1 + r^2)) + lambda * sum(abs(beta))
}

classo_path <- function(y, x, lambda_grid) {
  q <- length(lambda_grid)
  p <- ncol(x)
  
  beta_lambda <- matrix(0, nrow = q, ncol = p)
  lambda_used <- numeric(q)
  
  beta_start <- coef(lm(y ~ x - 1))
  beta_start[is.na(beta_start)] <- 0
  
  for (i in seq_along(lambda_grid)) {
    lam <- lambda_grid[i]
    
    fr <- function(beta) {
      r <- as.numeric(y - x %*% beta)
      sum(log(1 + r^2)) + lam * sum(abs(beta))
    }
    
    opt_fit <- optim(beta_start, fr, method = "BFGS", control = list(maxit = 1000))
    beta_lambda[i, ] <- opt_fit$par
    lambda_used[i] <- lam
    beta_start <- opt_fit$par
  }
  
  list(lambda = lambda_used, Beta = beta_lambda)
}

compute_github_ic <- function(X, y, beta_path, lambda_grid) {
  rp <- length(lambda_grid)
  n <- length(y)
  
  dp <- rep(0, rp)
  G  <- rep(0, rp)
  aic <- rep(0, rp)
  bic <- rep(0, rp)
  
  for (ib in 1:rp) {
    dp[ib] <- sum(abs(beta_path[ib, ]) > 0)
  }
  
  for (ic in 1:rp) {
    r <- as.numeric(y - X %*% beta_path[ic, ])
    G[ic] <- log(1 + sum(r^2)) / n
    aic[ic] <- G[ic] + (2 * dp[ic] / n)
    bic[ic] <- G[ic] + (log(n) * dp[ic] / n)
  }
  
  list(dp = dp, G = G, AIC = aic, BIC = bic)
}

fit_clr_select_lambda <- function(X, y, lambda_grid, criterion = c("AIC", "BIC")) {
  criterion <- match.arg(criterion)
  
  cl_fit <- classo_path(y = y, x = X, lambda_grid = lambda_grid)
  ic <- compute_github_ic(X = X, y = y, beta_path = cl_fit$Beta, lambda_grid = lambda_grid)
  
  idx <- if (criterion == "AIC") which.min(ic$AIC) else which.min(ic$BIC)
  
  beta_hat <- cl_fit$Beta[idx, ]
  beta_hat[abs(beta_hat) <= coef_tol] <- 0
  
  list(
    lambda = cl_fit$lambda[idx],
    beta = beta_hat,
    df = sum(abs(beta_hat) > 0),
    G = ic$G[idx],
    AIC = ic$AIC[idx],
    BIC = ic$BIC[idx],
    iterations = NA_real_
  )
}

##################################################
# 4) ACLR with Adaptive Ridge-Huber initialization
#################################################

robust_sigma_update <- function(residuals, min_sigma = 1e-4) {
  s <- mad(residuals, center = 0, constant = 1.4826, na.rm = TRUE)
  if (!is.finite(s) || is.na(s) || s < min_sigma) {
    s <- sqrt(mean(residuals^2))
  }
  max(s, min_sigma)
}

aclr_objective <- function(beta, X, y, lambda, sigma, weights) {
  r <- as.numeric(y - X %*% beta)
  sum(log(pmax(sigma^2 + r^2, eps_num))) + lambda * sum(weights * abs(beta))
}

# Huber loss used for the robust ridge initial estimator.
# This replaces the old ridge-only initializer in ACLR.
huber_loss <- function(r, k = 1.345) {
  abs_r <- abs(r)
  ifelse(abs_r <= k, 0.5 * r^2, k * abs_r - 0.5 * k^2)
}

# Initial estimator for the new ACLR version:
# beta_init = arg min_beta sum_i rho_H(y_i - x_i^T beta) + lambda_init sum_j beta_j^2.
fit_huber_ridge_initial <- function(X, y, lambda_init = NULL, k = 1.345,
                                    maxit = 1000) {
  p <- ncol(X)
  
  if (is.null(lambda_init)) {
    # A stable default ridge strength for the initializer.
    lambda_init <- 1 / max(1, nrow(X))
  }
  
  beta_start <- tryCatch({
    fit0 <- glmnet(
      x = X,
      y = y,
      alpha = 0,
      lambda = lambda_init,
      intercept = FALSE,
      standardize = FALSE
    )
    as.numeric(coef(fit0))[-1]
  }, error = function(e) rep(0, p))
  
  beta_start[!is.finite(beta_start)] <- 0
  
  objective <- function(beta) {
    r <- as.numeric(y - X %*% beta)
    sum(huber_loss(r, k = k)) + lambda_init * sum(beta^2)
  }
  
  gradient <- function(beta) {
    r <- as.numeric(y - X %*% beta)
    psi <- pmax(pmin(r, k), -k)
    as.numeric(-crossprod(X, psi) + 2 * lambda_init * beta)
  }
  
  opt <- tryCatch({
    optim(
      par = beta_start,
      fn = objective,
      gr = gradient,
      method = "BFGS",
      control = list(maxit = maxit)
    )
  }, error = function(e) NULL)
  
  if (is.null(opt) || any(!is.finite(opt$par))) {
    beta_start
  } else {
    opt$par
  }
}

get_initial_beta_for_aclr <- function(X, y, lambda_init = NULL, k = 1.345) {
  fit_huber_ridge_initial(
    X = X,
    y = y,
    lambda_init = lambda_init,
    k = k
  )
}

make_adaptive_weights <- function(beta_init, gamma = 1, eps = 1e-3) {
  1 / (abs(beta_init) + eps)^gamma
}

fit_aclr_fast_fixed_lambda <- function(X, y, lambda, weights,
                                       sigma_init = NULL,
                                       beta_init = NULL,
                                       max_outer = 60,
                                       tol = 1e-5) {
  p <- ncol(X)
  
  beta <- if (is.null(beta_init)) rep(0, p) else beta_init
  sigma <- if (is.null(sigma_init)) robust_sigma_update(y) else sigma_init
  old_obj <- aclr_objective(beta, X, y, lambda, sigma, weights)
  iter_used <- 0
  
  for (iter in 1:max_outer) {
    iter_used <- iter
    beta_prev <- beta
    sigma_prev <- sigma
    
    r_prev <- as.numeric(y - X %*% beta_prev)
    obs_w <- 1 / pmax(sigma_prev^2 + r_prev^2, eps_num)
    
    sqrt_w <- sqrt(obs_w)
    X_w <- X * sqrt_w
    y_w <- y * sqrt_w
    
    fit_wlasso <- glmnet(
      x = X_w,
      y = y_w,
      alpha = 1,
      lambda = lambda,
      intercept = FALSE,
      standardize = FALSE,
      penalty.factor = weights
    )
    
    beta <- as.numeric(coef(fit_wlasso))[-1]
    beta[abs(beta) <= coef_tol] <- 0
    
    r_new <- as.numeric(y - X %*% beta)
    sigma <- robust_sigma_update(r_new)
    
    new_obj <- aclr_objective(beta, X, y, lambda, sigma, weights)
    
    if (max(abs(beta - beta_prev)) < tol &&
        abs(sigma - sigma_prev) < tol &&
        abs(new_obj - old_obj) < tol) {
      old_obj <- new_obj
      break
    }
    
    old_obj <- new_obj
  }
  
  list(
    beta = beta,
    sigma = sigma,
    objective = old_obj,
    iterations = iter_used
  )
}

compute_ic_aclr <- function(X, y, beta_hat, sigma_hat) {
  e <- as.numeric(y - X %*% beta_hat)
  df_lambda <- sum(abs(beta_hat) > coef_tol)
  n <- length(y)
  
  fit_term <- sum(log(pmax(sigma_hat^2 + e^2, eps_num)))
  
  aic_value <- (fit_term + 2 * df_lambda) / n
  bic_value <- (fit_term + log(n) * df_lambda) / n
  
  list(
    residuals = e,
    df = df_lambda,
    AIC = aic_value,
    BIC = bic_value
  )
}

fit_aclr_fast_select_lambda <- function(X, y, lambda_grid,
                                        criterion = c("AIC", "BIC"),
                                        gamma = 1,
                                        eps = 1e-3,
                                        huber_k = 1.345,
                                        lambda_init = NULL) {
  criterion <- match.arg(criterion)
  
  beta_init0 <- get_initial_beta_for_aclr(
    X = X,
    y = y,
    lambda_init = lambda_init,
    k = huber_k
  )
  weights <- make_adaptive_weights(beta_init0, gamma = gamma, eps = eps)
  
  fits <- vector("list", length(lambda_grid))
  beta_start <- beta_init0
  sigma_start <- robust_sigma_update(y)
  
  for (k in seq_along(lambda_grid)) {
    lam <- lambda_grid[k]
    
    fit_k <- fit_aclr_fast_fixed_lambda(
      X = X,
      y = y,
      lambda = lam,
      weights = weights,
      sigma_init = sigma_start,
      beta_init = beta_start
    )
    
    ic_k <- compute_ic_aclr(
      X = X,
      y = y,
      beta_hat = fit_k$beta,
      sigma_hat = fit_k$sigma
    )
    
    fits[[k]] <- list(
      lambda = lam,
      beta = fit_k$beta,
      sigma = fit_k$sigma,
      df = ic_k$df,
      AIC = ic_k$AIC,
      BIC = ic_k$BIC,
      iterations = fit_k$iterations
    )
    
    beta_start <- fit_k$beta
    sigma_start <- fit_k$sigma
  }
  
  crit_values <- sapply(fits, function(z) z[[criterion]])
  best_idx <- which.min(crit_values)
  best_fit <- fits[[best_idx]]
  best_fit$weights <- weights
  best_fit
}

############################
# 5) other methods
############################

fit_cv_lasso <- function(X, y) {
  fit <- cv.glmnet(
    x = X,
    y = y,
    alpha = 1,
    intercept = FALSE,
    standardize = FALSE
  )
  
  beta_hat <- as.numeric(coef(fit, s = "lambda.min"))[-1]
  beta_hat[abs(beta_hat) <= coef_tol] <- 0
  
  list(
    beta = beta_hat,
    lambda = fit$lambda.min,
    iterations = NA_real_
  )
}

check_loss <- function(u, tau) {
  u * (tau - (u < 0))
}

fit_quantile_lasso_cv <- function(X, y, tau, lambda_grid, nfolds = 5) {
  n <- nrow(X)
  p <- ncol(X)
  
  fold_id <- sample(rep(1:nfolds, length.out = n))
  cv_errors <- rep(Inf, length(lambda_grid))
  
  for (k in seq_along(lambda_grid)) {
    lam <- lambda_grid[k]
    fold_losses <- rep(Inf, nfolds)
    
    for (fold in 1:nfolds) {
      train_idx <- which(fold_id != fold)
      valid_idx <- which(fold_id == fold)
      
      X_tr <- X[train_idx, , drop = FALSE]
      y_tr <- y[train_idx]
      X_va <- X[valid_idx, , drop = FALSE]
      y_va <- y[valid_idx]
      
      X_tr_int <- cbind(1, X_tr)
      X_va_int <- cbind(1, X_va)
      
      fit_try <- tryCatch({
        quantreg::rq.fit.lasso(
          x = X_tr_int,
          y = y_tr,
          tau = tau,
          lambda = c(0, rep(lam, p))
        )
      }, error = function(e) NULL)
      
      if (!is.null(fit_try)) {
        coef_hat <- as.numeric(fit_try$coefficients)
        pred_va  <- as.numeric(X_va_int %*% coef_hat)
        fold_losses[fold] <- mean(check_loss(y_va - pred_va, tau))
      }
    }
    
    cv_errors[k] <- mean(fold_losses)
  }
  
  best_lambda <- lambda_grid[which.min(cv_errors)]
  
  X_int <- cbind(1, X)
  final_fit <- quantreg::rq.fit.lasso(
    x = X_int,
    y = y,
    tau = tau,
    lambda = c(0, rep(best_lambda, p))
  )
  
  coef_hat <- as.numeric(final_fit$coefficients)
  intercept_hat <- coef_hat[1]
  beta_hat <- coef_hat[-1]
  
  beta_hat[abs(beta_hat) <= coef_tol] <- 0
  
  list(
    intercept = intercept_hat,
    beta = beta_hat,
    lambda = best_lambda,
    iterations = NA_real_
  )
}

############################
# 6) NOR model (MAE only)
############################

fit_nor <- function(X, y) {
  beta_start <- coef(lm(y ~ X - 1))
  beta_start[is.na(beta_start)] <- 0
  
  fr <- function(beta) {
    r <- as.numeric(y - X %*% beta)
    sum(log(1 + r^2))
  }
  
  opt_fit <- optim(beta_start, fr, method = "BFGS", control = list(maxit = 1000))
  beta_hat <- opt_fit$par
  
  list(
    beta = beta_hat,
    iterations = NA_real_
  )
}

############################
# 7) prediction + metrics
############################

predict_linear <- function(X, beta, intercept = 0) {
  as.numeric(intercept + X %*% beta)
}

compute_selection_metrics <- function(beta_hat, related_idx, unrelated_idx, tol = coef_tol) {
  selected <- abs(beta_hat) > tol
  
  selected_count <- sum(selected)
  unrelated_selected_count <- sum(selected[unrelated_idx])
  
  fnr <- mean(!selected[related_idx]) * 100
  fpr <- mean(selected[unrelated_idx]) * 100
  
  list(
    selected_count = selected_count,
    unrelated_selected_count = unrelated_selected_count,
    fnr = fnr,
    fpr = fpr
  )
}

compute_mae <- function(y_true, y_pred) {
  mean(abs(y_true - y_pred))
}

make_result_row <- function(n_train, rho, method, beta_hat, y_test, pred_test,
                            related_idx, unrelated_idx,
                            lambda_value = NA_real_,
                            sigma_value = NA_real_,
                            elapsed_time = NA_real_,
                            iterations = NA_real_,
                            include_selection = TRUE) {
  
  if (include_selection) {
    sel <- compute_selection_metrics(
      beta_hat = beta_hat,
      related_idx = related_idx,
      unrelated_idx = unrelated_idx,
      tol = coef_tol
    )
    
    selected_count <- sel$selected_count
    unrelated_selected_count <- sel$unrelated_selected_count
    fnr <- sel$fnr
    fpr <- sel$fpr
  } else {
    selected_count <- NA_real_
    unrelated_selected_count <- NA_real_
    fnr <- NA_real_
    fpr <- NA_real_
  }
  
  data.frame(
    n_train = n_train,
    rho = rho,
    method = method,
    selected_count = selected_count,
    unrelated_selected_count = unrelated_selected_count,
    fnr = fnr,
    fpr = fpr,
    mae_test = compute_mae(y_test, pred_test),
    lambda = lambda_value,
    sigma = sigma_value,
    time_sec = elapsed_time,
    iterations = iterations
  )
}

############################
# 8) Repetition 
############################

run_one_replication_all_methods <- function(n_train, n_test, rho, beta_true,
                                            lambda_grid_clr,
                                            lambda_grid_ql, lambda_grid_aclr,
                                            adaptive_gamma, adaptive_eps,
                                            huber_k, lambda_init_huber_ridge) {
  dat <- generate_example3_data(
    n_train = n_train,
    n_test = n_test,
    p = length(beta_true),
    rho = rho,
    beta_true = beta_true
  )
  
  X_train <- dat$X_train
  y_train <- dat$y_train
  X_test  <- dat$X_test
  y_test  <- dat$y_test
  
  out_rows <- list()
  
  t0 <- proc.time()
  fit_aic_clr <- fit_clr_select_lambda(
    X = X_train,
    y = y_train,
    lambda_grid = lambda_grid_clr,
    criterion = "AIC"
  )
  time_aic_clr <- (proc.time() - t0)[["elapsed"]]
  
  pred_aic_clr <- predict_linear(X_test, fit_aic_clr$beta)
  out_rows[[length(out_rows) + 1]] <- make_result_row(
    n_train = n_train, rho = rho, method = "AIC-CLR",
    beta_hat = fit_aic_clr$beta, y_test = y_test, pred_test = pred_aic_clr,
    related_idx = related_idx, unrelated_idx = unrelated_idx,
    lambda_value = fit_aic_clr$lambda, sigma_value = 1,
    elapsed_time = time_aic_clr, iterations = fit_aic_clr$iterations,
    include_selection = TRUE
  )
  
  t0 <- proc.time()
  fit_bic_clr <- fit_clr_select_lambda(
    X = X_train,
    y = y_train,
    lambda_grid = lambda_grid_clr,
    criterion = "BIC"
  )
  time_bic_clr <- (proc.time() - t0)[["elapsed"]]
  
  pred_bic_clr <- predict_linear(X_test, fit_bic_clr$beta)
  out_rows[[length(out_rows) + 1]] <- make_result_row(
    n_train = n_train, rho = rho, method = "BIC-CLR",
    beta_hat = fit_bic_clr$beta, y_test = y_test, pred_test = pred_bic_clr,
    related_idx = related_idx, unrelated_idx = unrelated_idx,
    lambda_value = fit_bic_clr$lambda, sigma_value = 1,
    elapsed_time = time_bic_clr, iterations = fit_bic_clr$iterations,
    include_selection = TRUE
  )
  
  t0 <- proc.time()
  fit_aic_aclr <- fit_aclr_fast_select_lambda(
    X = X_train,
    y = y_train,
    lambda_grid = lambda_grid_aclr,
    criterion = "AIC",
    gamma = adaptive_gamma, eps = adaptive_eps,
    huber_k = huber_k,
    lambda_init = lambda_init_huber_ridge
  )
  time_aic_aclr <- (proc.time() - t0)[["elapsed"]]
  
  pred_aic_aclr <- predict_linear(X_test, fit_aic_aclr$beta)
  out_rows[[length(out_rows) + 1]] <- make_result_row(
    n_train = n_train, rho = rho, method = "AIC-ACLR-Huber",
    beta_hat = fit_aic_aclr$beta, y_test = y_test, pred_test = pred_aic_aclr,
    related_idx = related_idx, unrelated_idx = unrelated_idx,
    lambda_value = fit_aic_aclr$lambda, sigma_value = fit_aic_aclr$sigma,
    elapsed_time = time_aic_aclr, iterations = fit_aic_aclr$iterations,
    include_selection = TRUE
  )
  
  t0 <- proc.time()
  fit_bic_aclr <- fit_aclr_fast_select_lambda(
    X = X_train,
    y = y_train,
    lambda_grid = lambda_grid_aclr,
    criterion = "BIC",
    gamma = adaptive_gamma, eps = adaptive_eps,
    huber_k = huber_k,
    lambda_init = lambda_init_huber_ridge
  )
  time_bic_aclr <- (proc.time() - t0)[["elapsed"]]
  
  pred_bic_aclr <- predict_linear(X_test, fit_bic_aclr$beta)
  out_rows[[length(out_rows) + 1]] <- make_result_row(
    n_train = n_train, rho = rho, method = "BIC-ACLR-Huber",
    beta_hat = fit_bic_aclr$beta, y_test = y_test, pred_test = pred_bic_aclr,
    related_idx = related_idx, unrelated_idx = unrelated_idx,
    lambda_value = fit_bic_aclr$lambda, sigma_value = fit_bic_aclr$sigma,
    elapsed_time = time_bic_aclr, iterations = fit_bic_aclr$iterations,
    include_selection = TRUE
  )
  
  t0 <- proc.time()
  fit_cv_las <- fit_cv_lasso(X_train, y_train)
  time_cv_las <- (proc.time() - t0)[["elapsed"]]
  
  pred_cv_las <- predict_linear(X_test, fit_cv_las$beta)
  out_rows[[length(out_rows) + 1]] <- make_result_row(
    n_train = n_train, rho = rho, method = "CV-LASSO",
    beta_hat = fit_cv_las$beta, y_test = y_test, pred_test = pred_cv_las,
    related_idx = related_idx, unrelated_idx = unrelated_idx,
    lambda_value = fit_cv_las$lambda, sigma_value = NA_real_,
    elapsed_time = time_cv_las, iterations = fit_cv_las$iterations,
    include_selection = TRUE
  )
  
  t0 <- proc.time()
  fit_ql_025 <- fit_quantile_lasso_cv(X_train, y_train, 0.25, lambda_grid_ql)
  time_ql_025 <- (proc.time() - t0)[["elapsed"]]
  
  pred_ql_025 <- predict_linear(X_test, fit_ql_025$beta, intercept = fit_ql_025$intercept)
  out_rows[[length(out_rows) + 1]] <- make_result_row(
    n_train = n_train, rho = rho, method = "QL-0.25",
    beta_hat = fit_ql_025$beta, y_test = y_test, pred_test = pred_ql_025,
    related_idx = related_idx, unrelated_idx = unrelated_idx,
    lambda_value = fit_ql_025$lambda, sigma_value = NA_real_,
    elapsed_time = time_ql_025, iterations = fit_ql_025$iterations,
    include_selection = TRUE
  )
  
  t0 <- proc.time()
  fit_ql_050 <- fit_quantile_lasso_cv(X_train, y_train, 0.5, lambda_grid_ql)
  time_ql_050 <- (proc.time() - t0)[["elapsed"]]
  
  pred_ql_050 <- predict_linear(X_test, fit_ql_050$beta, intercept = fit_ql_050$intercept)
  out_rows[[length(out_rows) + 1]] <- make_result_row(
    n_train = n_train, rho = rho, method = "QL-0.5",
    beta_hat = fit_ql_050$beta, y_test = y_test, pred_test = pred_ql_050,
    related_idx = related_idx, unrelated_idx = unrelated_idx,
    lambda_value = fit_ql_050$lambda, sigma_value = NA_real_,
    elapsed_time = time_ql_050, iterations = fit_ql_050$iterations,
    include_selection = TRUE
  )
  
  t0 <- proc.time()
  fit_ql_075 <- fit_quantile_lasso_cv(X_train, y_train, 0.75, lambda_grid_ql)
  time_ql_075 <- (proc.time() - t0)[["elapsed"]]
  
  pred_ql_075 <- predict_linear(X_test, fit_ql_075$beta, intercept = fit_ql_075$intercept)
  out_rows[[length(out_rows) + 1]] <- make_result_row(
    n_train = n_train, rho = rho, method = "QL-0.75",
    beta_hat = fit_ql_075$beta, y_test = y_test, pred_test = pred_ql_075,
    related_idx = related_idx, unrelated_idx = unrelated_idx,
    lambda_value = fit_ql_075$lambda, sigma_value = NA_real_,
    elapsed_time = time_ql_075, iterations = fit_ql_075$iterations,
    include_selection = TRUE
  )
  
  t0 <- proc.time()
  fit_nor_model <- fit_nor(X_train, y_train)
  time_nor <- (proc.time() - t0)[["elapsed"]]
  
  pred_nor <- predict_linear(X_test, fit_nor_model$beta)
  out_rows[[length(out_rows) + 1]] <- make_result_row(
    n_train = n_train, rho = rho, method = "NOR",
    beta_hat = fit_nor_model$beta, y_test = y_test, pred_test = pred_nor,
    related_idx = related_idx, unrelated_idx = unrelated_idx,
    lambda_value = 0, sigma_value = 1,
    elapsed_time = time_nor, iterations = fit_nor_model$iterations,
    include_selection = FALSE
  )
  
  bind_rows(out_rows)
}


###############################
# 9) ACLR hyperparameter tuning
###############################

run_one_replication_aclr_only <- function(n_train, n_test, rho, beta_true,
                                          lambda_grid_aclr,
                                          adaptive_gamma, adaptive_eps,
                                          huber_k, lambda_init_huber_ridge) {
  dat <- generate_example3_data(
    n_train = n_train,
    n_test = n_test,
    p = length(beta_true),
    rho = rho,
    beta_true = beta_true
  )
  
  X_train <- dat$X_train
  y_train <- dat$y_train
  X_test  <- dat$X_test
  y_test  <- dat$y_test
  
  out_rows <- list()
  
  fit_aic_aclr <- fit_aclr_fast_select_lambda(
    X = X_train,
    y = y_train,
    lambda_grid = lambda_grid_aclr,
    criterion = "AIC",
    gamma = adaptive_gamma,
    eps = adaptive_eps,
    huber_k = huber_k,
    lambda_init = lambda_init_huber_ridge
  )
  pred_aic_aclr <- predict_linear(X_test, fit_aic_aclr$beta)
  out_rows[[1]] <- make_result_row(
    n_train = n_train, rho = rho, method = "AIC-ACLR-Huber",
    beta_hat = fit_aic_aclr$beta, y_test = y_test, pred_test = pred_aic_aclr,
    related_idx = related_idx, unrelated_idx = unrelated_idx,
    lambda_value = fit_aic_aclr$lambda, sigma_value = fit_aic_aclr$sigma,
    elapsed_time = NA_real_, iterations = fit_aic_aclr$iterations,
    include_selection = TRUE
  )
  
  fit_bic_aclr <- fit_aclr_fast_select_lambda(
    X = X_train,
    y = y_train,
    lambda_grid = lambda_grid_aclr,
    criterion = "BIC",
    gamma = adaptive_gamma,
    eps = adaptive_eps,
    huber_k = huber_k,
    lambda_init = lambda_init_huber_ridge
  )
  pred_bic_aclr <- predict_linear(X_test, fit_bic_aclr$beta)
  out_rows[[2]] <- make_result_row(
    n_train = n_train, rho = rho, method = "BIC-ACLR-Huber",
    beta_hat = fit_bic_aclr$beta, y_test = y_test, pred_test = pred_bic_aclr,
    related_idx = related_idx, unrelated_idx = unrelated_idx,
    lambda_value = fit_bic_aclr$lambda, sigma_value = fit_bic_aclr$sigma,
    elapsed_time = NA_real_, iterations = fit_bic_aclr$iterations,
    include_selection = TRUE
  )
  
  dplyr::bind_rows(out_rows)
}

run_aclr_grid_tuning <- function(aclr_grid, tuning_n_rep) {
  tuning_results <- list()
  counter <- 1
  
  for (g in seq_len(nrow(aclr_grid))) {
    grid_row <- aclr_grid[g, ]
    lambda_grid_aclr_g <- make_lambda_grid_aclr(
      lambda_min = grid_row$lambda_min,
      lambda_max = grid_row$lambda_max,
      lambda_len = grid_row$lambda_len
    )
    
    rows_g <- list()
    inner_counter <- 1
    
    for (n_train in n_train_values) {
      for (rho in rho_values) {
        for (b in seq_len(tuning_n_rep)) {
          one_rep <- run_one_replication_aclr_only(
            n_train = n_train,
            n_test = n_test,
            rho = rho,
            beta_true = beta_true,
            lambda_grid_aclr = lambda_grid_aclr_g,
            adaptive_gamma = grid_row$adaptive_gamma,
            adaptive_eps = grid_row$adaptive_eps,
            huber_k = grid_row$huber_k,
            lambda_init_huber_ridge = lambda_init_huber_ridge
          )
          one_rep$replication <- b
          rows_g[[inner_counter]] <- one_rep
          inner_counter <- inner_counter + 1
        }
      }
    }
    
    grid_res <- dplyr::bind_rows(rows_g)
    grid_summary <- grid_res %>%
      dplyr::group_by(method) %>%
      dplyr::summarise(
        mean_fnr = mean(fnr, na.rm = TRUE),
        mean_fpr = mean(fpr, na.rm = TRUE),
        mean_mae = mean(mae_test, na.rm = TRUE),
        mean_selected = mean(selected_count, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      dplyr::mutate(
        tuning_score = mean_fnr + 0.50 * mean_fpr + 0.05 * mean_mae,
        grid_id = g,
        adaptive_gamma = grid_row$adaptive_gamma,
        adaptive_eps = grid_row$adaptive_eps,
        huber_k = grid_row$huber_k,
        lambda_min = grid_row$lambda_min,
        lambda_max = grid_row$lambda_max,
        lambda_len = grid_row$lambda_len
      )
    
    tuning_results[[counter]] <- grid_summary
    counter <- counter + 1
    cat("Finished ACLR tuning grid", g, "of", nrow(aclr_grid), "\n")
  }
  
  dplyr::bind_rows(tuning_results)
}

cat("Starting ACLR hyperparameter tuning for Example 3...\n")
aclr_tuning_summary <- run_aclr_grid_tuning(aclr_grid, tuning_n_rep)

best_aclr_setting <- aclr_tuning_summary %>%
  dplyr::group_by(grid_id, adaptive_gamma, adaptive_eps, huber_k, lambda_min, lambda_max, lambda_len) %>%
  dplyr::summarise(
    overall_fnr = mean(mean_fnr, na.rm = TRUE),
    overall_fpr = mean(mean_fpr, na.rm = TRUE),
    overall_mae = mean(mean_mae, na.rm = TRUE),
    overall_score = mean(tuning_score, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(overall_score) %>%
  dplyr::slice(1)

print(aclr_tuning_summary %>% dplyr::arrange(tuning_score) %>% utils::head(10))
cat("Best common ACLR setting selected by tuning:\n")
print(best_aclr_setting)

adaptive_gamma <- best_aclr_setting$adaptive_gamma[1]
adaptive_eps   <- best_aclr_setting$adaptive_eps[1]
huber_k        <- best_aclr_setting$huber_k[1]
lambda_grid_aclr <- make_lambda_grid_aclr(
  lambda_min = best_aclr_setting$lambda_min[1],
  lambda_max = best_aclr_setting$lambda_max[1],
  lambda_len = best_aclr_setting$lambda_len[1]
)

utils::write.csv(aclr_tuning_summary, "aclr_tuning_summary_example3.csv", row.names = FALSE)
utils::write.csv(best_aclr_setting, "best_aclr_setting_example3.csv", row.names = FALSE)

######################################
# 10) full simulation using tuned ACLR
######################################

all_results <- list()
counter <- 1

all_method_names <- c(
  "AIC-CLR", "BIC-CLR", "AIC-ACLR-Huber", "BIC-ACLR-Huber",
  "CV-LASSO", "QL-0.25", "QL-0.5", "QL-0.75", "NOR"
)

total_time_by_method <- setNames(rep(0, length(all_method_names)), all_method_names)

for (n_train in n_train_values) {
  for (rho in rho_values) {
    for (b in 1:n_rep) {
      one_rep <- run_one_replication_all_methods(
        n_train = n_train,
        n_test = n_test,
        rho = rho,
        beta_true = beta_true,
        lambda_grid_clr = lambda_grid_clr,
        lambda_grid_ql = lambda_grid_ql,
        lambda_grid_aclr = lambda_grid_aclr,
        adaptive_gamma = adaptive_gamma,
        adaptive_eps = adaptive_eps,
        huber_k = huber_k,
        lambda_init_huber_ridge = lambda_init_huber_ridge
      )
      
      one_rep$replication <- b
      
      for (m in names(total_time_by_method)) {
        total_time_by_method[m] <- total_time_by_method[m] + sum(one_rep$time_sec[one_rep$method == m], na.rm = TRUE)
      }
      
      all_results[[counter]] <- one_rep
      counter <- counter + 1
      
      if (b %% report_every == 0) {
        cat("n_train =", n_train, "rho =", rho, "rep =", b, "of", n_rep, "\n")
      }
    }
  }
}

results_all <- bind_rows(all_results)

selection_methods <- c("AIC-CLR", "BIC-CLR", "AIC-ACLR-Huber", "BIC-ACLR-Huber", "CV-LASSO", "QL-0.25", "QL-0.5", "QL-0.75")
mae_methods <- c(selection_methods, "NOR")

results_all$method <- factor(results_all$method, levels = mae_methods)
results_all$n_train <- factor(results_all$n_train, levels = c(50, 100, 200))

results_selection <- results_all %>% filter(method %in% selection_methods)
results_mae <- results_all %>% filter(method %in% mae_methods)

plot_selected <- ggplot(results_selection, aes(x = method, y = selected_count, fill = factor(rho))) +
  geom_boxplot() +
  facet_wrap(~ n_train, ncol = 1) +
  labs(x = "Selection Method", y = "Number of Selected Covariates", fill = expression(rho)) +
  theme_bw(base_size = 12)
print(plot_selected)

plot_unrelated <- ggplot(results_selection, aes(x = method, y = unrelated_selected_count, fill = factor(rho))) +
  geom_boxplot() +
  facet_wrap(~ n_train, ncol = 1) +
  labs(x = "Selection Method", y = "Number of Selected Unrelated Covariates", fill = expression(rho)) +
  theme_bw(base_size = 12)
print(plot_unrelated)

plot_mae <- ggplot(results_mae, aes(x = method, y = mae_test, fill = factor(rho))) +
  geom_boxplot() +
  facet_wrap(~ n_train, ncol = 1, scales = "free_y") +
  labs(x = "Method", y = "Test MAE", fill = expression(rho)) +
  theme_bw(base_size = 12)
print(plot_mae)

mae_heatmap_data <- results_mae %>%
  group_by(n_train, rho, method) %>%
  summarise(mean_mae = mean(mae_test), .groups = "drop") %>%
  mutate(scenario = paste0("n=", n_train, ", rho=", rho))

plot_mae_heatmap <- ggplot(mae_heatmap_data, aes(x = method, y = scenario, fill = mean_mae)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", mean_mae)), size = 3) +
  labs(x = "Method", y = "Scenario", fill = "Mean MAE") +
  theme_bw(base_size = 11)
print(plot_mae_heatmap)

fnr_table <- results_selection %>%
  group_by(rho, n_train, method) %>%
  summarise(FNR = mean(fnr), .groups = "drop") %>%
  pivot_wider(names_from = method, values_from = FNR) %>%
  arrange(rho, n_train)
print(fnr_table)


fpr_table <- results_selection %>%
  group_by(rho, n_train, method) %>%
  summarise(FPR = mean(fpr), .groups = "drop") %>%
  pivot_wider(names_from = method, values_from = FPR) %>%
  arrange(rho, n_train)
print(fpr_table)

mae_table <- results_mae %>%
  group_by(rho, n_train, method) %>%
  summarise(
    mean_mae = mean(mae_test),
    sd_mae = sd(mae_test),
    mae_text = sprintf("%.3f (%.3f)", mean_mae, sd_mae),
    .groups = "drop"
  ) %>%
  select(rho, n_train, method, mae_text) %>%
  pivot_wider(names_from = method, values_from = mae_text) %>%
  arrange(rho, n_train)
print(mae_table)

training_time_table <- data.frame(
  method = names(total_time_by_method),
  total_time_minutes = round(as.numeric(total_time_by_method) / 60, 3)
)
print(training_time_table)

selection_summary <- results_selection %>%
  group_by(n_train, rho, method) %>%
  summarise(
    mean_selected = mean(selected_count),
    median_selected = median(selected_count),
    mean_unrelated = mean(unrelated_selected_count),
    median_unrelated = median(unrelated_selected_count),
    mean_fnr = mean(fnr),
    mean_fpr = mean(fpr),
    mean_mae = mean(mae_test),
    mean_time_sec = mean(time_sec),
    .groups = "drop"
  )
print(selection_summary)

mae_summary <- results_mae %>%
  group_by(n_train, rho, method) %>%
  summarise(
    mean_mae = mean(mae_test),
    median_mae = median(mae_test),
    mean_time_sec = mean(time_sec),
    .groups = "drop"
  )
print(mae_summary)

out_dir <- "Example3_ACLR_Huber_results"
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

write.csv(results_all, file.path(out_dir, "example3_all_methods_results_full.csv"), row.names = FALSE)
write.csv(fnr_table, file.path(out_dir, "example3_fnr_table.csv"), row.names = FALSE)
write.csv(fpr_table, file.path(out_dir, "example3_fpr_table.csv"), row.names = FALSE)
write.csv(mae_table, file.path(out_dir, "example3_mae_table_with_NOR.csv"), row.names = FALSE)
write.csv(training_time_table, file.path(out_dir, "example3_training_time_minutes.csv"), row.names = FALSE)
write.csv(selection_summary, file.path(out_dir, "example3_selection_summary.csv"), row.names = FALSE)
write.csv(mae_summary, file.path(out_dir, "example3_mae_summary.csv"), row.names = FALSE)

ggsave(file.path(out_dir, "example3_selected_boxplot.png"), plot_selected, width = 12, height = 8, dpi = 300)
ggsave(file.path(out_dir, "example3_unrelated_boxplot.png"), plot_unrelated, width = 12, height = 8, dpi = 300)
ggsave(file.path(out_dir, "example3_mae_boxplot_with_NOR.png"), plot_mae, width = 12, height = 8, dpi = 300)
ggsave(file.path(out_dir, "example3_mae_heatmap.png"), plot_mae_heatmap, width = 11, height = 6, dpi = 300)

cat("\nSaved all outputs to:", normalizePath(out_dir), "\n")
cat("\n===== Whole training time for each model (minutes) =====\n")
print(training_time_table)

