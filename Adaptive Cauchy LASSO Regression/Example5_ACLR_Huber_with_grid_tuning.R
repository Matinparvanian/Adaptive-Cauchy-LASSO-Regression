.libPaths("~/R/4.3.1")
# ============================================================
# Fast high-dimensional ACLR simulation: p >> n
# Version: small ACLR tuning + final run
# Main changes from the long version:
#   1) Small representative tuning only
#   2) AIC-ACLR and BIC-ACLR selected from ONE ACLR path
#   3) AIC-CLR and BIC-CLR selected from ONE CLR path
#   4) Reduced ACLR outer iterations
#   5) Progress saved after each scenario
#   6) QL-0.25, QL-0.5, and QL-0.75 added
#   7) Boxplots saved for FNR, FPR, MAE, selected variables, and time
# ============================================================

set.seed(123)

# ============================================================
# 0) Packages
# ============================================================
required_packages <- c("MASS", "glmnet", "quantreg", "dplyr", "tidyr", "ggplot2")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# ============================================================
# 1) Global settings
# ============================================================

p_values <- c(100, 300, 500)
n_train_values <- c(50, 100)
rho_values <- c(0, 0.5, 0.75)

n_test <- 30

# Final simulation repetitions.
# Use 100 first. After checking results, increase to 500 if needed.
n_rep <- 500
report_every <- 500

# Small preliminary tuning.
tuning_n_rep <- 10

# Representative tuning scenarios only.
p_values_tune <- c(300)
n_train_values_tune <- c(100)
rho_values_tune <- c(0.5, 0.75)

n_active <- 5
active_beta_value <- 4

coef_tol <- 0.4
eps_num <- 1e-8
sigma_cauchy <- 1

# CLR lambda grid. Reduced from 300 values to 120 values.
lambda_grid_clr <- seq(0.01, 3, length.out = 120)

# Quantile LASSO lambda grid.
lambda_grid_ql <- exp(seq(log(0.01), log(20), length.out = 25))

# Small ACLR tuning grid.
aclr_grid <- expand.grid(
  adaptive_gamma = c(0.4, 0.6, 0.9, 1),
  adaptive_eps   = c(0.05, 0.10),
  lambda_min     = c(0.001),
  lambda_max     = c(3),
  lambda_len     = c(25),
  stringsAsFactors = FALSE
)
aclr_grid$grid_id <- seq_len(nrow(aclr_grid))

# Tuning score.
tuning_score_function <- function(fnr, fpr, mae) {
  fnr + 0.5 * fpr + 0.1 * mae
}

make_lambda_grid_aclr <- function(lambda_min, lambda_max, lambda_len) {
  exp(seq(log(lambda_min), log(lambda_max), length.out = lambda_len))
}

lambda_init_ridge <- NULL

out_dir <- "Example5_ACLR_Huber_results_fast"
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

# ============================================================
# 2) Data generation
# ============================================================

make_gamma <- function(p, rho) {
  outer(1:p, 1:p, function(i, j) rho^abs(i - j))
}

make_beta_true <- function(p, n_active = 5, active_beta_value = 4) {
  c(rep(active_beta_value, n_active), rep(0, p - n_active))
}

generate_highdim_data <- function(n_train, n_test, p, rho, beta_true, sigma_cauchy = 1) {
  Gamma <- make_gamma(p, rho)

  X_train <- MASS::mvrnorm(n = n_train, mu = rep(0, p), Sigma = Gamma)
  X_test  <- MASS::mvrnorm(n = n_test,  mu = rep(0, p), Sigma = Gamma)

  eps_train <- rcauchy(n_train, location = 0, scale = sigma_cauchy)
  eps_test  <- rcauchy(n_test,  location = 0, scale = sigma_cauchy)

  y_train <- as.numeric(X_train %*% beta_true + eps_train)
  y_test  <- as.numeric(X_test  %*% beta_true + eps_test)

  X_train <- scale(X_train, center = TRUE, scale = FALSE)
  X_test  <- scale(X_test,  center = TRUE, scale = FALSE)
  y_train <- as.numeric(scale(y_train, center = TRUE, scale = FALSE))
  y_test  <- as.numeric(scale(y_test,  center = TRUE, scale = FALSE))

  list(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)
}

# ============================================================
# 3) Utilities
# ============================================================

predict_linear <- function(X, beta, intercept = 0) {
  as.numeric(intercept + X %*% beta)
}

compute_mae <- function(y_true, y_pred) {
  mean(abs(y_true - y_pred))
}

compute_selection_metrics <- function(beta_hat, active_idx, inactive_idx, tol = coef_tol) {
  selected <- abs(beta_hat) > tol

  list(
    selected_count = sum(selected),
    inactive_selected_count = sum(selected[inactive_idx]),
    fnr = mean(!selected[active_idx]) * 100,
    fpr = mean(selected[inactive_idx]) * 100
  )
}

make_result_row <- function(p, n_train, rho, method, beta_hat, y_test, pred_test,
                            active_idx, inactive_idx,
                            lambda_value = NA_real_, sigma_value = NA_real_,
                            elapsed_time = NA_real_) {
  sel <- compute_selection_metrics(beta_hat, active_idx, inactive_idx, tol = coef_tol)

  data.frame(
    p = p,
    n_train = n_train,
    rho = rho,
    method = method,
    selected_count = sel$selected_count,
    inactive_selected_count = sel$inactive_selected_count,
    fnr = sel$fnr,
    fpr = sel$fpr,
    mae_test = compute_mae(y_test, pred_test),
    lambda = lambda_value,
    sigma = sigma_value,
    time_sec = elapsed_time
  )
}

robust_sigma_update <- function(residuals, min_sigma = 1e-4) {
  s <- mad(residuals, center = 0, constant = 1.4826, na.rm = TRUE)
  if (!is.finite(s) || is.na(s) || s < min_sigma) {
    s <- sqrt(mean(residuals^2))
  }
  max(s, min_sigma)
}

estimate_sigma_given_beta <- function(residuals, lower = 1e-4, upper = 100) {
  n <- length(residuals)

  negloglik_sigma <- function(sigma) {
    sigma <- max(sigma, 1e-8)
    n * log(sigma) + sum(log(pmax(sigma^2 + residuals^2, eps_num)))
  }

  opt <- optimize(negloglik_sigma, interval = c(lower, upper))
  opt$minimum
}

# ============================================================
# 4) CLR path: select AIC and BIC from same fit
# ============================================================

fit_clr_glmnet_path <- function(X, y, lambda_grid) {
  fit <- glmnet::glmnet(
    x = X,
    y = y,
    alpha = 1,
    lambda = lambda_grid,
    intercept = FALSE,
    standardize = FALSE
  )

  beta_path <- as.matrix(coef(fit))[-1, , drop = FALSE]
  beta_path <- t(beta_path)

  list(lambda = fit$lambda, Beta = beta_path)
}

compute_clr_ic_plugin_sigma <- function(X, y, beta_path) {
  rp <- nrow(beta_path)
  n <- length(y)

  dp <- rep(0, rp)
  sigma_hat <- rep(0, rp)
  fit_term <- rep(0, rp)
  aic <- rep(0, rp)
  bic <- rep(0, rp)

  for (i in seq_len(rp)) {
    beta_i <- beta_path[i, ]
    dp[i] <- sum(abs(beta_i) > coef_tol)

    r <- as.numeric(y - X %*% beta_i)
    sigma_hat[i] <- estimate_sigma_given_beta(r)
    fit_term[i] <- sum(log(pmax(sigma_hat[i]^2 + r^2, eps_num)))

    aic[i] <- (fit_term[i] + 2 * dp[i]) / n
    bic[i] <- (fit_term[i] + log(n) * dp[i]) / n
  }

  list(df = dp, sigma = sigma_hat, AIC = aic, BIC = bic)
}

fit_clr_select_aic_bic <- function(X, y, lambda_grid) {
  path <- fit_clr_glmnet_path(X, y, lambda_grid)
  ic <- compute_clr_ic_plugin_sigma(X, y, path$Beta)

  idx_aic <- which.min(ic$AIC)
  idx_bic <- which.min(ic$BIC)

  beta_aic <- path$Beta[idx_aic, ]
  beta_bic <- path$Beta[idx_bic, ]
  beta_aic[abs(beta_aic) <= coef_tol] <- 0
  beta_bic[abs(beta_bic) <= coef_tol] <- 0

  list(
    AIC = list(beta = beta_aic, lambda = path$lambda[idx_aic], sigma = ic$sigma[idx_aic]),
    BIC = list(beta = beta_bic, lambda = path$lambda[idx_bic], sigma = ic$sigma[idx_bic])
  )
}

# ============================================================
# 5) ACLR path: select AIC and BIC from same fit
# ============================================================

fit_ridge_initial <- function(X, y, lambda_init = NULL) {
  p <- ncol(X)

  if (is.null(lambda_init)) {
    lambda_init <- 1 / max(1, nrow(X))
  }

  beta_init <- tryCatch({
    fit <- glmnet::glmnet(
      x = X,
      y = y,
      alpha = 0,
      lambda = lambda_init,
      intercept = FALSE,
      standardize = FALSE
    )
    as.numeric(coef(fit))[-1]
  }, error = function(e) rep(0, p))

  beta_init[!is.finite(beta_init)] <- 0
  beta_init
}

make_adaptive_weights <- function(beta_init, gamma = 1, eps = 1e-3) {
  1 / (abs(beta_init) + eps)^gamma
}

aclr_objective <- function(beta, X, y, lambda, sigma, weights) {
  r <- as.numeric(y - X %*% beta)
  sum(log(pmax(sigma^2 + r^2, eps_num))) + lambda * sum(weights * abs(beta))
}

fit_aclr_fixed_lambda <- function(X, y, lambda, weights,
                                  beta_init = NULL,
                                  sigma_init = NULL,
                                  max_outer = 20,
                                  tol = 1e-5) {
  p <- ncol(X)

  beta <- if (is.null(beta_init)) rep(0, p) else beta_init
  sigma <- if (is.null(sigma_init)) robust_sigma_update(y) else sigma_init

  old_obj <- aclr_objective(beta, X, y, lambda, sigma, weights)

  for (iter in seq_len(max_outer)) {
    beta_prev <- beta
    sigma_prev <- sigma

    r_prev <- as.numeric(y - X %*% beta_prev)
    obs_w <- 1 / pmax(sigma_prev^2 + r_prev^2, eps_num)

    sqrt_w <- sqrt(obs_w)
    X_w <- X * sqrt_w
    y_w <- y * sqrt_w

    fit <- glmnet::glmnet(
      x = X_w,
      y = y_w,
      alpha = 1,
      lambda = lambda,
      intercept = FALSE,
      standardize = FALSE,
      penalty.factor = weights
    )

    beta <- as.numeric(coef(fit))[-1]
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

  list(beta = beta, sigma = sigma, objective = old_obj)
}

compute_ic_aclr <- function(X, y, beta_hat, sigma_hat) {
  e <- as.numeric(y - X %*% beta_hat)
  df <- sum(abs(beta_hat) > coef_tol)
  n <- length(y)

  fit_term <- sum(log(pmax(sigma_hat^2 + e^2, eps_num)))

  list(
    df = df,
    AIC = (fit_term + 2 * df) / n,
    BIC = (fit_term + log(n) * df) / n
  )
}

fit_aclr_select_aic_bic <- function(X, y, lambda_grid,
                                    gamma = 1,
                                    eps = 1e-3,
                                    lambda_init = NULL) {
  beta_init0 <- fit_ridge_initial(X, y, lambda_init = lambda_init)
  weights <- make_adaptive_weights(beta_init0, gamma = gamma, eps = eps)

  beta_start <- beta_init0
  sigma_start <- robust_sigma_update(y)

  fits <- vector("list", length(lambda_grid))

  for (k in seq_along(lambda_grid)) {
    lam <- lambda_grid[k]

    fit_k <- fit_aclr_fixed_lambda(
      X = X,
      y = y,
      lambda = lam,
      weights = weights,
      beta_init = beta_start,
      sigma_init = sigma_start,
      max_outer = 20
    )

    ic_k <- compute_ic_aclr(X, y, fit_k$beta, fit_k$sigma)

    fits[[k]] <- list(
      beta = fit_k$beta,
      sigma = fit_k$sigma,
      lambda = lam,
      AIC = ic_k$AIC,
      BIC = ic_k$BIC
    )

    beta_start <- fit_k$beta
    sigma_start <- fit_k$sigma
  }

  aic_values <- sapply(fits, function(z) z$AIC)
  bic_values <- sapply(fits, function(z) z$BIC)

  idx_aic <- which.min(aic_values)
  idx_bic <- which.min(bic_values)

  list(AIC = fits[[idx_aic]], BIC = fits[[idx_bic]])
}


# ============================================================
# 6) Quantile LASSO methods
# ============================================================

check_loss <- function(u, tau) {
  u * (tau - (u < 0))
}

fit_quantile_lasso_cv <- function(X, y, tau, lambda_grid, nfolds = 5) {
  n <- nrow(X)
  p <- ncol(X)

  fold_id <- sample(rep(seq_len(nfolds), length.out = n))
  cv_errors <- rep(Inf, length(lambda_grid))

  for (k in seq_along(lambda_grid)) {
    lam <- lambda_grid[k]
    fold_losses <- rep(Inf, nfolds)

    for (fold in seq_len(nfolds)) {
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
        pred_va <- as.numeric(X_va_int %*% coef_hat)
        fold_losses[fold] <- mean(check_loss(y_va - pred_va, tau), na.rm = TRUE)
      }
    }

    cv_errors[k] <- mean(fold_losses, na.rm = TRUE)
  }

  if (all(!is.finite(cv_errors))) {
    return(list(intercept = 0, beta = rep(0, p), lambda = NA_real_))
  }

  best_lambda <- lambda_grid[which.min(cv_errors)]

  final_fit <- tryCatch({
    quantreg::rq.fit.lasso(
      x = cbind(1, X),
      y = y,
      tau = tau,
      lambda = c(0, rep(best_lambda, p))
    )
  }, error = function(e) NULL)

  if (is.null(final_fit)) {
    return(list(intercept = 0, beta = rep(0, p), lambda = best_lambda))
  }

  coef_hat <- as.numeric(final_fit$coefficients)
  intercept_hat <- coef_hat[1]
  beta_hat <- coef_hat[-1]
  beta_hat[!is.finite(beta_hat)] <- 0
  beta_hat[abs(beta_hat) <= coef_tol] <- 0

  list(intercept = intercept_hat, beta = beta_hat, lambda = best_lambda)
}

# ============================================================
# 7) CV-LASSO
# ============================================================

fit_cv_lasso <- function(X, y) {
  fit <- glmnet::cv.glmnet(
    x = X,
    y = y,
    alpha = 1,
    intercept = FALSE,
    standardize = FALSE
  )

  beta_hat <- as.numeric(coef(fit, s = "lambda.min"))[-1]
  beta_hat[abs(beta_hat) <= coef_tol] <- 0

  list(beta = beta_hat, lambda = fit$lambda.min)
}

# ============================================================
# 8) One replication
# ============================================================

run_one_replication <- function(p, n_train, n_test, rho,
                                adaptive_gamma, adaptive_eps, lambda_grid_aclr) {
  beta_true <- make_beta_true(p, n_active = n_active, active_beta_value = active_beta_value)
  active_idx <- seq_len(n_active)
  inactive_idx <- (n_active + 1):p

  dat <- generate_highdim_data(n_train, n_test, p, rho, beta_true, sigma_cauchy)

  X_train <- dat$X_train
  y_train <- dat$y_train
  X_test <- dat$X_test
  y_test <- dat$y_test

  out <- list()

  # ACLR: one path gives both AIC and BIC.
  t0 <- proc.time()
  aclr_fit <- fit_aclr_select_aic_bic(
    X = X_train,
    y = y_train,
    lambda_grid = lambda_grid_aclr,
    gamma = adaptive_gamma,
    eps = adaptive_eps,
    lambda_init = lambda_init_ridge
  )
  elapsed <- (proc.time() - t0)[["elapsed"]]

  pred <- predict_linear(X_test, aclr_fit$AIC$beta)
  out[[length(out) + 1]] <- make_result_row(
    p, n_train, rho, "AIC-ACLR", aclr_fit$AIC$beta, y_test, pred,
    active_idx, inactive_idx, aclr_fit$AIC$lambda, aclr_fit$AIC$sigma, elapsed
  )

  pred <- predict_linear(X_test, aclr_fit$BIC$beta)
  out[[length(out) + 1]] <- make_result_row(
    p, n_train, rho, "BIC-ACLR", aclr_fit$BIC$beta, y_test, pred,
    active_idx, inactive_idx, aclr_fit$BIC$lambda, aclr_fit$BIC$sigma, elapsed
  )

  # CLR: one path gives both AIC and BIC.
  t0 <- proc.time()
  clr_fit <- fit_clr_select_aic_bic(X_train, y_train, lambda_grid_clr)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  pred <- predict_linear(X_test, clr_fit$AIC$beta)
  out[[length(out) + 1]] <- make_result_row(
    p, n_train, rho, "AIC-CLR", clr_fit$AIC$beta, y_test, pred,
    active_idx, inactive_idx, clr_fit$AIC$lambda, clr_fit$AIC$sigma, elapsed
  )

  pred <- predict_linear(X_test, clr_fit$BIC$beta)
  out[[length(out) + 1]] <- make_result_row(
    p, n_train, rho, "BIC-CLR", clr_fit$BIC$beta, y_test, pred,
    active_idx, inactive_idx, clr_fit$BIC$lambda, clr_fit$BIC$sigma, elapsed
  )

  # CV-LASSO.
  t0 <- proc.time()
  lasso_fit <- fit_cv_lasso(X_train, y_train)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  pred <- predict_linear(X_test, lasso_fit$beta)
  out[[length(out) + 1]] <- make_result_row(
    p, n_train, rho, "CV-LASSO", lasso_fit$beta, y_test, pred,
    active_idx, inactive_idx, lasso_fit$lambda, NA_real_, elapsed
  )

  # Quantile LASSO tau = 0.25.
  t0 <- proc.time()
  ql_fit <- fit_quantile_lasso_cv(X_train, y_train, tau = 0.25, lambda_grid = lambda_grid_ql)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  pred <- predict_linear(X_test, ql_fit$beta, intercept = ql_fit$intercept)
  out[[length(out) + 1]] <- make_result_row(
    p, n_train, rho, "QL-0.25", ql_fit$beta, y_test, pred,
    active_idx, inactive_idx, ql_fit$lambda, NA_real_, elapsed
  )

  # Quantile LASSO tau = 0.50.
  t0 <- proc.time()
  ql_fit <- fit_quantile_lasso_cv(X_train, y_train, tau = 0.50, lambda_grid = lambda_grid_ql)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  pred <- predict_linear(X_test, ql_fit$beta, intercept = ql_fit$intercept)
  out[[length(out) + 1]] <- make_result_row(
    p, n_train, rho, "QL-0.5", ql_fit$beta, y_test, pred,
    active_idx, inactive_idx, ql_fit$lambda, NA_real_, elapsed
  )

  # Quantile LASSO tau = 0.75.
  t0 <- proc.time()
  ql_fit <- fit_quantile_lasso_cv(X_train, y_train, tau = 0.75, lambda_grid = lambda_grid_ql)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  pred <- predict_linear(X_test, ql_fit$beta, intercept = ql_fit$intercept)
  out[[length(out) + 1]] <- make_result_row(
    p, n_train, rho, "QL-0.75", ql_fit$beta, y_test, pred,
    active_idx, inactive_idx, ql_fit$lambda, NA_real_, elapsed
  )

  dplyr::bind_rows(out)
}

# ============================================================
# 9) Small ACLR tuning
# ============================================================

run_one_replication_aclr_only <- function(p, n_train, n_test, rho,
                                          gamma_now, eps_now, lambda_grid_now) {
  beta_true <- make_beta_true(p, n_active = n_active, active_beta_value = active_beta_value)
  active_idx <- seq_len(n_active)
  inactive_idx <- (n_active + 1):p

  dat <- generate_highdim_data(n_train, n_test, p, rho, beta_true, sigma_cauchy)

  fit <- fit_aclr_select_aic_bic(
    X = dat$X_train,
    y = dat$y_train,
    lambda_grid = lambda_grid_now,
    gamma = gamma_now,
    eps = eps_now,
    lambda_init = lambda_init_ridge
  )

  out <- list()

  pred <- predict_linear(dat$X_test, fit$AIC$beta)
  out[[1]] <- make_result_row(
    p, n_train, rho, "AIC-ACLR", fit$AIC$beta, dat$y_test, pred,
    active_idx, inactive_idx, fit$AIC$lambda, fit$AIC$sigma, NA_real_
  )

  pred <- predict_linear(dat$X_test, fit$BIC$beta)
  out[[2]] <- make_result_row(
    p, n_train, rho, "BIC-ACLR", fit$BIC$beta, dat$y_test, pred,
    active_idx, inactive_idx, fit$BIC$lambda, fit$BIC$sigma, NA_real_
  )

  dplyr::bind_rows(out)
}

run_aclr_grid_tuning <- function() {
  tuning_summary_list <- list()

  for (g in seq_len(nrow(aclr_grid))) {
    grid_row <- aclr_grid[g, ]
    lambda_grid_now <- make_lambda_grid_aclr(
      lambda_min = grid_row$lambda_min,
      lambda_max = grid_row$lambda_max,
      lambda_len = grid_row$lambda_len
    )

    cat("\nTuning grid", grid_row$grid_id, "of", nrow(aclr_grid), "\n")
    flush.console()

    grid_raw <- list()
    grid_counter <- 1

    for (p_now in p_values_tune) {
      for (n_train_now in n_train_values_tune) {
        for (rho_now in rho_values_tune) {
          for (b in seq_len(tuning_n_rep)) {
            one_rep <- run_one_replication_aclr_only(
              p = p_now,
              n_train = n_train_now,
              n_test = n_test,
              rho = rho_now,
              gamma_now = grid_row$adaptive_gamma,
              eps_now = grid_row$adaptive_eps,
              lambda_grid_now = lambda_grid_now
            )

            one_rep$replication <- b
            one_rep$grid_id <- grid_row$grid_id
            one_rep$adaptive_gamma <- grid_row$adaptive_gamma
            one_rep$adaptive_eps <- grid_row$adaptive_eps
            one_rep$lambda_min <- grid_row$lambda_min
            one_rep$lambda_max <- grid_row$lambda_max
            one_rep$lambda_len <- grid_row$lambda_len

            grid_raw[[grid_counter]] <- one_rep
            grid_counter <- grid_counter + 1
          }
        }
      }
    }

    grid_results <- dplyr::bind_rows(grid_raw)

    grid_summary <- grid_results %>%
      dplyr::summarise(
        grid_id = grid_row$grid_id,
        adaptive_gamma = grid_row$adaptive_gamma,
        adaptive_eps = grid_row$adaptive_eps,
        lambda_min = grid_row$lambda_min,
        lambda_max = grid_row$lambda_max,
        lambda_len = grid_row$lambda_len,
        overall_fnr = mean(fnr, na.rm = TRUE),
        overall_fpr = mean(fpr, na.rm = TRUE),
        overall_mae = mean(mae_test, na.rm = TRUE),
        overall_score = tuning_score_function(overall_fnr, overall_fpr, overall_mae)
      )

    tuning_summary_list[[g]] <- grid_summary

    partial_tuning <- dplyr::bind_rows(tuning_summary_list) %>%
      dplyr::arrange(overall_score)

    write.csv(
      partial_tuning,
      file.path(out_dir, "highdim_aclr_tuning_grid_results_partial.csv"),
      row.names = FALSE
    )
  }

  tuning_results <- dplyr::bind_rows(tuning_summary_list) %>%
    dplyr::arrange(overall_score)

  best_grid <- tuning_results[1, ]

  write.csv(tuning_results, file.path(out_dir, "highdim_aclr_tuning_grid_results.csv"), row.names = FALSE)
  write.csv(best_grid, file.path(out_dir, "highdim_aclr_best_grid.csv"), row.names = FALSE)

  list(tuning_results = tuning_results, best_grid = best_grid)
}

# ============================================================
# 10) Run tuning
# ============================================================

tuning_out <- run_aclr_grid_tuning()
tuning_results <- tuning_out$tuning_results
best_grid <- tuning_out$best_grid

cat("\nSelected ACLR hyperparameters:\n")
print(best_grid)
flush.console()

adaptive_gamma <- best_grid$adaptive_gamma
adaptive_eps <- best_grid$adaptive_eps
lambda_grid_aclr <- make_lambda_grid_aclr(
  lambda_min = best_grid$lambda_min,
  lambda_max = best_grid$lambda_max,
  lambda_len = best_grid$lambda_len
)

# ============================================================
# 11) Main simulation loop
# ============================================================

all_results <- list()
counter <- 1

for (p_now in p_values) {
  for (n_train_now in n_train_values) {
    for (rho_now in rho_values) {

      scenario_results <- list()
      scenario_counter <- 1

      cat("\nStarting scenario: p =", p_now,
          "n_train =", n_train_now,
          "rho =", rho_now, "\n")
      flush.console()

      for (b in seq_len(n_rep)) {
        one_rep <- run_one_replication(
          p = p_now,
          n_train = n_train_now,
          n_test = n_test,
          rho = rho_now,
          adaptive_gamma = adaptive_gamma,
          adaptive_eps = adaptive_eps,
          lambda_grid_aclr = lambda_grid_aclr
        )

        one_rep$replication <- b

        all_results[[counter]] <- one_rep
        scenario_results[[scenario_counter]] <- one_rep

        counter <- counter + 1
        scenario_counter <- scenario_counter + 1

        if (b %% report_every == 0) {
          cat("p =", p_now,
              "n_train =", n_train_now,
              "rho =", rho_now,
              "rep =", b,
              "of", n_rep, "\n")
          flush.console()
        }
      }

      # Save scenario-level raw results immediately.
      scenario_df <- dplyr::bind_rows(scenario_results)
      scenario_file <- paste0("raw_p", p_now, "_n", n_train_now, "_rho", rho_now, ".csv")
      write.csv(scenario_df, file.path(out_dir, scenario_file), row.names = FALSE)
    }
  }
}

raw_results <- dplyr::bind_rows(all_results)

# ============================================================
# 12) Summary tables
# ============================================================

summary_results <- raw_results %>%
  dplyr::group_by(p, n_train, rho, method) %>%
  dplyr::summarise(
    mean_fnr = mean(fnr, na.rm = TRUE),
    mean_fpr = mean(fpr, na.rm = TRUE),
    mean_mae = mean(mae_test, na.rm = TRUE),
    sd_mae = sd(mae_test, na.rm = TRUE),
    mean_selected = mean(selected_count, na.rm = TRUE),
    mean_time_sec = mean(time_sec, na.rm = TRUE),
    .groups = "drop"
  )

fnr_table <- summary_results %>%
  dplyr::select(p, n_train, rho, method, mean_fnr) %>%
  tidyr::pivot_wider(names_from = method, values_from = mean_fnr) %>%
  dplyr::arrange(p, n_train, rho)

fpr_table <- summary_results %>%
  dplyr::select(p, n_train, rho, method, mean_fpr) %>%
  tidyr::pivot_wider(names_from = method, values_from = mean_fpr) %>%
  dplyr::arrange(p, n_train, rho)

mae_table <- summary_results %>%
  dplyr::mutate(mae_sd = sprintf("%.3f (%.3f)", mean_mae, sd_mae)) %>%
  dplyr::select(p, n_train, rho, method, mae_sd) %>%
  tidyr::pivot_wider(names_from = method, values_from = mae_sd) %>%
  dplyr::arrange(p, n_train, rho)


# ============================================================
# 13) Boxplots
# ============================================================

plot_dir <- file.path(out_dir, "boxplots")
if (!dir.exists(plot_dir)) {
  dir.create(plot_dir, recursive = TRUE)
}

method_order <- c(
  "AIC-ACLR", "BIC-ACLR",
  "AIC-CLR", "BIC-CLR",
  "QL-0.25", "QL-0.5", "QL-0.75",
  "CV-LASSO"
)

raw_results <- raw_results %>%
  dplyr::mutate(
    method = factor(method, levels = method_order),
    scenario = paste0("p=", p, ", n=", n_train, ", rho=", rho)
  )

save_boxplot <- function(data, y_var, y_label, file_stub, facet_formula = NULL,
                         width = 12, height = 7) {
  p_obj <- ggplot2::ggplot(data, ggplot2::aes(x = method, y = .data[[y_var]])) +
    ggplot2::geom_boxplot(outlier.size = 0.7, na.rm = TRUE) +
    ggplot2::labs(x = "Method", y = y_label) +
    ggplot2::theme_bw(base_size = 12) +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
      panel.grid.minor = ggplot2::element_blank()
    )

  if (!is.null(facet_formula)) {
    p_obj <- p_obj + ggplot2::facet_grid(facet_formula, scales = "free_y")
  }

  ggplot2::ggsave(
    filename = file.path(plot_dir, paste0(file_stub, ".png")),
    plot = p_obj,
    width = width,
    height = height,
    dpi = 300
  )

  ggplot2::ggsave(
    filename = file.path(plot_dir, paste0(file_stub, ".pdf")),
    plot = p_obj,
    width = width,
    height = height
  )
}

# Overall boxplots across all simulation scenarios.
save_boxplot(raw_results, "fnr", "False negative rate (%)", "boxplot_fnr_overall")
save_boxplot(raw_results, "fpr", "False positive rate (%)", "boxplot_fpr_overall")
save_boxplot(raw_results, "mae_test", "Test MAE", "boxplot_mae_overall")
save_boxplot(raw_results, "selected_count", "Number of selected coefficients", "boxplot_selected_count_overall")
save_boxplot(raw_results, "time_sec", "Computation time (seconds)", "boxplot_time_overall")

# Faceted boxplots by simulation setting.
save_boxplot(raw_results, "fnr", "False negative rate (%)", "boxplot_fnr_by_scenario", rho ~ p + n_train, width = 15, height = 9)
save_boxplot(raw_results, "fpr", "False positive rate (%)", "boxplot_fpr_by_scenario", rho ~ p + n_train, width = 15, height = 9)
save_boxplot(raw_results, "mae_test", "Test MAE", "boxplot_mae_by_scenario", rho ~ p + n_train, width = 15, height = 9)
save_boxplot(raw_results, "selected_count", "Number of selected coefficients", "boxplot_selected_count_by_scenario", rho ~ p + n_train, width = 15, height = 9)
save_boxplot(raw_results, "time_sec", "Computation time (seconds)", "boxplot_time_by_scenario", rho ~ p + n_train, width = 15, height = 9)

# Scenario-specific boxplots saved separately for easier inspection.
for (p_now in p_values) {
  for (n_train_now in n_train_values) {
    for (rho_now in rho_values) {
      dat_now <- raw_results %>%
        dplyr::filter(p == p_now, n_train == n_train_now, rho == rho_now)

      scenario_stub <- paste0("p", p_now, "_n", n_train_now, "_rho", rho_now)

      save_boxplot(dat_now, "fnr", "False negative rate (%)",
                   paste0("boxplot_fnr_", scenario_stub), width = 10, height = 6)
      save_boxplot(dat_now, "fpr", "False positive rate (%)",
                   paste0("boxplot_fpr_", scenario_stub), width = 10, height = 6)
      save_boxplot(dat_now, "mae_test", "Test MAE",
                   paste0("boxplot_mae_", scenario_stub), width = 10, height = 6)
    }
  }
}

# Save a plot index so it is clear what was generated.
plot_index <- data.frame(
  file = list.files(plot_dir, full.names = FALSE),
  path = file.path(plot_dir, list.files(plot_dir, full.names = FALSE))
)
write.csv(plot_index, file.path(plot_dir, "boxplot_file_index.csv"), row.names = FALSE)

# ============================================================
 # 14) Save final outputs
# ============================================================

write.csv(raw_results, file.path(out_dir, "highdim_raw_results.csv"), row.names = FALSE)
write.csv(summary_results, file.path(out_dir, "highdim_summary_results.csv"), row.names = FALSE)
write.csv(fnr_table, file.path(out_dir, "highdim_fnr_table.csv"), row.names = FALSE)
write.csv(fpr_table, file.path(out_dir, "highdim_fpr_table.csv"), row.names = FALSE)
write.csv(mae_table, file.path(out_dir, "highdim_mae_table.csv"), row.names = FALSE)

cat("\nDone. Files saved in:", out_dir, "\n")
cat("  highdim_aclr_tuning_grid_results.csv\n")
cat("  highdim_aclr_best_grid.csv\n")
cat("  highdim_raw_results.csv\n")
cat("  highdim_summary_results.csv\n")
cat("  highdim_fnr_table.csv\n")
cat("  highdim_fpr_table.csv\n")
cat("  highdim_mae_table.csv\n")
