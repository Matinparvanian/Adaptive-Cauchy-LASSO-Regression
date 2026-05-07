library(MASS)
library(glmnet)
library(ggplot2)
library(dplyr)
library(tidyr)
library(faraway)
library(stabledist)

set.seed(123)

############################################################
# USER PATHS
############################################################

forestfires_path <- "forestfires.csv"
out_dir <- "ACLR-Huber_real_dataset"

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

############################################################
# SETTINGS
############################################################

train_prop <- 0.75
split_seed <- 123
coef_tol <- 1e-4
eps_num <- 1e-10
lambda_grid_aclr <- exp(seq(log(0.02), log(2), length.out = 20))
adaptive_gamma <- 1
adaptive_eps <- 1e-3

# Adaptive Ridge-Huber initialization settings
huber_k <- 1.345
huber_lambda_grid <- exp(seq(log(0.001), log(10), length.out = 25))
huber_cv_folds <- 5

aclr_max_outer <- 25
aclr_tol <- 1e-4
stable_maxit <- 30

############################################################
# DATA
############################################################

prepare_forestfires_data <- function(path) {
  df <- read.csv(path, stringsAsFactors = FALSE)
  df$LBAF <- log(df$area + 1)
  keep_vars <- c("LBAF", "X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain")
  df <- df[, keep_vars]
  list(data = df, response_name = "LBAF", dataset_name = "Forest Fires")
}

prepare_prostate_data <- function() {
  data(prostate, package = "faraway")
  df <- prostate
  df$ILPSA <- 1 / df$lpsa
  keep_vars <- c("ILPSA", "lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45")
  df <- df[, keep_vars]
  list(data = df, response_name = "ILPSA", dataset_name = "Prostate Cancer")
}

make_train_test_split <- function(df, response_name, train_prop = 0.75, seed = 123) {
  set.seed(seed)
  n <- nrow(df)
  train_idx <- sample(seq_len(n), size = floor(train_prop * n), replace = FALSE)
  test_idx  <- setdiff(seq_len(n), train_idx)
  
  train_df <- df[train_idx, , drop = FALSE]
  test_df  <- df[test_idx, , drop = FALSE]
  
  y_train_raw <- train_df[[response_name]]
  y_test_raw  <- test_df[[response_name]]
  
  X_train_raw <- as.matrix(train_df[, setdiff(names(train_df), response_name), drop = FALSE])
  X_test_raw  <- as.matrix(test_df[, setdiff(names(test_df), response_name), drop = FALSE])
  
  x_means <- colMeans(X_train_raw)
  y_mean  <- mean(y_train_raw)
  
  X_train <- scale(X_train_raw, center = x_means, scale = FALSE)
  X_test  <- scale(X_test_raw,  center = x_means, scale = FALSE)
  
  y_train <- as.numeric(y_train_raw - y_mean)
  y_test  <- as.numeric(y_test_raw  - y_mean)
  
  list(
    X_train = X_train,
    X_test = X_test,
    y_train = y_train,
    y_test = y_test,
    y_train_raw = y_train_raw,
    y_test_raw = y_test_raw
  )
}

############################################################
# ACLR with Adaptive Ridge-Huber initialization
#
# beta_init = argmin_beta { sum_i rho_H(y_i - x_i^T beta)
#                           + lambda_init sum_j beta_j^2 }
# w_j = 1 / (|beta_init_j| + adaptive_eps)^adaptive_gamma
# ACLR objective: sum_i log(sigma^2 + residual_i^2)
#                 + lambda sum_j w_j |beta_j|
############################################################

profile_sigma_ml <- function(residuals) {
  residuals <- as.numeric(residuals)
  obj_sigma <- function(sigma) sum(log(pmax(sigma^2 + residuals^2, eps_num)))
  upper <- max(5, sd(residuals), mad(residuals), sqrt(mean(residuals^2)), 1)
  fit <- optimize(obj_sigma, interval = c(1e-6, upper))
  list(sigma = fit$minimum, value = fit$objective)
}

robust_sigma_update <- function(residuals, min_sigma = 1e-4) {
  s <- mad(residuals, center = 0, constant = 1.4826, na.rm = TRUE)
  if (!is.finite(s) || is.na(s) || s < min_sigma) s <- sqrt(mean(residuals^2))
  max(s, min_sigma)
}

huber_rho <- function(r, k = huber_k) {
  r_abs <- abs(r)
  ifelse(r_abs <= k, 0.5 * r^2, k * r_abs - 0.5 * k^2)
}

fit_huber_ridge_given_lambda <- function(X, y, lambda_init, k = huber_k) {
  p <- ncol(X)

  # Stable starting point from ordinary ridge.
  beta_start <- tryCatch({
    fit0 <- glmnet(
      x = X, y = y,
      alpha = 0,
      lambda = lambda_init,
      intercept = FALSE,
      standardize = FALSE
    )
    as.numeric(coef(fit0))[-1]
  }, error = function(e) rep(0, p))

  obj <- function(beta) {
    r <- as.numeric(y - X %*% beta)
    sum(huber_rho(r, k = k)) + lambda_init * sum(beta^2)
  }

  fit <- tryCatch(
    optim(
      par = beta_start,
      fn = obj,
      method = "BFGS",
      control = list(maxit = 500, reltol = 1e-8)
    ),
    error = function(e) NULL
  )

  beta_hat <- if (is.null(fit) || !all(is.finite(fit$par))) beta_start else fit$par
  beta_hat[!is.finite(beta_hat)] <- 0
  beta_hat
}

select_huber_ridge_lambda_cv <- function(X, y, lambda_grid = huber_lambda_grid,
                                         k = huber_k, nfolds = huber_cv_folds,
                                         seed = split_seed) {
  n <- nrow(X)
  set.seed(seed)
  fold_id <- sample(rep(seq_len(nfolds), length.out = n))

  cv_errors <- numeric(length(lambda_grid))

  for (l in seq_along(lambda_grid)) {
    lam <- lambda_grid[l]
    fold_errors <- numeric(nfolds)

    for (fold in seq_len(nfolds)) {
      train_idx <- which(fold_id != fold)
      valid_idx <- which(fold_id == fold)

      beta_fold <- fit_huber_ridge_given_lambda(
        X = X[train_idx, , drop = FALSE],
        y = y[train_idx],
        lambda_init = lam,
        k = k
      )

      pred_valid <- as.numeric(X[valid_idx, , drop = FALSE] %*% beta_fold)
      fold_errors[fold] <- mean(abs(y[valid_idx] - pred_valid))
    }

    cv_errors[l] <- mean(fold_errors, na.rm = TRUE)
  }

  lambda_grid[which.min(cv_errors)]
}

get_initial_beta_for_aclr <- function(X, y) {
  lambda_init <- select_huber_ridge_lambda_cv(
    X = X,
    y = y,
    lambda_grid = huber_lambda_grid,
    k = huber_k,
    nfolds = huber_cv_folds,
    seed = split_seed
  )

  beta_init <- fit_huber_ridge_given_lambda(
    X = X,
    y = y,
    lambda_init = lambda_init,
    k = huber_k
  )

  attr(beta_init, "lambda_init") <- lambda_init
  beta_init
}

make_adaptive_weights <- function(beta_init, gamma = 1, eps = 1e-3) {
  1 / (abs(beta_init) + eps)^gamma
}

aclr_objective <- function(beta, X, y, lambda, sigma, weights) {
  r <- as.numeric(y - X %*% beta)
  sum(log(pmax(sigma^2 + r^2, eps_num))) + lambda * sum(weights * abs(beta))
}

fit_aclr_fast_fixed_lambda <- function(X, y, lambda, weights,
                                       sigma_init = NULL,
                                       beta_init = NULL,
                                       max_outer = aclr_max_outer,
                                       tol = aclr_tol) {
  p <- ncol(X)
  beta <- if (is.null(beta_init)) rep(0, p) else beta_init
  sigma <- if (is.null(sigma_init)) robust_sigma_update(y) else sigma_init
  old_obj <- aclr_objective(beta, X, y, lambda, sigma, weights)
  
  for (iter in 1:max_outer) {
    beta_prev <- beta
    sigma_prev <- sigma
    
    r_prev <- as.numeric(y - X %*% beta_prev)
    obs_w <- 1 / pmax(sigma_prev^2 + r_prev^2, eps_num)
    
    sqrt_w <- sqrt(obs_w)
    X_w <- X * sqrt_w
    y_w <- y * sqrt_w
    
    fit_wlasso <- glmnet(
      x = X_w, y = y_w,
      alpha = 1,
      lambda = lambda,
      intercept = FALSE,
      standardize = FALSE,
      penalty.factor = weights
    )
    
    beta <- as.numeric(coef(fit_wlasso))[-1]
    beta[abs(beta) <= coef_tol] <- 0
    
    r_new <- as.numeric(y - X %*% beta)
    sigma <- profile_sigma_ml(r_new)$sigma
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
  n <- length(y)
  df_lambda <- sum(abs(beta_hat) > coef_tol)
  fit_term <- sum(log(pmax(sigma_hat^2 + e^2, eps_num)))
  aic_value <- (fit_term + 2 * df_lambda) / n
  bic_value <- (fit_term + log(n) * df_lambda) / n
  list(AIC = aic_value, BIC = bic_value)
}

fit_aclr_path <- function(X, y, lambda_grid, gamma = 1, eps = 1e-3) {
  beta_init0 <- get_initial_beta_for_aclr(X, y)
  weights <- make_adaptive_weights(beta_init0, gamma = gamma, eps = eps)
  
  fits <- vector("list", length(lambda_grid))
  beta_start <- beta_init0
  sigma_start <- robust_sigma_update(y)
  
  for (k in seq_along(lambda_grid)) {
    lam <- lambda_grid[k]
    
    fit_k <- fit_aclr_fast_fixed_lambda(
      X = X, y = y, lambda = lam,
      weights = weights,
      sigma_init = sigma_start,
      beta_init = beta_start
    )
    
    ic_k <- compute_ic_aclr(X, y, fit_k$beta, fit_k$sigma)
    
    fits[[k]] <- list(
      lambda = lam,
      beta = fit_k$beta,
      sigma = fit_k$sigma,
      AIC = ic_k$AIC,
      BIC = ic_k$BIC
    )
    
    beta_start <- fit_k$beta
    sigma_start <- fit_k$sigma
  }
  
  list(
    fits = fits,
    beta_init = beta_init0,
    lambda_init_huber = attr(beta_init0, "lambda_init"),
    weights = weights
  )
}

select_best_fit <- function(path_obj, criterion = c("AIC", "BIC")) {
  criterion <- match.arg(criterion)
  crit_values <- sapply(path_obj$fits, function(z) z[[criterion]])
  path_obj$fits[[which.min(crit_values)]]
}

############################################################
# STABLE MLE
############################################################

fit_stable_mle_multistart <- function(x, maxit = stable_maxit) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  
  if (length(x) < 10) {
    return(data.frame(alpha = NA, beta = NA, gamma = NA, delta = NA))
  }
  
  sx <- sd(x)
  if (!is.finite(sx) || sx <= 0) sx <- mad(x)
  if (!is.finite(sx) || sx <= 0) sx <- 1
  
  negloglik <- function(par) {
    alpha <- par[1]
    beta  <- par[2]
    gamma <- exp(par[3])
    delta <- par[4]
    
    dens <- tryCatch(
      stabledist::dstable(x, alpha = alpha, beta = beta, gamma = gamma, delta = delta, pm = 0),
      error = function(e) rep(NA_real_, length(x))
    )
    
    if (any(!is.finite(dens)) || any(dens <= 0)) return(1e12)
    -sum(log(pmax(dens, 1e-300)))
  }
  
  # Focused multistart grid: same concept, less duplicated work
  starts <- unique(rbind(
    expand.grid(
      alpha = c(0.7, 1.0, 1.5),
      beta  = c(-0.5, 0, 0.5),
      gamma = c(log(max(0.05, sx / 2)), log(max(0.05, sx))),
      delta = c(median(x)),
      KEEP.OUT.ATTRS = FALSE,
      stringsAsFactors = FALSE
    ),
    data.frame(
      alpha = c(0.45, 1.8),
      beta = c(0.9, -0.9),
      gamma = c(log(max(0.05, 2 * sx)), log(max(0.05, sx))),
      delta = c(mean(x), mean(x))
    )
  ))
  
  lower <- c(0.2, -0.99, log(1e-4), min(x) - 5 * sx)
  upper <- c(1.99,  0.99, log(max(10 * sx, 10)), max(x) + 5 * sx)
  
  best_val <- Inf
  best_par <- rep(NA_real_, 4)
  
  for (i in seq_len(nrow(starts))) {
    init <- c(starts$alpha[i], starts$beta[i], starts$gamma[i], starts$delta[i])
    
    fit <- tryCatch(
      optim(
        par = init,
        fn = negloglik,
        method = "L-BFGS-B",
        lower = lower,
        upper = upper,
        control = list(maxit = maxit)
      ),
      error = function(e) NULL
    )
    
    if (!is.null(fit) && is.finite(fit$value) && fit$value < best_val) {
      best_val <- fit$value
      best_par <- fit$par
    }
  }
  
  if (!all(is.finite(best_par))) {
    return(data.frame(alpha = NA, beta = NA, gamma = NA, delta = NA))
  }
  
  data.frame(
    alpha = round(best_par[1], 4),
    beta  = round(best_par[2], 4),
    gamma = round(exp(best_par[3]), 4),
    delta = round(best_par[4], 4)
  )
}

############################################################
# RUN ONE DATASET
############################################################

analyze_dataset_aclr <- function(prepared_obj, seed_split = 123) {
  df <- prepared_obj$data
  response_name <- prepared_obj$response_name
  dataset_name <- prepared_obj$dataset_name
  
  split_obj <- make_train_test_split(df, response_name, train_prop = train_prop, seed = seed_split)
  X_train <- split_obj$X_train
  y_train <- split_obj$y_train
  
  aclr_path <- fit_aclr_path(
    X = X_train,
    y = y_train,
    lambda_grid = lambda_grid_aclr,
    gamma = adaptive_gamma,
    eps = adaptive_eps
  )
  
  fit_aic_aclr <- select_best_fit(aclr_path, "AIC")
  fit_bic_aclr <- select_best_fit(aclr_path, "BIC")

  predictor_names <- colnames(X_train)
  coefficient_summary <- bind_rows(
    data.frame(
      dataset = dataset_name,
      method = "Huber-ridge initial",
      variable = predictor_names,
      coefficient = as.numeric(aclr_path$beta_init),
      selected = abs(aclr_path$beta_init) > coef_tol,
      lambda = aclr_path$lambda_init_huber,
      sigma = NA_real_,
      AIC = NA_real_,
      BIC = NA_real_
    ),
    data.frame(
      dataset = dataset_name,
      method = "AIC-ACLR",
      variable = predictor_names,
      coefficient = as.numeric(fit_aic_aclr$beta),
      selected = abs(fit_aic_aclr$beta) > coef_tol,
      lambda = fit_aic_aclr$lambda,
      sigma = fit_aic_aclr$sigma,
      AIC = fit_aic_aclr$AIC,
      BIC = fit_aic_aclr$BIC
    ),
    data.frame(
      dataset = dataset_name,
      method = "BIC-ACLR",
      variable = predictor_names,
      coefficient = as.numeric(fit_bic_aclr$beta),
      selected = abs(fit_bic_aclr$beta) > coef_tol,
      lambda = fit_bic_aclr$lambda,
      sigma = fit_bic_aclr$sigma,
      AIC = fit_bic_aclr$AIC,
      BIC = fit_bic_aclr$BIC
    )
  )
  
  resid_aic <- as.numeric(y_train - X_train %*% fit_aic_aclr$beta)
  resid_bic <- as.numeric(y_train - X_train %*% fit_bic_aclr$beta)
  
  response_density_df <- data.frame(
    value = df[[response_name]],
    type = "Response",
    dataset = dataset_name
  )
  
  resid_density_df <- bind_rows(
    data.frame(value = resid_aic, type = "ACLR Residuals (AIC)", dataset = dataset_name),
    data.frame(value = resid_bic, type = "ACLR Residuals (BIC)", dataset = dataset_name)
  )
  
  response_stable <- cbind(
    dataset = dataset_name,
    variable = response_name,
    fit_stable_mle_multistart(df[[response_name]])
  )
  
  residual_stable <- bind_rows(
    cbind(dataset = dataset_name, method = "AIC-ACLR", fit_stable_mle_multistart(resid_aic)),
    cbind(dataset = dataset_name, method = "BIC-ACLR", fit_stable_mle_multistart(resid_bic))
  )
  
  lambda_summary <- data.frame(
    dataset = dataset_name,
    lambda_init_Huber = aclr_path$lambda_init_huber,
    huber_k = huber_k,
    adaptive_gamma = adaptive_gamma,
    adaptive_eps = adaptive_eps,
    lambda_AIC = fit_aic_aclr$lambda,
    sigma_AIC = fit_aic_aclr$sigma,
    selected_AIC = sum(abs(fit_aic_aclr$beta) > coef_tol),
    AIC_value = fit_aic_aclr$AIC,
    lambda_BIC = fit_bic_aclr$lambda,
    sigma_BIC = fit_bic_aclr$sigma,
    selected_BIC = sum(abs(fit_bic_aclr$beta) > coef_tol),
    BIC_value = fit_bic_aclr$BIC
  )
  
  list(
    response_density_df = response_density_df,
    resid_density_df = resid_density_df,
    response_stable = response_stable,
    residual_stable = residual_stable,
    lambda_summary = lambda_summary,
    coefficient_summary = coefficient_summary
  )
}

############################################################
# EXECUTE
############################################################

forest_obj <- prepare_forestfires_data(forestfires_path)
prostate_obj <- prepare_prostate_data()

forest_res <- analyze_dataset_aclr(forest_obj, split_seed)
prostate_res <- analyze_dataset_aclr(prostate_obj, split_seed)

density_response_all <- bind_rows(forest_res$response_density_df, prostate_res$response_density_df)
density_resid_all <- bind_rows(forest_res$resid_density_df, prostate_res$resid_density_df)
response_stable_all <- bind_rows(forest_res$response_stable, prostate_res$response_stable)
residual_stable_all <- bind_rows(forest_res$residual_stable, prostate_res$residual_stable)
lambda_summary_all <- bind_rows(forest_res$lambda_summary, prostate_res$lambda_summary)
coefficient_summary_all <- bind_rows(forest_res$coefficient_summary, prostate_res$coefficient_summary)

############################################################
# PLOTS
############################################################

plot_response_density <- ggplot(density_response_all, aes(x = value)) +
  geom_density(color = "black", fill = "grey80") +
  facet_wrap(~ dataset, scales = "free") +
  labs(title = "Density curves of response variables", x = "Response value", y = "Density") +
  theme_bw(base_size = 12)
print(plot_response_density)

plot_residual_density <- ggplot(density_resid_all, aes(x = value, linetype = type)) +
  geom_density(color = "black", linewidth = 0.8) +
  facet_wrap(~ dataset, scales = "free") +
  labs(title = "Density curves of ACLR residuals", x = "Residual value", y = "Density", linetype = "") +
  theme_bw(base_size = 12)
print(plot_residual_density)

############################################################
# SAVE
############################################################

write.csv(response_stable_all, file.path(out_dir, "response_stable_ACLR.csv"), row.names = FALSE)
write.csv(residual_stable_all, file.path(out_dir, "residual_stable_ACLR.csv"), row.names = FALSE)
write.csv(lambda_summary_all, file.path(out_dir, "lambda_summary_ACLR.csv"), row.names = FALSE)
write.csv(coefficient_summary_all, file.path(out_dir, "coefficient_summary_ACLR_Huber.csv"), row.names = FALSE)

ggsave(file.path(out_dir, "response_density_ACLR.png"), plot_response_density, width = 10, height = 5, dpi = 300)
ggsave(file.path(out_dir, "residual_density_ACLR.png"), plot_residual_density, width = 10, height = 5, dpi = 300)

cat("\n===== Lambda summary for ACLR with Adaptive Ridge-Huber initialization =====\n")
print(lambda_summary_all)

cat("\n===== Coefficient summary for Huber initial, AIC-ACLR, and BIC-ACLR =====\n")
print(coefficient_summary_all)

cat("\n===== Stable fit for response variables =====\n")
print(response_stable_all)

cat("\n===== Stable fit for ACLR residuals =====\n")
print(residual_stable_all)




############################################################
# ADD-ON: Compute MAE for selected ACLR models
# This does NOT rerun the whole training path.
# It only refits the final selected lambda from lambda_summary_all.
############################################################

compute_mae_only_final <- function(prepared_obj, lambda_value, method_name, seed_split = 123) {
  df <- prepared_obj$data
  response_name <- prepared_obj$response_name
  dataset_name <- prepared_obj$dataset_name
  
  # rebuild the same train/test split
  split_obj <- make_train_test_split(
    df = df,
    response_name = response_name,
    train_prop = train_prop,
    seed = seed_split
  )
  
  X_train <- split_obj$X_train
  y_train <- split_obj$y_train
  X_test  <- split_obj$X_test
  y_test  <- split_obj$y_test
  
  # rebuild the same adaptive weights
  beta_init0 <- get_initial_beta_for_aclr(X_train, y_train)
  weights <- make_adaptive_weights(beta_init0, gamma = adaptive_gamma, eps = adaptive_eps)
  
  # refit ONLY the final selected lambda
  fit_final <- fit_aclr_fast_fixed_lambda(
    X = X_train,
    y = y_train,
    lambda = lambda_value,
    weights = weights,
    sigma_init = robust_sigma_update(y_train),
    beta_init = beta_init0
  )
  
  # test prediction
  y_pred_test <- as.numeric(X_test %*% fit_final$beta)
  
  # MAE on centered scale
  mae_centered <- mean(abs(y_test - y_pred_test))
  
  # MAE on original response scale
  y_mean <- mean(split_obj$y_train_raw)
  y_pred_test_raw <- y_pred_test + y_mean
  mae_raw <- mean(abs(split_obj$y_test_raw - y_pred_test_raw))
  
  data.frame(
    dataset = dataset_name,
    method = method_name,
    p_selected = sum(abs(fit_final$beta) > coef_tol),
    lambda = lambda_value,
    sigma = fit_final$sigma,
    MAE_centered = mae_centered,
    MAE_raw = mae_raw
  )
}

# Forest Fires
forest_lambda_aic <- lambda_summary_all$lambda_AIC[lambda_summary_all$dataset == "Forest Fires"]
forest_lambda_bic <- lambda_summary_all$lambda_BIC[lambda_summary_all$dataset == "Forest Fires"]

# Prostate Cancer
prostate_lambda_aic <- lambda_summary_all$lambda_AIC[lambda_summary_all$dataset == "Prostate Cancer"]
prostate_lambda_bic <- lambda_summary_all$lambda_BIC[lambda_summary_all$dataset == "Prostate Cancer"]

# compute MAE
mae_forest_aic <- compute_mae_only_final(forest_obj, forest_lambda_aic, "AIC-ACLR", split_seed)
mae_forest_bic <- compute_mae_only_final(forest_obj, forest_lambda_bic, "BIC-ACLR", split_seed)
mae_prostate_aic <- compute_mae_only_final(prostate_obj, prostate_lambda_aic, "AIC-ACLR", split_seed)
mae_prostate_bic <- compute_mae_only_final(prostate_obj, prostate_lambda_bic, "BIC-ACLR", split_seed)

mae_results_all <- dplyr::bind_rows(
  mae_forest_aic,
  mae_forest_bic,
  mae_prostate_aic,
  mae_prostate_bic
)

cat("\n===== MAE results for selected ACLR models =====\n")
print(mae_results_all)

write.csv(
  mae_results_all,
  file.path(out_dir, "MAE_selected_ACLR.csv"),
  row.names = FALSE
)
