#This study introduces a regime-switching approach for capturing and predicting the behavior of 
#financial markets, employing a Hidden Markov Model (HMM) with skew-normal emission distributions 
#applied to daily returns of the S&P500 index from 2016 to 2025. Unlike traditional Gaussian-based 
#models, this formulation allows for asymmetric returns and volatility clustering, thus reflecting 
#important non-linear characteristics commonly observed in equity markets.

#Using model selection criteria such as Akaike Information Criterion (AIC) and Bayesian 
#Information Criterion (BIC), a three-regime specification was selected as the optimal 
#trade-off between complexity and interpretability. Each latent state displays distinct 
#statistical properties.

#Building on this classification, we design 5 different regime-sensitive trading strategies, which adapts market exposure based on the prevailing latent state,
# each progressively integrating additional sources of information — from statistical regime detection to technical and macroeconomic indicators — in order to dynamically adjust market exposure and control risk.

#1. Conservative Strategy
#The first model adopts a strictly defensive stance.
#Using filtered regime probabilities from the HMM, it avoids exposure whenever the probability of being in State 2 (the most turbulent regime) exceeds 40%.
#Positions are taken only in States 1 and 3, which exhibit better risk‑adjusted characteristics.
#The goal is to preserve capital by staying invested solely in statistically favorable conditions.

#2. Long–Short–Leverage Strategy
#The second approach introduces directional and leveraged exposure.
#Based on the most probable regime:
  
#State 1: long with 2× leverage
#State 2: short
#State 3: long with 2× leverage
#This rule ties portfolio exposure directly to the market phase identified by the HMM, increasing or reversing positions according to the estimated state’s risk–return profile.

# 3. Active Management with RSI in State 2
#The third strategy enhances regime signals with the Relative Strength Index (RSI), a momentum indicator capturing overbought and oversold conditions.
#Fixed decisions are applied in the extreme regimes (long 2× in State 1, lon 2× in State 3), while State 2 — the speculative, high‑volatility phase — is managed actively:
  
#RSI<30→2× long (oversold)
#RSI > 70 → 2× short (overbought)
# Otherwise → light short (–1).
#This configuration exploits short‑term reversals within unstable markets.

#4. Regime‑Adaptive Strategy with Dynamic Leverage and RSI
#This model combines regime probabilities, RSI signals, and variable leverage to tailor exposure levels between –2 and +2.
#Confidence thresholds on the regime probability determine whether the strategy scales up or down its position:
  
# In stable states (1 and 3), exposure increases to 2× only when confidence>60%.
# In the volatile regime (2), RSI guides the direction (oversold→long, overbought→short).
#The objective is to magnify returns during highly reliable phases while limiting risk during uncertainty.


#5. HMM + RSI + Term‑Spread Strategy
#The fifth and most comprehensive algorithm introduces a macroeconomic dimension.
#It integrates the HMM, RSI, and the Term Spread (10‑year minus 2‑year U.S.Treasury yields), a key measure of yield‑curve slope and recession risk.

# When the spread is negative, exposure is reduced or reversed to reflect macro stress.
# When positive, the strategy follows regime‑specific and RSI‑driven rules, taking stronger positions in supportive environments.
#This hybrid design aligns short‑term technical signals with longer‑term economic conditions, creating a regime‑ and cycle‑aware decision framework.



# Load required libraries
#install.packages("patchwork")
library(tseries)
library(xts)
library(sn) 
library(dplyr)
library(e1071) 
library(ggplot2)
library(ggthemes)
library(PerformanceAnalytics)
library(reshape2)
library(patchwork)
library(scales) 

# Download STOXX600 adjusted close prices from 2010 to early 2025
sx600 <- get.hist.quote(instrument = "^GSPC",
                        start = "2016-01-01", 
                        end = "2025-05-01", 
                        quote = "AdjClose")
## time series starts 2010-01-04
## time series ends   2025-04-30
# Compute daily log-returns
sx600.num <- as.numeric(sx600)
sx600.ret <- diff(log(sx600.num))
dates <- index(sx600)[-1]
sx600.ret <- xts(sx600.ret, order.by = dates)
colnames(sx600.ret) <- "log_returns"


# Plot STOXX600 daily log-returns (2010–2025)
plot(sx600.ret, 
     main = "STOXX600 Daily Log-Returns (2010–2025)", 
     col = "steelblue", 
     ylab = "Log-Return", 
     xlab = "Date", 
     major.ticks = "years", 
     grid.ticks.on = "years")

# Plot STOXX600 adjusted closing prices (2010–2025)
plot(sx600, 
     main = "STOXX600 Adjusted Closing Prices (2010–2025)", 
     col = "darkblue", 
     ylab = "Price", 
     xlab = "Date", 
     major.ticks = "years", 
     grid.ticks.on = "years")



# Extract full returns vector
returns <- na.omit(as.numeric(sx600.ret$log_returns))

# Plot histogram with density overlays
hist(returns,
     breaks = 50,
     freq = FALSE,
     col = "lightblue",
     main = "Histogram of STOXX600 Log-Returns (2010–2025)",
     xlab = "Log-Return")

# Kernel density
lines(density(returns), col = "blue", lwd = 2)

# Fit skew-normal distribution
fit_sn <- selm(returns ~ 1)
params <- coef(fit_sn, "DP")  # Extract xi, omega, alpha

# Overlay skew-normal density
curve(dsn(x, xi = params["xi"], omega = params["omega"], alpha = params["alpha"]),
      add = TRUE, col = "red", lwd = 2)

# Overlay normal density
curve(dnorm(x, mean = mean(returns), sd = sd(returns)),
      add = TRUE, col = "darkgreen", lwd = 2, lty = 2)

# Add legend
legend("topright",
       legend = c("Kernel Density", "Skew-Normal Fit", "Normal Fit"),
       col = c("blue", "red", "darkgreen"),
       lwd = 2, lty = c(1, 1, 2))

# === Compute and compare skewness measures ===

# Extract skewness parameter (alpha)
alpha <- params["alpha"]

# Compute implied skewness from skew-normal fit
delta <- alpha / sqrt(1 + alpha^2)
skew_sn <- ((4 - pi) / 2) * ((delta * sqrt(2 / pi))^3) / ((1 - (2 * delta^2 / pi))^(3/2))

# Compute empirical skewness
empirical_skew <- skewness(returns)

# Display comparison
cat("Empirical skewness from data:       ", round(empirical_skew, 5), "\n")
## Empirical skewness from data:        -0.79483
cat("Skewness implied by skew-normal fit:", round(skew_sn, 5), "\n")
## Skewness implied by skew-normal fit: -0.22654
cat("Skewness parameter alpha:           ", round(alpha, 5), "\n")
## Skewness parameter alpha:            -1.27892



# Create train/test split
train_end <- as.Date("2022-12-31")
test_start <- as.Date("2023-01-01")

dates_full <- index(sx600.ret)
returns_full <- na.omit(as.numeric(sx600.ret$log_returns))

# Find date indices
train_idx <- which(dates_full <= train_end)
test_idx  <- which(dates_full >= test_start)

# Split returns
returns_train <- returns_full[train_idx]
returns_test  <- returns_full[test_idx]

# Also split dates (for plotting or xts conversion later)
dates_train <- dates_full[train_idx]
dates_test  <- dates_full[test_idx]


# Forward algorithm with skew-normal emission distributions
forward_skew <- function(delta, gamma, xi, omega, alpha, data) {
  m <- length(delta)
  T <- length(data)
  
  pred <- matrix(NA, nrow = m, ncol = T)
  filtering <- matrix(NA, nrow = m, ncol = T)
  
  pred[, 1] <- delta
  filtering[, 1] <- pred[, 1] * dsn(x = data[1], xi = xi, omega = omega, alpha = alpha)
  loglik <- log(sum(filtering[, 1]))
  filtering[, 1] <- filtering[, 1] / sum(filtering[, 1])
  
  for (t in 2:T) {
    pred[, t] <- as.vector(t(filtering[, t - 1]) %*% gamma)
    filtering[, t] <- pred[, t] * dsn(x = data[t], xi = xi, omega = omega, alpha = alpha)
    loglik <- loglik + log(sum(filtering[, t]))
    filtering[, t] <- filtering[, t] / sum(filtering[, t])
  }
  
  return(list(loglik = loglik, filtering = filtering, pred = pred))
}


loglik_skewHMM <- function(par, m, data) {
  # Transition matrix
  gamma <- matrix(par[1:(m^2)], nrow = m, byrow = TRUE)
  gamma <- gamma / rowSums(gamma)  # normalize rows
  
  # Skew-normal parameters
  start <- m^2 + 1
  xi <- par[start:(start + m - 1)]
  omega <- par[(start + m):(start + 2 * m - 1)]
  alpha <- par[(start + 2 * m):(start + 3 * m - 1)]
  
  # Stationary distribution
  I <- diag(m)
  U <- matrix(1, m, m)
  delta <- solve(t(I - gamma + U), rep(1, m))
  
  out <- forward_skew(delta, gamma, xi, omega, alpha, data)
  return(out$loglik)
}



#Selection of the number of states

#We study models with 1, 2, 3, and 4 states.

#For each model, we calculate the AIC and BIC information criteria to select the optimal number of states.

#We save all estimated parameters for the optimal model ( m = 4).



# Set training data
data <- returns_train
T <- length(data)

fit_models <- list()
aic_bic_results <- data.frame()

# === Case m = 1: single skew-normal (no HMM) ===
cat("Fitting model with 1 state (no regime switching)...\n")
## Fitting model with 1 state (no regime switching)...
loglik_m1 <- function(par, data) {
  xi <- par[1]
  omega <- par[2]
  alpha <- par[3]
  -sum(dsn(data, xi = xi, omega = omega, alpha = alpha, log = TRUE))  # negative log-likelihood
}

init_par1 <- c(mean(data), sd(data), 0)
lower_bounds1 <- c(-Inf, 1e-4, -Inf)
upper_bounds1 <- c(Inf, Inf, Inf)

fit1 <- optim(
  par = init_par1,
  fn = loglik_m1,
  data = data,
  method = "L-BFGS-B",
  lower = lower_bounds1,
  upper = upper_bounds1
)

loglik1 <- -fit1$value
num_params1 <- 3
aic1 <- 2 * num_params1 - 2 * loglik1
bic1 <- log(T) * num_params1 - 2 * loglik1

aic_bic_results <- rbind(
  aic_bic_results,
  data.frame(States = 1, LogLikelihood = loglik1, NumParameters = num_params1, AIC = aic1, BIC = bic1)
)

fit_models[["HMM_1_state"]] <- list(
  par = fit1$par,
  loglik = loglik1,
  AIC = aic1,
  BIC = bic1,
  m = 1
)



# === Fit HMMs for m = 2 to 4 on training set ===
for (m in 2:4) {
  cat("Fitting model with", m, "states...\n")
  set.seed(123)
  
  k <- m^2 + 3 * m
  
  gamma_init <- rep(1 / m, m^2)
  xi_init <- rnorm(m, mean = 0, sd = 0.01)
  omega_init <- runif(m, 0.005, 0.05)
  alpha_init <- rnorm(m, 0, 2)
  
  init_par <- c(gamma_init, xi_init, omega_init, alpha_init)
  
  lower_bounds <- c(rep(1e-6, m^2), rep(-Inf, m), rep(1e-4, m), rep(-Inf, m))
  upper_bounds <- c(rep(1, m^2), rep(Inf, m), rep(Inf, m), rep(Inf, m))
  
  fit <- optim(
    par = init_par,
    fn = function(par) -loglik_skewHMM(par, m, data),
    method = "L-BFGS-B",
    lower = lower_bounds,
    upper = upper_bounds,
    control = list(maxit = 1000)
  )
  
  loglik <- -fit$value
  num_params <- m^2 + 3 * m
  aic <- 2 * num_params - 2 * loglik
  bic <- log(T) * num_params - 2 * loglik
  
  aic_bic_results <- rbind(aic_bic_results, data.frame(
    States = m,
    LogLikelihood = loglik,
    NumParameters = num_params,
    AIC = aic,
    BIC = bic
  ))
  
  fit_models[[paste0("HMM_", m, "_states")]] <- list(
    par = fit$par,
    loglik = loglik,
    AIC = aic,
    BIC = bic,
    m = m
  )
}
## Fitting model with 2 states...
## Fitting model with 3 states...
## Fitting model with 4 states...
# View results


print(aic_bic_results)


# Extract the 4-state model fitted on training data
best_model <- fit_models[["HMM_3_states"]]
par <- best_model$par
m <- best_model$m  # Should be 4

# Unpack transition matrix and emission parameters
gamma_vec <- par[1:(m^2)]
gamma <- matrix(gamma_vec, nrow = m, byrow = TRUE)
gamma <- gamma / rowSums(gamma)  # Ensure rows sum to 1

start <- m^2 + 1
xi <- par[start:(start + m - 1)]
omega <- par[(start + m):(start + 2 * m - 1)]
alpha <- par[(start + 2 * m):(start + 3 * m - 1)]

# Compute stationary distribution (solves: δᵗΓ = δᵗ)
I <- diag(m)
U <- matrix(1, m, m)
delta <- solve(t(I - gamma + U), rep(1, m))

# Display parameters in a table
parameter_table <- data.frame(
  State = 1:m,
  Location = xi,
  Scale = omega,
  Skewness = alpha,
  Stationary_Prob = delta
)

print(parameter_table)

#State     Location       Scale   Skewness Stationary_Prob
#1     1  0.001644039 0.004040517 -0.2204732       0.4476389
#2     2  0.001235783 0.026142073 -0.2392483       0.1314588
#3     3 -0.001305207 0.010068147  0.3664549       0.4209022


#For each state identified in the training period:
  
#We plot the histogram of returns conditional on that state.

#We overlay the estimated skew-normal density to check its goodness of fit.




# Run forward algorithm once using the full data and fitted model
forward_results <- forward_skew(delta, gamma, xi, omega, alpha, returns_full)
filtering <- forward_results$filtering  
most_likely_states <- apply(filtering, 2, which.max)
dev.off()
# Plot histograms by state using only training data
par(mfrow = c(2, 2))  # 2x2 grid for 4 states

for (state in 1:4) {
  # Filter: only training period & only current state
  data_state <- returns_train[most_likely_states[train_idx] == state]
  print(paste("State", state, "has", length(data_state), "points"))
  if (length(data_state) == 0) {
    message(paste("No data for State", state))
    next
  }
  
  hist_info <- hist(data_state, probability = TRUE, breaks = 50, plot = FALSE)
  
  data_min <- min(data_state)
  data_max <- max(data_state)
  range_expansion <- (data_max - data_min) * 0.15
  
  x_min <- data_min - range_expansion
  x_max <- data_max + range_expansion
  x_vals <- seq(x_min, x_max, length.out = 500)
  y_vals <- dsn(x_vals, xi = xi[state], omega = omega[state], alpha = alpha[state])
  
  ylim_max <- max(max(hist_info$density), max(y_vals)) * 1.1
  
  hist(data_state, probability = TRUE, breaks = 50,
       main = paste("Histogram of Returns - State", state),
       xlab = "Log-Return", ylab = "Density",
       col = "lightgray", border = "white",
       xlim = c(x_min, x_max), ylim = c(0, ylim_max))
  
  curve(dsn(x, xi = xi[state], omega = omega[state], alpha = alpha[state]),
        from = x_min, to = x_max, add = TRUE, col = "red", lwd = 2)
}



# Visualization of State Probabilities and Regimes

#We plot the returns over the period 2010–2022, coloring each point according to the most likely latent state.

#This visualization allows us to interpret the evolution of regimes over time and visually connect them to known
#macroeconomic events or market behavior.


## Plot regimes over time (training set only)

# Step 1: Extract training states and dates
dates_train <- dates_full[train_idx]
states_train <- most_likely_states[train_idx]
returns_train_plot <- returns_train  # already aligned with dates_train

# Step 2: Build data frame for ggplot
plot_data_train <- data.frame(
  Date = dates_train,
  Return = returns_train_plot,
  State = factor(states_train, levels = 1:4)  # Only 4 states
)

# Step 3: Plot
ggplot(plot_data_train, aes(x = Date, y = Return, color = State)) +
  geom_point(size = 0.5, alpha = 0.8) +
  scale_color_brewer(palette = "Dark2") +
  labs(title = "Training Set: STOXX600 Daily Returns Colored by Hidden State (2010–2022)",
       y = "Log-Return",
       x = "Date") +
  theme_minimal()




#Given this fitted model, let's try to use it in some trading stategies 


#########################################################################################
#########################################################################################
#########################################################################################

#STRATEGY 1: CONSERVATIVE STRATEGY


#We build a trading strategy based on the filtered regime probabilities from the HMM:
  
#  * Avoid exposure if the probability of being in State 2 is > 40%
#(this regimes show poor risk-adjusted performance)
#* Stay invested if in State 1 or 3, which offer better return/risk profiles

#This rule is:
  
#  * Updated daily
#  * Binary (0 or 1) exposure
#  * Tested out-of-sample (2023–2025) using model trained on 2016–2022

#Goal: Preserve capital in high-risk regimes while capturing upside during favorable phases.


# Step 0: Use filtering probabilities
posterior_probs <- filtering  # dimensions: m × T

# Step 1: Lag posterior probabilities across time
posterior_probs_lagged <- posterior_probs[, -ncol(posterior_probs)]
posterior_probs_lagged <- cbind(rep(NA, nrow(posterior_probs)), posterior_probs_lagged)
colnames(posterior_probs_lagged) <- colnames(posterior_probs)

# Step 2: Risky states and thresholds
thresholds <- c( "2" = 0.4)

# Step 3: Define valid test index range
max_t <- ncol(posterior_probs_lagged)
adjusted_test_idx <- test_idx[test_idx <= max_t]  # prevent out-of-bounds error

# Step 4: Generate exit signals
exit_signal <- sapply(adjusted_test_idx, function(t) {
  any(posterior_probs_lagged[as.numeric(names(thresholds)), t] > thresholds)
})
signal_test <- ifelse(exit_signal, 0, 1)

# Step 5: Align test returns/dates
returns_test <- returns_full[adjusted_test_idx]
dates_test <- dates_full[adjusted_test_idx]

# Step 6: Compute strategy and benchmark
strategy_returns_test <- signal_test * returns_test
strategy_xts <- xts(strategy_returns_test, order.by = dates_test)
benchmark_xts <- xts(returns_test, order.by = dates_test)



# Combine strategy and benchmark returns into a single xts object
returns_df <- na.omit(merge(strategy_xts, benchmark_xts))
colnames(returns_df) <- c("Regime_Strategy", "Benchmark")

# === Cumulative returns
cum_returns <- cumprod(1 + returns_df) - 1
cum_returns_df <- fortify.zoo(cum_returns, name = "Date")
cum_returns_df <- melt(cum_returns_df, id.vars = "Date")

# === Drawdowns
drawdowns <- PerformanceAnalytics::Drawdowns(returns_df)
drawdowns_df <- fortify.zoo(drawdowns, name = "Date")
drawdowns_df <- melt(drawdowns_df, id.vars = "Date")


library(scales)  

# Plot 1: Cumulative returns
p1 <- ggplot(cum_returns_df, aes(x = Date, y = value, color = variable)) +
  geom_line(size = 0.8) +
  geom_vline(xintercept = as.Date("2023-01-01"), linetype = "dashed", color = "gray40") +
  labs(
    title = "Cumulative Returns: Regime-Based Strategy vs Benchmark (2023–2025)",
    y = "Cumulative Return",
    x = NULL,
    color = "Strategy"
  ) +
  scale_color_manual(values = c("Regime_Strategy" = "steelblue", "Benchmark" = "darkgray")) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  theme_minimal() +
  theme(legend.position = "top")

# Plot 2: Drawdowns
p2 <- ggplot(drawdowns_df, aes(x = Date, y = value, fill = variable)) +
  geom_area(position = "identity", alpha = 0.4) +
  labs(
    title = "Drawdown Comparison (2023–2025)",
    y = "Drawdown",
    x = "Date",
    fill = "Strategy"
  ) +
  scale_fill_manual(values = c("Regime_Strategy" = "steelblue", "Benchmark" = "darkgray")) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  theme_minimal() +
  theme(legend.position = "none")

# Combine plots using patchwork
p1 / p2 + plot_layout(heights = c(2, 1))



# Ensure column names
colnames(returns_df) <- c("Regime_Strategy", "Benchmark")

# 1. Cumulative return
cum_returns <- cumprod(1 + returns_df) - 1
final_cum_return <- round(as.numeric(tail(cum_returns, 1)), 4)

# 2. Annualized performance
annual_stats <- round(table.AnnualizedReturns(returns_df), 4)

# 3. Max drawdown
drawdowns <- round(maxDrawdown(returns_df), 4)

# 4. Combine all in one table
performance_summary <- data.frame(
  Metric = c(
    "Cumulative Return",
    "Annualized Return",
    "Annualized Volatility",
    "Sharpe Ratio (Rf=0%)",
    "Maximum Drawdown"
  ),
  Regime_Strategy = c(
    final_cum_return[1],
    annual_stats[1, "Regime_Strategy"],
    annual_stats[2, "Regime_Strategy"],
    annual_stats[3, "Regime_Strategy"],
    drawdowns[1]
  ),
  Benchmark = c(
    final_cum_return[2],
    annual_stats[1, "Benchmark"],
    annual_stats[2, "Benchmark"],
    annual_stats[3, "Benchmark"],
    drawdowns[2]
  )
)

# View table
print(performance_summary)






#########################################################################################
#########################################################################################
#########################################################################################


#STRATEGY 2: LONG, SHORT AND LAVERAGE 



signal_test <- sapply(adjusted_test_idx, function(t) {
  probs <- posterior_probs_lagged[, t]
  if (which.max(probs) == 1) return(2)  
  if (which.max(probs) == 2) return(-1)   
  if (which.max(probs) == 3) return(2)   
})

# Step 5: Align test returns/dates
returns_test <- returns_full[adjusted_test_idx]
dates_test <- dates_full[adjusted_test_idx]

# Step 6: Compute strategy and benchmark
strategy_returns_test <- signal_test * returns_test
strategy_xts <- xts(strategy_returns_test, order.by = dates_test)
benchmark_xts <- xts(returns_test, order.by = dates_test)


# Combine strategy and benchmark returns into a single xts object
returns_df <- na.omit(merge(strategy_xts, benchmark_xts))
colnames(returns_df) <- c("Regime_Strategy", "Benchmark")

# === Cumulative returns
cum_returns <- cumprod(1 + returns_df) - 1
cum_returns_df <- fortify.zoo(cum_returns, name = "Date")
cum_returns_df <- melt(cum_returns_df, id.vars = "Date")

# === Drawdowns
drawdowns <- PerformanceAnalytics::Drawdowns(returns_df)
drawdowns_df <- fortify.zoo(drawdowns, name = "Date")
drawdowns_df <- melt(drawdowns_df, id.vars = "Date")
library(scales)  

# Plot 1: Cumulative returns
p1 <- ggplot(cum_returns_df, aes(x = Date, y = value, color = variable)) +
  geom_line(size = 0.8) +
  geom_vline(xintercept = as.Date("2023-01-01"), linetype = "dashed", color = "gray40") +
  labs(
    title = "Cumulative Returns: Regime-Based Strategy vs Benchmark (2023–2025)",
    y = "Cumulative Return",
    x = NULL,
    color = "Strategy"
  ) +
  scale_color_manual(values = c("Regime_Strategy" = "steelblue", "Benchmark" = "darkgray")) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  theme_minimal() +
  theme(legend.position = "top")

# Plot 2: Drawdowns
p2 <- ggplot(drawdowns_df, aes(x = Date, y = value, fill = variable)) +
  geom_area(position = "identity", alpha = 0.4) +
  labs(
    title = "Drawdown Comparison (2023–2025)",
    y = "Drawdown",
    x = "Date",
    fill = "Strategy"
  ) +
  scale_fill_manual(values = c("Regime_Strategy" = "steelblue", "Benchmark" = "darkgray")) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  theme_minimal() +
  theme(legend.position = "none")

# Combine plots using patchwork
p1 / p2 + plot_layout(heights = c(2, 1))

# Ensure column names
colnames(returns_df) <- c("Regime_Strategy", "Benchmark")

# 1. Cumulative return
cum_returns <- cumprod(1 + returns_df) - 1
final_cum_return <- round(as.numeric(tail(cum_returns, 1)), 4)

# 2. Annualized performance
annual_stats <- round(table.AnnualizedReturns(returns_df), 4)

# 3. Max drawdown
drawdowns <- round(maxDrawdown(returns_df), 4)

# 4. Combine all in one table
performance_summary <- data.frame(
  Metric = c(
    "Cumulative Return",
    "Annualized Return",
    "Annualized Volatility",
    "Sharpe Ratio (Rf=0%)",
    "Maximum Drawdown"
  ),
  Regime_Strategy = c(
    final_cum_return[1],
    annual_stats[1, "Regime_Strategy"],
    annual_stats[2, "Regime_Strategy"],
    annual_stats[3, "Regime_Strategy"],
    drawdowns[1]
  ),
  Benchmark = c(
    final_cum_return[2],
    annual_stats[1, "Benchmark"],
    annual_stats[2, "Benchmark"],
    annual_stats[3, "Benchmark"],
    drawdowns[2]
  )
)

# View table
print(performance_summary)






#########################################################################################
#########################################################################################
#########################################################################################



# ===========================
# STRATEGY: ACTIVE MANAGEMENT WITH RSI IN STATE 3
# ===========================



# ===========================
# STRATEGY: HMM + RSI for State 3, Fixed Leverage in State 4
# ===========================

# Strategy Description:
# This strategy combines:
# - The probabilities of latent regimes (hidden states) obtained via the HMM;
# - A technical indicator: RSI (Relative Strength Index);
# - Fixed decisions for “extreme” states (1 and 4), but active management in the intermediate regime (3).

#️ Trading rules applied each day during the test period:

# 1. State 1 (positive regime): 
#    - Enter long with 2x leverage (position 2).
#    - This is the most favorable regime with contained volatility → ideal for strong exposure.


# 2. Stato 2 (regime instabile):
#    - If RSI < 30 → market is oversold → enter long (1).
#    - If RSI > 70 → market is overbought → enter short (-2).
#    - Otherwise → neutral, no position (-1).
#    - RSI helps navigate the unstable nature of this regime.


# 3. Stato 3 (regime negativo e turbolento):
#    → long +2



#What does it mean Oversold (RSI < 30)?

#The market has suffered significant recent losses → the price fell too quickly.

#RSI < 30 means there has been very strong selling pressure.

#Interpretation: The stock may be "oversold" → a rebound is likely.

#Strategy: Go long (buy) → position +1.


#What does it mean Overbought (overbought, RSI > 70)?

#The market has seen significant recent gains → the price has risen too quickly.

#RSI > 70 means that buying pressure has been very strong.

#Interpretation: The stock may be "overbought" → a downward correction is likely.

#Strategy: Short entry (selling) → position -1.



library(TTR)  # per RSI

# Calcola RSI su tutti i returns (es. 14 giorni)
rsi_values <- RSI(returns_full, n = 14)

# Crea segnali dinamici su periodo di test
signal_test_rsi <- sapply(adjusted_test_idx, function(t) {
  # Se RSI non è disponibile (inizio serie), escludi
  if (is.na(rsi_values[t])) return(NA)
  
  probs <- posterior_probs_lagged[, t]
  state <- which.max(probs)
  rsi <- rsi_values[t]
  
  if (state == 1) {
    return(2)  
  } else if (state == 3) {
    return(2)  
  } else if (state == 2) {
    if (rsi < 30) {
      return(1)   # RSI basso → long
    } else if (rsi > 70) {
      return(-2)  # RSI alto → short aggressivo
    } else {
      return(-1)   # Neutro
    }
  } else {
    return(0)   # fallback
  }
})

# Allineamento
returns_test <- returns_full[adjusted_test_idx]
dates_test <- dates_full[adjusted_test_idx]

# Rendimenti della strategia
strategy_rsi_returns <- signal_test_rsi * returns_test
strategy_rsi_xts <- xts(strategy_rsi_returns, order.by = dates_test)

# Benchmark (buy & hold)
benchmark_xts <- xts(returns_test, order.by = dates_test)

# Dataset finale
returns_df_rsi <- na.omit(merge(strategy_rsi_xts, benchmark_xts))
colnames(returns_df_rsi) <- c("Strategy_RSI", "Benchmark")



library(PerformanceAnalytics)
library(ggplot2)
library(reshape2)
library(scales)
library(patchwork)

# === Cumulative returns
cum_returns_rsi <- cumprod(1 + returns_df_rsi) - 1
cum_returns_df_rsi <- fortify.zoo(cum_returns_rsi, name = "Date")
cum_returns_df_rsi <- melt(cum_returns_df_rsi, id.vars = "Date")

# === Drawdowns
drawdowns_rsi <- PerformanceAnalytics::Drawdowns(returns_df_rsi)
drawdowns_df_rsi <- fortify.zoo(drawdowns_rsi, name = "Date")
drawdowns_df_rsi <- melt(drawdowns_df_rsi, id.vars = "Date")

# === Plot 1: Cumulative returns
p1_rsi <- ggplot(cum_returns_df_rsi, aes(x = Date, y = value, color = variable)) +
  geom_line(size = 0.8) +
  geom_vline(xintercept = as.Date("2023-01-01"), linetype = "dashed", color = "gray40") +
  labs(
    title = "Cumulative Returns: Strategy RSI vs Benchmark (2023–2025)",
    y = "Cumulative Return",
    x = NULL,
    color = "Strategy"
  ) +
  scale_color_manual(values = c("Strategy_RSI" = "steelblue", "Benchmark" = "darkgray")) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  theme_minimal() +
  theme(legend.position = "top")

# === Plot 2: Drawdowns
p2_rsi <- ggplot(drawdowns_df_rsi, aes(x = Date, y = value, fill = variable)) +
  geom_area(position = "identity", alpha = 0.4) +
  labs(
    title = "Drawdown Comparison: Strategy RSI vs Benchmark",
    y = "Drawdown",
    x = "Date",
    fill = "Strategy"
  ) +
  scale_fill_manual(values = c("Strategy_RSI" = "steelblue", "Benchmark" = "darkgray")) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  theme_minimal() +
  theme(legend.position = "none")

# === Combine plots
p1_rsi / p2_rsi + plot_layout(heights = c(2, 1))


##Tabella di Performance###

# Calcolo metriche di performance
colnames(returns_df_rsi) <- c("Strategy_RSI", "Benchmark")

# 1. Rendimento cumulativo
cum_returns_final_rsi <- cumprod(1 + returns_df_rsi) - 1
final_cum_return_rsi <- round(as.numeric(tail(cum_returns_final_rsi, 1)), 4)

# 2. Rendimento e volatilità annualizzati
annual_stats_rsi <- round(table.AnnualizedReturns(returns_df_rsi), 4)

# 3. Maximum drawdown
drawdowns_rsi <- round(maxDrawdown(returns_df_rsi), 4)

# 4. Tabella finale
performance_summary_rsi <- data.frame(
  Metric = c(
    "Cumulative Return",
    "Annualized Return",
    "Annualized Volatility",
    "Sharpe Ratio (Rf=0%)",
    "Maximum Drawdown"
  ),
  Strategy_RSI = c(
    final_cum_return_rsi[1],
    annual_stats_rsi[1, "Strategy_RSI"],
    annual_stats_rsi[2, "Strategy_RSI"],
    annual_stats_rsi[3, "Strategy_RSI"],
    drawdowns_rsi[1]
  ),
  Benchmark = c(
    final_cum_return_rsi[2],
    annual_stats_rsi[1, "Benchmark"],
    annual_stats_rsi[2, "Benchmark"],
    annual_stats_rsi[3, "Benchmark"],
    drawdowns_rsi[2]
  )
)

# Visualizza
print(performance_summary_rsi)















#########################################################################################
#########################################################################################
#########################################################################################



# ================================================
# STRATEGY: Regime-Adaptive with Dynamic Leverage and RSI
# ================================================
# Description:
# This strategy combines the filtered probabilities from a 3-state Hidden Markov Model (HMM)
# with the Relative Strength Index (RSI) indicator to make regime-aware trading decisions
# with dynamic exposure levels (positions: -2, -1, 1, 2).

# ️ Trading Logic:
# -  State 1 (stable and positive regime):
#     → LONG with 2x leverage if the probability > 60%, otherwise standard LONG.
# -  State 2 (intermediate and volatile regime):
#     → RSI < 30 → LONG with 2x leverage (oversold market)
#     → RSI > 70 → SHORT with 2x leverage (overbought market)
#     → Otherwise → moderate SHORT (no leverage)
# -  State 3 (controlled growth regime):
#     → LONG with 2x leverage if the probability > 60%, otherwise standard LONG.

#  Objective:
# Increase exposure in highly confident, favorable regimes;
# use RSI-based tactics in unstable conditions;
# and dynamically adjust leverage to balance risk and opportunity.



library(TTR)  # per calcolare RSI

# Calcola RSI su tutta la serie
rsi_values <- RSI(returns_full, n = 14)

# Genera segnali su periodo di test
signal_confident <- sapply(adjusted_test_idx, function(t) {
  # Escludi se mancano dati
  if (is.na(rsi_values[t])) return(NA)
  
  probs <- posterior_probs_lagged[, t]
  state <- which.max(probs)
  prob <- probs[state]
  rsi <- rsi_values[t]
  
  if (state == 3){
    if (prob > 0.6){
      return(2)
    }else{
      return(1)}
    } else if (state == 2) {
    if ( rsi < 30) {
      return(2)  
    } else if (rsi > 70) {
      return(-2)  
    } else {
      return(-1)   
    }
  } else if (state == 1) {
    if (prob > 0.6) {
      return(2)   # Long con leva 2x se molto probabile
    }else {
      return(1)   
    }
  } else {
    return(0)
  }
})



# === Costruzione strategia e benchmark
returns_test <- returns_full[adjusted_test_idx]
dates_test <- dates_full[adjusted_test_idx]
strategy_xts_conf <- xts(signal_confident * returns_test, order.by = dates_test)
benchmark_xts <- xts(returns_test, order.by = dates_test)

# === Dataset completo
returns_df_conf <- na.omit(merge(strategy_xts_conf, benchmark_xts))
colnames(returns_df_conf) <- c("Strategy_Advanced", "Benchmark")

# === Cumulative returns
cum_returns_conf <- cumprod(1 + returns_df_conf) - 1
cum_returns_df_conf <- fortify.zoo(cum_returns_conf, name = "Date")
cum_returns_df_conf <- melt(cum_returns_df_conf, id.vars = "Date")

# === Drawdowns
drawdowns_conf <- PerformanceAnalytics::Drawdowns(returns_df_conf)
drawdowns_df_conf <- fortify.zoo(drawdowns_conf, name = "Date")
drawdowns_df_conf <- melt(drawdowns_df_conf, id.vars = "Date")

# === Plots
p1_conf <- ggplot(cum_returns_df_conf, aes(x = Date, y = value, color = variable)) +
  geom_line(size = 0.8) +
  geom_vline(xintercept = as.Date("2023-01-01"), linetype = "dashed", color = "gray40") +
  labs(title = "Cumulative Returns: Advanced Strategy vs Benchmark",
       y = "Cumulative Return", x = NULL, color = "Strategy") +
  scale_color_manual(values = c("Strategy_Advanced" = "steelblue", "Benchmark" = "darkgray")) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  theme_minimal() +
  theme(legend.position = "top")

p2_conf <- ggplot(drawdowns_df_conf, aes(x = Date, y = value, fill = variable)) +
  geom_area(position = "identity", alpha = 0.4) +
  labs(title = "Drawdown Comparison: Advanced Strategy vs Benchmark",
       y = "Drawdown", x = "Date", fill = "Strategy") +
  scale_fill_manual(values = c("Strategy_Advanced" = "steelblue", "Benchmark" = "darkgray")) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  theme_minimal() +
  theme(legend.position = "none")

# === Visualizza grafici
p1_conf / p2_conf + plot_layout(heights = c(2, 1))

# === Tabella performance finale
colnames(returns_df_conf) <- c("Strategy_Advanced", "Benchmark")
cum_returns_final_conf <- cumprod(1 + returns_df_conf) - 1
final_cum_return_conf <- round(as.numeric(tail(cum_returns_final_conf, 1)), 4)
annual_stats_conf <- round(table.AnnualizedReturns(returns_df_conf), 4)
drawdowns_conf <- round(maxDrawdown(returns_df_conf), 4)

performance_summary_conf <- data.frame(
  Metric = c(
    "Cumulative Return",
    "Annualized Return",
    "Annualized Volatility",
    "Sharpe Ratio (Rf=0%)",
    "Maximum Drawdown"
  ),
  Strategy_Advanced = c(
    final_cum_return_conf[1],
    annual_stats_conf[1, "Strategy_Advanced"],
    annual_stats_conf[2, "Strategy_Advanced"],
    annual_stats_conf[3, "Strategy_Advanced"],
    drawdowns_conf[1]
  ),
  Benchmark = c(
    final_cum_return_conf[2],
    annual_stats_conf[1, "Benchmark"],
    annual_stats_conf[2, "Benchmark"],
    annual_stats_conf[3, "Benchmark"],
    drawdowns_conf[2]
  )
)

print(performance_summary_conf)














#########################################################################################
#########################################################################################
#########################################################################################


# ================================================
# STRATEGY: HMM + RSI + Term Spread (3-state model)
# ================================================

#  Description:
# This trading strategy integrates:
# - Regime probabilities from a 3-state Hidden Markov Model (HMM)
# - A technical indicator: RSI (Relative Strength Index) to detect overbought and oversold conditions;
# - A macroeconomic filter: the Term Spread (T10Y2Y = 10Y - 2Y Treasury yield)

# What is the Term Spread (T10Y2Y)?
# --------------------------------
# The “term spread” is the difference between the yield on 10-year U.S. Treasuries and the yield on 2-year Treasuries:
#     TERM_SPREAD = Yield(10Y) - Yield(2Y)
# We download it from the FRED series “T10Y2Y”.
#
# It is a widely used indicator to measure the shape of the yield curve:
# - If positive → the curve is upward sloping (normal), meaning markets are optimistic about future growth.
# - If negative → the curve is inverted, reflecting pessimistic expectations (historically a recession signal).
#
# In this strategy, the term spread is used as a macroeconomic filter:
# - If the spread < 0 → avoid long exposure (more caution).
# - If the spread ≥ 0 → be more willing to take risk, depending on the HMM state and RSI value.


# ️ Daily Decision Logic:

#  State 1 (Stable & Positive Regime):
# - If TERM_SPREAD ≥ 0 → enter LONG with 2x leverage (position 2)
# - If TERM_SPREAD < 0 → reduce risk → standard LONG (position 1)

#  State 2 (Volatile / Speculative Regime):
# - If TERM_SPREAD ≥ 0:
#     → RSI < 30 → market oversold → LONG (position 2)
#     → RSI > 70 → market overbought → SHORT (position -2)
#     → Otherwise → -1
# - If TERM_SPREAD < 0 → -2

#  State 3 (Negative / Loss Regime):
# - If TERM_SPREAD < 0 → enter SHORT (position -1)
# - If TERM_SPREAD ≥ 0 → +2

#  Objective:
# Use the HMM to identify market regimes, enhance decisions with RSI signals,
# and adjust risk-taking according to macroeconomic conditions (term spread).

if (!require("quantmod")) install.packages("quantmod")
library(quantmod)


spread_df <- read.csv("T10Y2Y.csv", stringsAsFactors = FALSE) 
spread_xts <- xts(as.numeric(spread_df[, 2]), order.by = as.Date(spread_df[, 1]))  
colnames(spread_xts) <- "T10Y2Y"  

# Compute RSI over the full return series
rsi_values <- RSI(returns_full, n = 14)

# Generate trading signals for the strategy
signal_test_spread <- sapply(adjusted_test_idx, function(t) {
  date_t <- dates_full[t]  
  if (is.na(rsi_values[t]) || is.na(spread_xts[as.character(date_t)])) return(NA)  
  
  probs <- posterior_probs_lagged[, t]
  state <- which.max(probs)
  prob_state <- probs[state]
  rsi <- rsi_values[t]
  spread <- as.numeric(spread_xts[as.character(date_t)])  
  
  # Strategy rules
  if (state == 1) {
    if (spread >= 0) return(2) else return(1)
  } else if (state == 2) {
    if (spread >= 0) {
      if (rsi < 30) return(2)
      else if (rsi > 70) return(-2)
      else return(-1)
    } else return(2)
  } else if (state == 3) {
    if (spread < 0) return(2)
    else return(1)
  } else {
    return(0)
  }
})

# Align returns and dates during test period
returns_test <- returns_full[adjusted_test_idx]
dates_test <- dates_full[adjusted_test_idx]
strategy_spread_returns <- signal_test_spread * returns_test

strategy_spread_xts <- xts(strategy_spread_returns, order.by = dates_test)
benchmark_xts <- xts(returns_test, order.by = dates_test)

# Combine and clean returns for both strategy and benchmark
returns_df_spread <- na.omit(merge(strategy_spread_xts, benchmark_xts))
colnames(returns_df_spread) <- c("Strategy_Spread", "Benchmark")

# === Cumulative returns
cum_returns_spread <- cumprod(1 + returns_df_spread) - 1
cum_returns_df <- fortify.zoo(cum_returns_spread, name = "Date")
cum_returns_df <- reshape2::melt(cum_returns_df, id.vars = "Date")

# === Drawdowns
drawdowns_spread <- PerformanceAnalytics::Drawdowns(returns_df_spread)
drawdowns_df <- fortify.zoo(drawdowns_spread, name = "Date")
drawdowns_df <- reshape2::melt(drawdowns_df, id.vars = "Date")

# === Plot: Cumulative Returns
p1 <- ggplot(cum_returns_df, aes(x = Date, y = value, color = variable)) +
  geom_line(size = 0.8) +
  scale_color_manual(values = c("Strategy_Spread" = "blue", "Benchmark" = "darkgray")) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_x_date(date_breaks = "3 months", date_labels = "%Y-%m") +
  geom_vline(xintercept = as.Date("2023-01-01"), linetype = "dashed", color = "gray40") +
  labs(
    title = "Cumulative Returns: Spread Strategy vs Benchmark",
    y = "Cumulative Return", x = "Date", color = "Strategy"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top")

# === Plot: Drawdowns
p2 <- ggplot(drawdowns_df, aes(x = Date, y = value, fill = variable)) +
  geom_area(position = "identity", alpha = 0.4) +
  scale_fill_manual(values = c("Strategy_Spread" = "blue", "Benchmark" = "darkgray")) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_x_date(date_breaks = "3 months", date_labels = "%Y-%m") +
  labs(
    title = "Drawdowns: Spread Strategy vs Benchmark",
    y = "Drawdown", x = "Date", fill = "Strategy"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")

# === Combine plots
p1 / p2 + plot_layout(heights = c(2, 1))

# === Performance Table
colnames(returns_df_spread) <- c("Strategy_Spread", "Benchmark")
cum_returns_final <- cumprod(1 + returns_df_spread) - 1
final_cum_return <- round(as.numeric(tail(cum_returns_final, 1)), 4)
annual_stats <- round(table.AnnualizedReturns(returns_df_spread), 4)
drawdowns <- round(maxDrawdown(returns_df_spread), 4)

performance_summary_spread <- data.frame(
  Metric = c(
    "Cumulative Return",
    "Annualized Return",
    "Annualized Volatility",
    "Sharpe Ratio (Rf=0%)",
    "Maximum Drawdown"
  ),
  Strategy_Spread = c(
    final_cum_return[1],
    annual_stats[1, "Strategy_Spread"],
    annual_stats[2, "Strategy_Spread"],
    annual_stats[3, "Strategy_Spread"],
    drawdowns[1]
  ),
  Benchmark = c(
    final_cum_return[2],
    annual_stats[1, "Benchmark"],
    annual_stats[2, "Benchmark"],
    annual_stats[3, "Benchmark"],
    drawdowns[2]
  )
)

# === Print Performance Summary
print(performance_summary_spread)
