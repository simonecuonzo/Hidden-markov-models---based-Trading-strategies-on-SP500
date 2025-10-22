This study introduces a regime-switching approach for capturing and predicting the behavior of 
financial markets, employing a Hidden Markov Model (HMM) with skew-normal emission distributions 
applied to daily returns of the S&P500 index from 2016 to 2025. Unlike traditional Gaussian-based 
models, this formulation allows for asymmetric returns and volatility clustering, thus reflecting 
important non-linear characteristics commonly observed in equity markets.

Using model selection criteria such as Akaike Information Criterion (AIC) and Bayesian 
Information Criterion (BIC), a three-regime specification was selected as the optimal 
trade-off between complexity and interpretability. Each latent state displays distinct 
statistical properties.

Building on this classification, we design 5 different regime-sensitive trading strategies, which adapts market exposure based on the prevailing latent state,
each progressively integrating additional sources of information — from statistical regime detection to technical and macroeconomic indicators — in order to dynamically adjust market exposure and control risk.
