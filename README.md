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
