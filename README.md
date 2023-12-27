# Strategy using Beta_Asymmetry in S&P500 stocks
### Alpha Strategy using the market feature that stocks have different betas depend on market situation, going up or down
### We define 'upbeta' as the beta when market goes up, and 'downbeta' as the beta when market goes down
### Using this feature, we can make straddle-like strategy without disadvantage of options, only using market ETF
### But it has quite high volatility, so to make smooth PnL graph, I used strategy-itself momentum to control the position.
### The code "main.ipynb" is to see how code works and it imports original module "pybeta_asymmetry.py" and uses the classes in it.
### I tried to finetune the hyperparameters in my strategy and the result is shown below.

