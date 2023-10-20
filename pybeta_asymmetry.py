# 1957-03-01~ Data에 맞게 수정됨
# Weight NaN 0으로 설정 완료
# trim 양방향으로 적용되지 않던 것 수정됨

# Core Libaries
import pandas as pd # pip install pandas
import numpy as np # pip install numpy
import numba # pip install numba
from scipy.stats.mstats import winsorize # pip install scipy
from numba_progress import ProgressBar # pip install numba-progress
from tqdm import tqdm # pip install tqdm
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings(action='ignore')

# OLS from Var-Covar
# Beta Calculator with Numpy & Numba with OLS
# For calcuation only
@numba.njit(nogil=True)
def beta_via_ols_(
    num_iterations, # for ProgressBar (which is equal to len(stock_matrix)
    progress_update, # for ProgressBar (which is equal to len(stock_matrix)
    stock_matrix, # matrix (2d np.array)
    market_vector, # vector (1d np.array)
    window_size, # int
    beta_type="up", # str ('up', 'down', 'all')
    white_std_err=False # White Standard Error
):
    beta_matrix = np.abs(stock_matrix.copy()) * 0 # Result matrix for Beta
    t_matrix = beta_matrix.copy()
    
    lack_of_data = np.array([np.nan])

    for stock_cnt in range(num_iterations): # For each stock (each iteration)
        each_stock = stock_matrix[stock_cnt]
        each_market = market_vector.copy()
        nan_index = np.where(np.isnan(each_stock))[0] # array index for NaN values
        nan_removed_index = np.array([idx for idx in [i for i in range(len(each_stock))] if idx not in (nan_index)])
        if len(nan_index) == 0:
            pass
        else:
            each_stock = each_stock[nan_removed_index] # NaN removed stock return data
            each_market = each_market[nan_removed_index] # Matching the market return with stock return
        
        beta_vector = []
        t_vector = []
        for date_cnt in range(len(each_stock)-window_size+1): # Includes today's return for beta calculation
            stock_window = each_stock[date_cnt:date_cnt+window_size]
            market_window = each_market[date_cnt:date_cnt+window_size]

            # Screen windows with up/down side return
            if beta_type == "up":
                beta_type_index = np.where(market_window > 0)[0]
            elif beta_type == "down":
                beta_type_index = np.where(market_window <= 0)[0]
            elif beta_type == "all":
                beta_type_index = np.where(market_window != np.nan)[0]
            
            stock_window = stock_window[beta_type_index]
            market_window = market_window[beta_type_index].reshape(1,-1)[0] # Reshape to 1d

            ### Beta Calculation Part ###
            beta = (np.cov(stock_window, market_window) / np.var(market_window))[0][1]
            beta_vector.append(beta)
            residual_vector = stock_window - (beta*market_window)
            ### END ###

            sigma_square_hat = np.sum(np.square(residual_vector)) / (len(stock_window)-2)

            if (sigma_square_hat != 0) and (np.var(market_window) != 0):
                if white_std_err == False:
                    standard_error = np.sqrt( sigma_square_hat / np.sum(np.square(market_window - np.mean(market_window))) )
                elif white_std_err == True:
                    standard_error = np.sqrt(
                                        np.sum(np.square(market_window - np.mean(market_window)) * np.square(residual_vector))
                                        /
                                        np.square(np.sum(np.square(market_window - np.mean(market_window))))
                                        )

                t_statistic = beta/standard_error
                t_vector.append(t_statistic)
            else:
                t_vector.append(0)

        if len(nan_index) == 0:
            beta_matrix[stock_cnt]= np.nan
            t_matrix[stock_cnt] = np.nan
            beta_matrix[stock_cnt][window_size-1:] = np.array(beta_vector)
            t_matrix[stock_cnt][window_size-1:] = np.array(t_vector)
        else:
            beta_vector = np.append(np.array([np.nan for i in range(window_size-1)]), np.array(beta_vector))
            t_vector = np.append(np.array([np.nan for i in range(window_size-1)]), np.array(t_vector))
            if len(beta_vector) == len(beta_matrix[stock_cnt][nan_removed_index]):
                beta_matrix[stock_cnt][nan_removed_index] = beta_vector
                t_matrix[stock_cnt][nan_removed_index] = t_vector
            else:
                beta_matrix[stock_cnt][window_size-1:] = np.nan
                t_matrix[stock_cnt][window_size-1:] = np.nan
                lack_of_data = np.append(lack_of_data, stock_cnt)
        
        progress_update.update(1) # ProgressBar update

    return beta_matrix, t_matrix, lack_of_data

### Beta Calculation Class for Warpping Beta Calculator ###
class BETA_CALCULATOR:
    def __init__(self):
        pass

    def run(
        self,
        stock_return, # pd.DataFrame (columns:unique code, index:date, values:return)
        market_return, # pd.DataFrame (columns:unique code, index:date, values:return)
        window, # int
        type="up", # str ('up', 'down', 'all')
        white=False, # White Standard Error
    ):
        if type not in ['up','down','all']:
            raise ValueError("beta_type must be one in ['up','down','all']")
        if white not in [True, False]:
            raise ValueError("white must be True or False")

        print("Please Wait...")

        stock_input = stock_return.replace([np.inf, -np.inf], np.nan) # Replace np.inf to np.nan
        market_input = market_return.replace([np.inf, -np.inf], np.nan)

        stock_input = stock_input.values.transpose() # Transform to numpy format for fast calculation
        market_input = market_input.values.transpose()[0]

        num_iterations = len(stock_return.columns) # number of numba iterations for ProgressBar

        with ProgressBar(total=num_iterations) as progress_update:
            self.beta, self.tstat, self.lack = beta_via_ols_(
                                                    num_iterations,
                                                    progress_update,
                                                    stock_matrix=stock_input,
                                                    market_vector=market_input,
                                                    window_size=window,
                                                    beta_type=type,
                                                    white_std_err=white
                                                )

        self.beta = pd.DataFrame(self.beta.transpose(), columns=stock_return.columns, index=stock_return.index)
        self.tstat = pd.DataFrame(self.tstat.transpose(), columns=stock_return.columns, index=stock_return.index)

        self.lack = [int(x) for x in self.lack if np.isnan(x) == False]
        self.lack = stock_return.columns[self.lack]
        print("### Calculation Complete ###")
        print("Check the result with .beta & .tstat & .lack")
        print()
        print("Warning! Below stocks don't have enough observations to calculate Beta under given window size(%s)" % (window))
        print(self.lack.values)

### Preprocessor Class for Winsorizaton & Replacing ###
class PREPROCESSOR:

    def do_winsorize(data, method='quantile', quantile=None, hurdle=None):
        print("Please Wait...")
        tmp = data.copy()
        if method == 'quantile':
            if quantile == None:
                raise ValueError("you should set quantile")
            for idx in tqdm(tmp.index, total=len(tmp.index), desc='Winsorizing...'):
                tmp.loc[idx] = tmp.loc[idx].clip(tmp.loc[idx].quantile(quantile), tmp.loc[idx].quantile(1-quantile))
        elif method == 'hurdle':
            if hurdle == None:
                raise ValueError("you shoud set hurdle")
            tmp = tmp.applymap(lambda x: hurdle if x > hurdle else x)
            tmp = tmp.applymap(lambda x: -hurdle if x < -hurdle else x)
        elif method == 'both':
            if (quantile == None) or (hurdle == None):
                raise ValueError("you should set both quantile and hurdle")
            for idx in tqdm(tmp.index, total=len(tmp.index), desc='Winsorizing...'):
                tmp.loc[idx] = tmp.loc[idx].clip(tmp.loc[idx].quantile(quantile), tmp.loc[idx].quantile(1-quantile))
            tmp = tmp.applymap(lambda x: hurdle if x > hurdle else x)
            tmp = tmp.applymap(lambda x: -hurdle if x < -hurdle else x)        
        return tmp
    
    def replace_extreme(data, hurdle):
        result = data.copy()
        result = result.applymap(lambda x: np.nan if abs(x)>hurdle else x)
        return result

### Strategy Class for Creating Straddle Position ###
### Strategy Class for Creating Straddle Position ###
class STRATEGY:
    def __init__(self):
        pass

    def straddle(
        self,
        upbeta,
        downbeta,
        long_stock_down_hurdle=0.7,
        short_stock_up_hurdle=0.7,
        up_minus_down_hurdle=0.25,
        down_minus_up_hurdle=0.25,
        momentum_weight=[]
    ):
        print("Please Wait...")
        
        call_stock_beta = upbeta - downbeta
        put_stock_beta = downbeta - upbeta

        call_stock_beta  = call_stock_beta.applymap(lambda x: np.nan if x < 0 else x)
        put_stock_beta = put_stock_beta.applymap(lambda x: np.nan if x < 0 else x) 

        call_stock_beta_masking = upbeta.copy().applymap(lambda x: 1 if x < long_stock_down_hurdle else np.nan)
        put_stock_beta_masking = downbeta.copy().applymap(lambda x: 1 if x < short_stock_up_hurdle else np.nan)

        long_stock = call_stock_beta.applymap(lambda x: 1 if x > up_minus_down_hurdle else np.nan) * call_stock_beta_masking
        short_stock = put_stock_beta.applymap(lambda x: 1 if x > down_minus_up_hurdle else np.nan) * put_stock_beta_masking 

        beta_need_short_hedge_from_long_stock = (long_stock * downbeta).apply(lambda x: x.mean(), axis=1) 
        beta_need_long_hedge_from_short_stock = (short_stock * upbeta).apply(lambda x: x.mean(), axis=1) 

        if len(momentum_weight) == 0:
            long_stock_weight = 2/(1-beta_need_short_hedge_from_long_stock)
            long_stock_weight = long_stock_weight.fillna(0)
            short_stock_weight = -1/(1-beta_need_long_hedge_from_short_stock)
            short_stock_weight = short_stock_weight.fillna(0)
            etf_weight = pd.DataFrame(1 - (long_stock_weight + short_stock_weight))
        else:
            long_stock_weight = (2*momentum_weight.values.flatten())/(1-beta_need_short_hedge_from_long_stock.values.flatten())
            short_stock_weight = (-1*momentum_weight.values.flatten())/(1-beta_need_long_hedge_from_short_stock.values.flatten())
            print(long_stock_weight.shape)
            print(pd.DataFrame(long_stock_weight).shape)
            long_stock_weight = pd.DataFrame(long_stock_weight).fillna(0)
            short_stock_weight = pd.DataFrame(short_stock_weight).fillna(0)
            etf_weight = pd.DataFrame(momentum_weight.values.flatten() - (long_stock_weight.values.flatten() + short_stock_weight.values.flatten()))
        etf_weight.index = pd.to_datetime(upbeta.index)
        long_stock_weight = pd.DataFrame(long_stock_weight)
        long_stock_weight.index = pd.to_datetime(upbeta.index)
        short_stock_weight = pd.DataFrame(short_stock_weight)
        short_stock_weight.index = pd.to_datetime(upbeta.index)
            # long_stock_weight = (3*momentum_weight -1).apply(lambda x: x/(1-beta_need_short_hedge_from_long_stock))
            # short_stock_weight = (1-(3*momentum_weight -1).apply(lambda x: x/(1-beta_need_long_hedge_from_short_stock)))
            # long_stock_weight = long_stock_weight.fillna(0)
            # short_stock_weight = short_stock_weight.fillna(0)

        weight_result = pd.concat([long_stock_weight, short_stock_weight, etf_weight], axis=1)
        weight_result.columns = ['long_stock_weight','short_stock_weight','etf_weight']
        weight_result.index = pd.to_datetime(upbeta.index)
        # long_stock.index = pd.to_datetime(long_stock.index)
        # short_stock.index = pd.to_datetime(short_stock.index)

        self.weight = weight_result.copy()
        self.long_stock = long_stock.copy()
        self.short_stock = short_stock.copy()

        self.long_weight = self.long_stock.apply(lambda x: x/x.sum(), axis=1)
        for idx in tqdm(range(len(self.weight.index)), desc="Calculating long_weight..."):
            self.long_weight.iloc[idx,:] = self.long_weight.iloc[idx,:] * self.weight.iloc[idx,0]
        self.long_weight = self.long_weight.fillna(0)

        self.short_weight = self.short_stock.apply(lambda x: x/x.sum(), axis=1)
        for idx in tqdm(range(len(self.weight.index)), desc="Calculating short_weight..."):
            self.short_weight.iloc[idx,:] = self.short_weight.iloc[idx,:] * self.weight.iloc[idx,1]
        self.short_weight = self.short_weight.fillna(0)
        
        self.etf_weight = pd.DataFrame(self.weight['etf_weight'])
        self.etf_weight = self.etf_weight.fillna(0)
        self.etf_weight.columns = ['ETF']
        # self.etf_weight.index = pd.to_datetime(self.etf_weight.index)
        
        self.total_weight = pd.concat([(self.long_weight + self.short_weight), self.etf_weight], axis=1)
        self.total_weight = self.total_weight.shift(1).fillna(0)

        print("Straddle making done! check the result with")
        print(" .weight, .long_stock, .short_stock, .long_weight, .short_weight")
        print(" .total_weight")
        print()
        print("### Notice ### .total_weight is lagged 1 day for backtesting to avoid forward-looking bias")

def momentum(pf_value, lookback, rebalance=None, method='long_short', hurdle=0.8):
    if rebalance != None:
        pf_return = pf_value.resample(rebalance).pct_change()
    else:
        pf_return = pf_value.pct_change()
    pf_return = pf_return.pct_change()
    pf_return = pf_return.fillna(0)

    long_weight = [0] * lookback
    short_weight = [0] * lookback
    for i in tqdm(range(lookback, len(pf_return)), desc="Calculating Momentum..."):
        long_weight.append( np.sum(pf_return.values[i-lookback:i] > 0) / lookback )
        short_weight.append( np.sum(pf_return.values[i-lookback:i] < 0) / lookback ) 

    if method == 'proportional':
        weight = long_weight
    elif method == 'inverse':
        weight = short_weight
    elif method == "gap_pos_neg":
        weight = np.array([max(0, x) for x in np.array(long_weight)*2 - np.array(short_weight)])
    elif method == 'hurdle_strict':
        weight = np.array([1 if w >= hurdle else 0 for w in long_weight])
    elif method == 'hurdle_smooth':
        weight = np.array([1 if w >= hurdle else w for w in long_weight])
    elif method == "off":
        weight = [1] * len(pf_value)
    elif method == "full_short":
        weight = [-1] * len(pf_value)
    else:
        raise Exception("Arguement for Parameter |method| must be one in ['proportional', 'inverse', 'gap_pos_neg', 'hurdle_strict', 'hurdle_smooth']")

    weight = pd.DataFrame(weight, columns=['weight'])
    weight.index = pd.to_datetime(pf_return.index)

    return weight

def backtesting_sketch(stock_return, etf_return, weight, resample):
    stock_return.index = pd.to_datetime(stock_return.index)
    etf_return.index = pd.to_datetime(etf_return.index)
    weight.index= pd.to_datetime(weight.index)

    if np.sum(stock_return.index == etf_return.index) != len(stock_return.index):
        raise Exception("Index of stock_return and weight is not same")
    elif np.sum(stock_return.index == etf_return.index) != len(stock_return.index):
        raise Exception("Index of stock_return and etf_return is not same")
    elif np.sum(weight.index == etf_return.index) != len(weight.index):
        raise Exception("Index of weight and etf_return is not same")

    nns_s = stock_return.resample(resample).mean()
    nns_mask_s = (nns_s.isna()).applymap(lambda x: np.NaN if x==True else float(1))

    stock_return = (1+stock_return).resample(resample).prod()
    stock_return = stock_return * nns_mask_s
    etf_return = (1+etf_return).resample(resample).prod()
    weight = weight.resample(resample).last().shift(-1)

    universe = pd.concat([stock_return-1, etf_return-1], axis=1)
    universe.columns = weight.columns
    pf_return = pd.DataFrame(((universe) * weight).sum(axis=1), columns=['pf_return'])
    pf_value = (1+pf_return).cumprod()
    pf_value.columns = ["pf_value"]

    return pf_return, pf_value

def drawdown(returns): # Calculate Drawdowns
    cumulative = returns
    highwatermark = cumulative.cummax()
    drawdown = (cumulative / highwatermark)-1
    return drawdown

def max_dd(returns): # Calculate Maxium Drawdowns
    return np.min(drawdown(returns))

def cagr(data): # Calculate CAGR
    years = data.index[-1].year - data.index[0].year
    return (data.iloc[-1] ** (1 / years)) -1
    
def sharpe(returns, days): # Calculate Sharpe Ratio
    return cagr((returns+1).cumprod()) / (np.std(returns) * np.sqrt(days))

def cum(returns): # Calculate Cummlative Returns from Returns
    return (returns+1).cumprod()

def weight_fill(raw, momentum): # Function for matching resampled data with another data
    raw.index = pd.to_datetime(raw.index)
    dummy = pd.DataFrame(pd.to_datetime(np.array(raw.index)), columns=['tmp'])
    dummy.index = raw.index
    dummy['tmp'] = pd.to_datetime(dummy.index)

    momentum.index = pd.to_datetime(momentum.index)
    momentum['tmp'] = pd.to_datetime(momentum.index)

    result = pd.merge(dummy, momentum, on='tmp', how='left')
    result.index = pd.to_datetime(result["tmp"])
    result = result.drop("tmp", axis=1)
    result = result.pad()
    return result