from __future__ import annotations
import numpy as np
import pandas as pd
import math
from scipy import stats
import pywt
import copy

#import pandas_ta as ta
from statsmodels.tsa.stattools import adfuller
from ta.momentum import (
    AwesomeOscillatorIndicator,
    PercentagePriceOscillator,
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
    TSIIndicator,
    WilliamsRIndicator,
)
from ta.trend import (
    MACD,
    DPOIndicator,
    MassIndex,
    TRIXIndicator,
    VortexIndicator,
)
from ta.volatility import (
    DonchianChannel,
    KeltnerChannel,
)
from ta.volume import (
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    EaseOfMovementIndicator,
    ForceIndexIndicator,
)
from typing import List, Tuple
import ta
import pandas_ta as ta2
'''def ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte le colonne del DataFrame in formato OHLCV.
    
    Args:
        df (pd.DataFrame): DataFrame di input.
    
    Returns:
        pd.DataFrame: DataFrame con colonne OHLCV.
    """
    df.columns = [i.lower() for i in df.columns]
    close = pd.to_numeric(df.close)
    open = pd.to_numeric(df.open)
    high = pd.to_numeric(df.high)
    low = pd.to_numeric(df.low)
    volume = pd.to_numeric(df.volume)
    df_ohlcv = pd.DataFrame([open,high,low,close,volume]).T
    return df_ohlcv'''
def ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte le colonne del DataFrame in formato OHLCV.
    
    Args:
        df (pd.DataFrame): DataFrame di input.
    
    Returns:
        pd.DataFrame: DataFrame con colonne OHLCV.
    """
    return pd.DataFrame([df.open,df.high,df.low,df.close,df.volume]).T

def ta_stationary_test(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Esegue un test di stazionarietà su tutti gli indicatori tecnici.
    
    Args:
        df_ (pd.DataFrame): DataFrame di input.
    
    Returns:
        pd.DataFrame: DataFrame con i risultati del test di stazionarietà.
    """
    df = ohlcv(df_)
    

    all_ta = ta.add_all_ta_features(df,'open','high','low','close','volume')
    result_df = pd.DataFrame()
    for i in all_ta.columns:
        adf = adfuller(all_ta[i].dropna(),autolag='AIC')
        result_df['{}'.format(i)] = pd.Series(adf[0:2],index=['Test Statistic','p_value'])
    return result_df.T

def linear_regression_slope(series: pd.Series) -> float:
    x = np.arange(len(series))
    y = series.values
    x_mean = x.mean()
    y_mean = y.mean()
    slope = np.dot(x - x_mean, y - y_mean) / np.dot(x - x_mean, x - x_mean)
    return slope

def get_all_stationary_ta(df_: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Ottiene tutti gli indicatori tecnici stazionari.
    
    Args:
        df_ (pd.DataFrame): DataFrame di input.
        threshold (float, optional): Soglia per il p-value. Default è 0.05.
    
    Returns:
        pd.DataFrame: DataFrame con gli indicatori tecnici stazionari.
    """
    df = ohlcv(df_)
    all_ta = ta.add_all_ta_features(df,'open','high','low','close','volume')
    result_df = pd.DataFrame()
    for i in all_ta.columns:
        adf = adfuller(all_ta[i].dropna(),autolag='AIC')
        result_df['{}'.format(i)] = pd.Series(adf[0:2],index=['Test Statistic','p_value'])
    result = result_df.T
    all_ta = all_ta[result.loc[result.p_value<threshold].index]
    return all_ta



def get_stationary_ta_window_0(
    df_: pd.DataFrame,
    fillna: bool = False,
    mt: int = 10
) -> pd.DataFrame:
    """
    Aggiunge indicatori tecnici di volume al DataFrame.
    
    Args:
        df_ (pd.DataFrame): DataFrame di input.
        fillna (bool, optional): Se True, riempie i valori NaN. Default è False.
        mt (int, optional): Moltiplicatore per le finestre. Default è 10.
    
    Returns:
        pd.DataFrame: DataFrame con nuovi indicatori.
    """
    df = ohlcv(df_)
    high='high';low='low';close='close';volume='volume'
    # Accumulation Distribution Index
    df["volume_adi"] = AccDistIndexIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume]+1, fillna=fillna
    ).acc_dist_index()
    df["volume_adi"] = df["volume_adi"]/df['close']
    # Chaikin Money Flow
    df["volume_cmf_{}".format(20*mt)] = ChaikinMoneyFlowIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume]+1, window=mt,
        fillna=fillna
    ).chaikin_money_flow()
    # Force Index
    df["volume_fi_{}".format(15*mt)] = ForceIndexIndicator(
        close=df[close], volume=df[volume]+1, window=mt, fillna=fillna
    ).force_index()
    # Ease of Movement
    indicator_eom = EaseOfMovementIndicator(
        high=df[high], low=df[low], volume=df[volume], window=mt, fillna=fillna
    )
    df["volume_em"] = indicator_eom.ease_of_movement()
    df["volume_sma_em_{}".format(mt)] = indicator_eom.sma_ease_of_movement()

    # Keltner Channel
    indicator_kc = KeltnerChannel(
        close=df[close], high=df[high], low=df[low], window=mt, fillna=fillna
    )
    df["volatility_kcp_{}".format(10*mt)] = indicator_kc.keltner_channel_pband()

    # Donchian Channel
    indicator_dc = DonchianChannel(
        high=df[high], low=df[low], close=df[close], window=mt, offset=0, fillna=fillna
    )
    df["volatility_dcp_{}".format(20*mt)] = indicator_dc.donchian_channel_pband()

    # MACD
    indicator_macd = MACD(
        close=df[close], window_slow=2*mt, window_fast=mt, window_sign=9, fillna=fillna
    )
    df["trend_macd_{}_{}_{}".format(2*mt,1*mt,9)] = indicator_macd.macd()
    df["trend_macd_signal_{}_{}_{}".format(2*mt,1*mt,9)] = indicator_macd.macd_signal()
    df["trend_macd_diff_{}_{}_{}".format(2*mt,mt,9)] = indicator_macd.macd_diff()

    # Vortex Indicator
    indicator_vortex = VortexIndicator(
        high=df[high], low=df[low], close=df[close], window=mt, fillna=fillna
    )
    df["trend_vortex_ind_diff_{}".format(mt)] = indicator_vortex.vortex_indicator_diff()

    # TRIX Indicator
    df["trend_trix_{}".format(mt)] = TRIXIndicator(
        close=df[close], window=mt, fillna=fillna
    ).trix()

    # Mass Index
    df["trend_mass_index_{}_{}".format(1*mt,2*mt)] = MassIndex(
        high=df[high], low=df[low], window_fast=mt, window_slow=2*mt, fillna=fillna
    ).mass_index()

    # DPO Indicator
    df["trend_dpo_{}".format(2*mt)] = DPOIndicator(
        close=df[close], window=2*mt, fillna=fillna
    ).dpo()

    # Relative Strength Index (RSI)
    df["momentum_rsi_{}".format(1*mt)] = RSIIndicator(
        close=df[close], window=1*mt, fillna=fillna
    ).rsi()

    # TSI Indicator
    df["momentum_tsi_{}_{}".format(2*mt,1*mt)] = TSIIndicator(
        close=df[close], window_slow=2*mt, window_fast=1*mt, fillna=fillna
    ).tsi()

    # Stoch Indicator
    indicator_so = StochasticOscillator(
        high=df[high],
        low=df[low],
        close=df[close],
        window=mt,
        smooth_window=3,
        fillna=fillna,
    )
    df["momentum_stoch_{}".format(1*mt)] = indicator_so.stoch()
    df["momentum_stoch_signal_{}".format(1*mt)] = indicator_so.stoch_signal()

    # Williams R Indicator
    df["momentum_wr_{}".format(1*mt)] = WilliamsRIndicator(
        high=df[high], low=df[low], close=df[close], lbp=1*mt, fillna=fillna
    ).williams_r()

    # Awesome Oscillator
    df["momentum_ao_{}_{}".format(mt,2*mt)] = AwesomeOscillatorIndicator(
        high=df[high], low=df[low], window1=mt, window2=2*mt, fillna=fillna
    ).awesome_oscillator()

    # Rate Of Change
    df["momentum_roc_{}".format(1*mt)] = ROCIndicator(
        close=df[close], window=1*mt, fillna=fillna
    ).roc()

    # Percentage Price Oscillator
    indicator_ppo = PercentagePriceOscillator(
        close=df[close], window_slow=2*mt, window_fast=1*mt, window_sign=9, fillna=fillna
    )
    df["momentum_ppo_{}_{}_{}".format(2*mt,1*mt,9)] = indicator_ppo.ppo()
    df["momentum_ppo_signal_{}_{}_{}".format(2*mt,1*mt,9)] = indicator_ppo.ppo_signal()
    df["momentum_ppo_hist_{}_{}_{}".format(2*mt,1*mt,9)] = indicator_ppo.ppo_hist()

    df = df.drop(columns=['open','high','low','close','volume'])
    return df

def get_stationary_ta_windows(df: pd.DataFrame, mts: List[int]) -> pd.DataFrame:
    """
    Ottiene indicatori tecnici stazionari per diverse finestre temporali.
    
    Args:
        df (pd.DataFrame): DataFrame di input.
        mts (List[int]): Lista di finestre temporali.
    
    Returns:
        pd.DataFrame: DataFrame con indicatori tecnici stazionari.
    """
    TA = []
    for mt in mts:
        TA.append(get_stationary_ta_window_0(df, mt=mt))
    TA = pd.concat(TA, axis=1)
    TA = TA.loc[:, ~TA.columns.duplicated()]
    TA = TA.reindex(sorted(TA.columns), axis=1)
    return TA

def get_stationary_ta_windows_volume(df: pd.DataFrame, mts: List[int]) -> pd.DataFrame:
    """
    Ottiene indicatori tecnici stazionari per diverse finestre temporali.
    
    Args:
        df (pd.DataFrame): DataFrame di input.
        mts (List[int]): Lista di finestre temporali.
    
    Returns:
        pd.DataFrame: DataFrame con indicatori tecnici stazionari.
    """
    TA = []
    for mt in mts:
        TA.append(get_stationary_ta_window_0(df, mt=mt))
    TA = pd.concat(TA, axis=1)
    TA = TA.loc[:, ~TA.columns.duplicated()]
    TA = TA.reindex(sorted(TA.columns), axis=1)
    return TA

def filter_bank(index_list: np.ndarray, wavefunc: str = 'db4', lv: int = 4, m: int = 1, n: int = 4, plot: bool = False) -> List[np.ndarray]:
    """
    Applica la trasformata wavelet per la denoising dei dati.
    
    Args:
        index_list (np.ndarray): Sequenza di input.
        wavefunc (str, optional): Funzione wavelet. Default è 'db4'.
        lv (int, optional): Livello di decomposizione. Default è 4.
        m (int, optional): Livello di inizio per il processo di soglia. Default è 1.
        n (int, optional): Livello di fine per il processo di soglia. Default è 4.
        plot (bool, optional): Se True, genera grafici. Default è False.
    
    Returns:
        List[np.ndarray]: Lista di coefficienti wavelet denoised.
    """
    # Decomposing
    coeff = pywt.wavedec(index_list, wavefunc, mode='sym', level=lv)   #  Decomposing by levels，cD is the details coefficient
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # sgn function
    
    for i in range(m, n+1):   #  Select m~n Levels of the wavelet coefficients，and no need to dispose the cA coefficients(approximation coefficients)
        cD = coeff[i]
        Tr = np.sqrt(2*np.log2(len(cD)))  # Compute Threshold
        for j in range(len(cD)):
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) * (np.abs(cD[j]) -  Tr)  # Shrink to zero
            else:
                coeff[i][j] = 0   # Set to zero if smaller than threshold
    # Reconstructing
    coeffs = {}
    for i in range(len(coeff)):
        coeffs[i] = copy.deepcopy(coeff)
        for j in range(len(coeff)):
            if j != i:
                coeffs[i][j] = np.zeros_like(coeff[j])

    for i in range(len(coeff)):
        coeff[i] = pywt.waverec(coeffs[i], wavefunc)
        if len(coeff[i]) > len(index_list):
            coeff[i] = coeff[i][:-1]

    if plot:
        denoised_index = np.sum(coeff, axis=0)
        data = pd.DataFrame({'CLOSE': index_list, 'denoised': denoised_index})
        data.plot(figsize=(10,10),subplots=(2,1))
        data.plot(figsize=(10,5))

    return coeff

def get_estimator(price_data: pd.DataFrame, window: int = 30, trading_periods: int = 252, clean: bool = True) -> pd.Series:
    """
    Calcola la volatilità implicita.
    
    Args:
        price_data (pd.DataFrame): DataFrame dei prezzi.
        window (int, optional): Finestra di calcolo. Default è 30.
        trading_periods (int, optional): Periodi di trading. Default è 252.
        clean (bool, optional): Se True, rimuove i valori NaN. Default è True.
    
    Returns:
        pd.Series: Volatilità implicita calcolata.
    """
    log_ho = (price_data['high'] / price_data['open']).apply(np.log)
    log_lo = (price_data['low'] / price_data['open']).apply(np.log)
    log_co = (price_data['close'] / price_data['open']).apply(np.log)

    log_oc = (price_data['open'] / price_data['close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2

    log_cc = (price_data['close'] / price_data['close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(trading_periods)

    if clean:
        return result.dropna()
    else:
        return result

def vsa_indicator(data: pd.DataFrame, norm_lookback: int = 168) -> pd.Series:
    """
    Calcola l'indicatore VSA (Volume Synchronized Probability of Informed Trading).
    
    Args:
        data (pd.DataFrame): DataFrame dei dati.
        norm_lookback (int, optional): Finestra di normalizzazione. Default è 168.
    
    Returns:
        pd.Series: Indicatore VSA calcolato.
    """
    # Norm lookback should be fairly large

    atr = ta2.atr(data['high'], data['low'], data['close'], norm_lookback)
    vol_med = data['volume'].rolling(norm_lookback).median()

    data['norm_range'] = (data['high'] - data['low']) / atr
    data['norm_volume'] = data['volume'] / vol_med

    norm_vol = data['norm_volume'].to_numpy()
    norm_range = data['norm_range'].to_numpy()

    range_dev = np.zeros(len(data))
    range_dev[:] = np.nan

    for i in range(norm_lookback * 2, len(data)):
        window = data.iloc[i - norm_lookback + 1: i+ 1]
        slope, intercept, r_val,_,_ = stats.linregress(window['norm_volume'], window['norm_range'])

        if slope <= 0.0 or r_val < 0.2:
            range_dev[i] = 0.0
            continue

        pred_range = intercept + slope * norm_vol[i]
        range_dev[i] = norm_range[i] - pred_range

    return pd.Series(range_dev, index=data.index)

'''def rolling_hurst(price_series: pd.Series, window: int, min_window: int = 10, max_window: int = 100, num_windows: int = 20, num_samples: int = 100) -> pd.Series:
    """
    Calcola l'esponente di Hurst su una finestra mobile.
    
    Args:
        price_series (pd.Series): Serie dei prezzi.
        window (int): Dimensione della finestra.
        min_window (int, optional): Finestra minima. Default è 10.
        max_window (int, optional): Finestra massima. Default è 100.
        num_windows (int, optional): Numero di finestre. Default è 20.
        num_samples (int, optional): Numero di campioni. Default è 100.
    
    Returns:
        pd.Series: Esponente di Hurst calcolato.
    """
    return price_series.rolling(window=window).apply(lambda x: hurst_fd(x, min_window, max_window, num_windows, num_samples)[0], raw=True)'''

'''def hurst_fd(price_series: np.ndarray, min_window: int = 10, max_window: int = 100, num_windows: int = 20, num_samples: int = 100) -> Tuple[float, float]:
    """
    Calcola l'esponente di Hurst e la dimensione frattale.
    
    Args:
        price_series (np.ndarray): Serie dei prezzi.
        min_window (int, optional): Finestra minima. Default è 10.
        max_window (int, optional): Finestra massima. Default è 100.
        num_windows (int, optional): Numero di finestre. Default è 20.
        num_samples (int, optional): Numero di campioni. Default è 100.
    
    Returns:
        Tuple[float, float]: Esponente di Hurst e dimensione frattale.
    """
    price_series = np.array(price_series, dtype=np.float32)
    log_returns = np.diff(np.log(price_series))
    log_returns = np.array(log_returns, dtype=np.float32)
    window_sizes = np.linspace(min_window, max_window, num_windows, dtype=np.int32)
    R = np.empty((num_windows, num_samples), dtype=np.float32)
    S = np.empty((num_windows, num_samples), dtype=np.float32)

    for i, w in enumerate(window_sizes):
        start = np.random.randint(0, len(log_returns) - w, num_samples)
        seq = log_returns[start[:, np.newaxis] + np.arange(w)]
        R[i] = np.max(seq, axis=1) - np.min(seq, axis=1)
        S[i] = np.std(seq, axis=1)

    R_mean = np.mean(R, axis=1)
    S_mean = np.mean(S, axis=1)
    R_S = R_mean / S_mean
    log_window_sizes = np.log(window_sizes, dtype=np.float32)
    log_R_S = np.log(R_S, dtype=np.float32)
    coeffs = np.polyfit(log_window_sizes, log_R_S, 1)
    hurst_exponent = coeffs[0]
    fractal_dimension = 2 - hurst_exponent
    return hurst_exponent, fractal_dimension'''