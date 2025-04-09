from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List
#import pandas_ta as ta
from ta.momentum import RSIIndicator
from ta.trend import DPOIndicator
from ta.volatility import UlcerIndex
from ta.volume import ChaikinMoneyFlowIndicator
def cusum(g_raw: pd.Series, h: float) -> List[int]:
    diff = g_raw.diff().to_numpy()
    n = len(diff)
    
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    t_events = np.zeros(n, dtype=int)

    for i in range(1, n):
        s_pos[i] = max(0, s_pos[i-1] + max(0, diff[i]))
        s_neg[i] = min(0, s_neg[i-1] + min(0, diff[i]))

        if s_pos[i] > h:
            t_events[i] = 1
            s_pos[i] = 0
        elif s_neg[i] < -h:
            t_events[i] = -1
            s_neg[i] = 0

    return t_events.tolist()
def calculate_rsi(prices: pd.Series, window: int) -> pd.Series:
    """
    Calcola l'RSI (Relative Strength Index) per una serie di prezzi.

    Args:
        prices (pd.Series): Serie di prezzi.
        window (int): Dimensione della finestra per il calcolo dell'RSI.

    Returns:
        pd.Series: Serie con i valori dell'RSI calcolati.
    """
    # Calcola la variazione del prezzo
    delta = prices.pct_change().dropna()

    # Calcola le variazioni positive e negative
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    # Calcola il Relative Strength (RS)
    rs = gain / loss

    # Calcola l'RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_rsi(price: pd.Series, windows: List[int]) -> pd.DataFrame:
    """
    Calcola l'RSI (Relative Strength Index) per diverse finestre temporali.

    Args:
        price (pd.Series): Serie di prezzi.
        windows (List[int]): Lista di dimensioni delle finestre per calcolare l'RSI.

    Returns:
        pd.DataFrame: DataFrame con i valori dell'RSI calcolati.
    """
    # Controlla se price è una Serie Pandas
    if not isinstance(price, pd.Series):
        raise ValueError("Il parametro 'price' deve essere una Serie Pandas")
    df = pd.DataFrame(index=price.index)  # Usa lo stesso indice di price
    for i, w in enumerate(windows):
        df[f'rsi_{w}'] = calculate_rsi(price, w)
    return df

def get_rsi_decision(close: pd.Series, windows: List[int]) -> pd.DataFrame:
    """
    Calcola i segnali di decisione basati sull'RSI per le finestre specificate.

    Args:
        close (pd.Series): Serie temporale dei prezzi di chiusura.
        windows (List[int]): Lista di dimensioni delle finestre per calcolare l'RSI.

    Returns:
        pd.DataFrame: DataFrame con segnali di decisione RSI.
    """
    # Calcola l'RSI per le finestre specificate
    rsi = get_rsi(close, windows=windows)

    # DataFrame per memorizzare le decisioni
    decisions = pd.DataFrame(index=close.index)

    for w in windows:
        # Genera i segnali di decisione basati sull'RSI
        rsi_col = f'rsi_{w}'
        rsi_data = rsi[rsi_col]

        # Inizializza una Serie vuota con valori NaN per evitare problemi
        rsi_decision = pd.Series(index=rsi_data.index, dtype='float64')

        # Calcola le condizioni di overbought, oversold, up_trend e down_trend
        overbought = rsi_data > 70
        oversold = rsi_data < 30

        # Rimuove i valori NaN per evitare warning con pct_change
        rsi_data = rsi_data.dropna()

        # Calcola le variazioni percentuali con fill_method=None per evitare il default pad
        pct_change = rsi_data.pct_change(fill_method=None)

        # Calcola up_trend e down_trend usando pct_change senza riempire NaN
        up_trend = (~overbought & ~oversold & (pct_change >= 0))
        down_trend = (~overbought & ~oversold & (pct_change < 0))

        # Assegna i valori di decisione
        rsi_decision.loc[overbought | down_trend] = -1
        rsi_decision.loc[oversold | up_trend] = 1

        # Usa `fillna` per gestire i NaN e assicurarsi che tutti i valori siano assegnati
        rsi_decision = rsi_decision.fillna(0)  # Imposta i valori mancanti a 0

        # Aggiungi la Serie di decisioni al DataFrame delle decisioni
        decisions[f'rsid_{w}'] = rsi_decision

    return decisions

def mom_std(df: pd.DataFrame, windows_mom: List[int], windows_std: List[int]) -> pd.DataFrame:
    """
    Calcola il momentum e la deviazione standard per diverse finestre temporali.

    Args:
        df (pd.DataFrame): DataFrame di input.
        windows_mom (List[int]): Lista di finestre temporali per il momentum.
        windows_std (List[int]): Lista di finestre temporali per la deviazione standard.

    Returns:
        pd.DataFrame: DataFrame con momentum e deviazione standard calcolati.
    """
    mkt = df.copy()

    for i in windows_mom:
        mkt = mkt.join(df.volume.pct_change(i).rename(f'vol_mom_{i}'))
        mkt = mkt.join(df.close.pct_change(i).rename(f'mom_{i}'))

    for i in windows_std:
        mkt = mkt.join(df.close.pct_change().rolling(i).std().rename(f'std_{i}'))
        mkt = mkt.join(df.volume.pct_change().rolling(i).std().rename(f'vol_std_{i}'))
    mkt = mkt.drop(columns=df.columns)
    return mkt

def get_my_ta_windows(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Calcola indicatori tecnici per diverse finestre temporali.

    Args:
        df (pd.DataFrame): DataFrame di input.
        windows (List[int]): Lista di finestre temporali.

    Returns:
        pd.DataFrame: DataFrame con indicatori tecnici calcolati.
    """
    
    TA = []
    for mt in windows:

        TA.append(get_my_ta(df, window=mt))
    TA = pd.concat(TA, axis=1)
    TA = TA.loc[:, ~TA.columns.duplicated()]
    TA = TA.reindex(sorted(TA.columns), axis=1)
    return TA

def get_my_ta(
    df_: pd.DataFrame,
    fillna: bool = False,
    window: int = 1
) -> pd.DataFrame:
    """
    Aggiunge indicatori tecnici di volume al DataFrame.

    Args:
        df_ (pd.DataFrame): DataFrame di input.
        fillna (bool): Se True, riempie i valori NaN. Default è False.
        window (int): Dimensione della finestra per gli indicatori. Default è 15.

    Returns:
        pd.DataFrame: DataFrame con nuovi indicatori tecnici.
    """
    df = ohlcv(df_)
    high = 'high'
    low = 'low'
    close = 'close'
    volume = 'volume'

    # Chaikin Money Flow
    df[f"volume_cmf_{window}"] = ChaikinMoneyFlowIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], window=window,
        fillna=fillna
    ).chaikin_money_flow()

    # Ulcer Index
    df[f"volatility_ui_{window}"] = UlcerIndex(
        close=df[close], window=window, fillna=fillna
    ).ulcer_index()

    # DPO Indicator
    df[f"trend_dpo_{window}"] = DPOIndicator(
        close=df[close], window=window, fillna=fillna
    ).dpo()

    # Relative Strength Index (RSI)
    df[f"momentum_rsi_{window}"] = RSIIndicator(
        close=df[close], window=window, fillna=fillna
    ).rsi()

    df = df.drop(columns=['open', 'high', 'low', 'close', 'volume'])
    return df
def ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte le colonne del DataFrame in formato OHLCV.
    
    Args:
        df (pd.DataFrame): DataFrame di input.
    
    Returns:
        pd.DataFrame: DataFrame con colonne OHLCV.
    """
    return pd.DataFrame([df.open,df.high,df.low,df.close,df.volume]).T


def get_roll_measure(close_prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calcola la misura di Roll per stimare lo spread bid-ask effettivo.

    Args:
        close_prices (pd.Series): Serie dei prezzi di chiusura.
        window (int): Finestra di stima. Default è 20.

    Returns:
        pd.Series: Misura di Roll calcolata.
    """
    price_diff = close_prices.pct_change()
    price_diff_lag = price_diff.shift(window)
    return 2 * np.sqrt(abs(price_diff.rolling(window=window).cov(price_diff_lag)))
def get_roll_measure_special(close_prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calcola la misura di Roll per stimare lo spread bid-ask effettivo.

    Args:
        close_prices (pd.Series): Serie dei prezzi di chiusura.
        window (int): Finestra di stima. Default è 20.

    Returns:
        pd.Series: Misura di Roll calcolata.
    """
    price_diff = close_prices.diff()
    price_diff_lag = price_diff.shift(1)
    return 2 * np.sqrt(abs(price_diff.rolling(window=window).cov(price_diff_lag)))

def get_roll_impact(close_prices: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Calcola l'impatto di Roll.

    Derivata dalla misura di Roll che tiene conto del volume di dollari scambiati.

    Args:
        close_prices (pd.Series): Serie dei prezzi di chiusura.
        dollar_volume (pd.Series): Serie dei volumi in dollari.
        window (int): Finestra di stima. Default è 20.

    Returns:
        pd.Series: Impatto di Roll calcolato.
    """
    roll_measure = get_roll_measure(close_prices, window)
    return roll_measure / (dollar_volume + 1)
def _get_gamma(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Calcola la stima di gamma dall'algoritmo Corwin-Schultz.

    Args:
        high (pd.Series): Serie dei prezzi massimi.
        low (pd.Series): Serie dei prezzi minimi.

    Returns:
        pd.Series: Stime di gamma.
    """
    high_max = high.rolling(window=2).max()
    low_min = low.rolling(window=2).min()
    gamma = np.log(high_max / low_min) ** 2
    return gamma

def _get_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """
    Calcola alpha dall'algoritmo Corwin-Schultz.

    Args:
        beta (pd.Series): Stime di beta.
        gamma (pd.Series): Stime di gamma.

    Returns:
        pd.Series: Valori di alpha.
    """
    den = 3 - 2 * 2 ** .5
    alpha = (2 ** .5 - 1) * (beta ** .5) / den
    alpha -= (gamma / den) ** .5
    alpha[alpha < 0] = 0  # Imposta gli alpha negativi a 0 (vedi p.727 del documento)
    return alpha

def get_corwin_schultz_estimator(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Calcola lo stimatore di Corwin-Schultz utilizzando i prezzi high-low.

    Args:
        high (pd.Series): Serie dei prezzi massimi.
        low (pd.Series): Serie dei prezzi minimi.
        window (int): Finestra di stima. Default è 20.

    Returns:
        pd.Series: Stimatori di Corwin-Schultz.
    """
    # Nota: S<0 iif alpha<0
    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low)
    alpha = _get_alpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    start_time = pd.Series(high.index[0:spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, start_time], axis=1)
    spread.columns = ['Spread', 'Start_Time']  # 1st loc used to compute beta
    return spread.Spread
def _get_beta(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Calcola la stima di beta dall'algoritmo Corwin-Schultz.

    Args:
        high (pd.Series): Serie dei prezzi massimi.
        low (pd.Series): Serie dei prezzi minimi.
        window (int): Finestra di stima.

    Returns:
        pd.Series: Stime di beta.
    """
    ret = np.log(high / low)
    high_low_ret = ret ** 2
    beta = high_low_ret.rolling(window=2).sum()
    beta = beta.rolling(window=window).mean()
    return beta
def get_bekker_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Calcola la volatilità di Bekker-Parkinson utilizzando beta e gamma nell'algoritmo Corwin-Schultz.

    Args:
        high (pd.Series): Serie dei prezzi massimi.
        low (pd.Series): Serie dei prezzi minimi.
        window (int): Finestra di stima. Default è 20.

    Returns:
        pd.Series: Stime della volatilità di Bekker-Parkinson.
    """
    # pylint: disable=invalid-name
    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low)

    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2 ** .5
    sigma = (2 ** -0.5 - 1) * beta ** 0.5 / (k2 * den)
    sigma += (gamma / (k2 ** 2 * den)) ** 0.5
    sigma[sigma < 0] = 0
    return sigma
def get_bar_based_kyle_lambda(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Calcola il lambda di Kyle dai dati delle barre.

    Args:
        close (pd.Series): Prezzi di chiusura.
        volume (pd.Series): Volume delle barre.
        window (int): Finestra mobile usata per la stima. Default è 20.

    Returns:
        pd.Series: Lambda di Kyle.
    """
    close_diff = close.pct_change()
    close_diff_sign = close_diff.apply(np.sign)
    close_diff_sign[close_diff_sign == 0] = np.nan
    close_diff_sign.ffill(inplace=True)  # Usa forward fill per sostituire i valori NaN
    volume_mult_trade_signs = volume * close_diff_sign  # bt * Vt
    return (close_diff / volume_mult_trade_signs).rolling(window=window).mean()
def get_bar_based_amihud_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Calcola il lambda di Amihud dai dati delle barre.

    Args:
        close (pd.Series): Prezzi di chiusura.
        dollar_volume (pd.Series): Volumi in dollari.
        window (int): Finestra mobile usata per la stima. Default è 20.

    Returns:
        pd.Series: Lambda di Amihud.
    """
    returns_abs =close.pct_change().abs()
    return (returns_abs / dollar_volume).rolling(window=window).mean()
def get_bar_based_hasbrouck_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Calcola il lambda di Hasbrouck dai dati delle barre.

    Args:
        close (pd.Series): Prezzi di chiusura.
        dollar_volume (pd.Series): Volumi in dollari.
        window (int): Finestra mobile usata per la stima. Default è 20.

    Returns:
        pd.Series: Lambda di Hasbrouck.
    """
    log_ret = close.pct_change().dropna().values
    log_ret_sign = log_ret.apply(np.sign)
    log_ret_sign[log_ret_sign == 0] = np.nan
    log_ret_sign.ffill(inplace=True)  # Usa forward fill per sostituire i valori NaN
    signed_dollar_volume_sqrt = log_ret_sign * np.sqrt(dollar_volume)
    return (log_ret / signed_dollar_volume_sqrt).rolling(window=window).mean()




























































































