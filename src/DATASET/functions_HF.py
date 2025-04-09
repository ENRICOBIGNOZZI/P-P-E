from __future__ import annotations
import numpy as np
import pandas as pd
import math
from scipy.stats import entropy
from typing import List, Union

def volume_ask_mediato(average_size_ask: pd.Series, average_size_bid: pd.Series, steps: int) -> pd.Series:
    """
    Calcola il volume ask mediato.
    
    Args:
        average_size_ask (pd.Series): Serie con la dimensione media dell'ask.
        average_size_bid (pd.Series): Serie con la dimensione media del bid.
        steps (int): Numero di passi per la media mobile.
    
    Returns:
        pd.Series: Volume ask mediato.
    """
    return (average_size_ask.rolling(steps).mean()) / (np.mean(average_size_bid.rolling(steps).mean()) + (average_size_ask.rolling(steps).mean()))

def volume_bid_mediato(average_size_ask: pd.Series, average_size_bid: pd.Series, steps: int) -> pd.Series:
    """
    Calcola il volume bid mediato.
    
    Args:
        average_size_ask (pd.Series): Serie con la dimensione media dell'ask.
        average_size_bid (pd.Series): Serie con la dimensione media del bid.
        steps (int): Numero di passi per la media mobile.
    
    Returns:
        pd.Series: Volume bid mediato.
    """
    return (average_size_bid.rolling(steps).mean()) / ((average_size_ask.rolling(steps).mean()) + (average_size_ask.rolling(steps).mean()))

def volume_imbalance(average_size_ask: pd.Series, average_size_bid: pd.Series, steps: int) -> pd.Series:
    """
    Calcola lo sbilanciamento del volume.
    
    Args:
        average_size_ask (pd.Series): Serie con la dimensione media dell'ask.
        average_size_bid (pd.Series): Serie con la dimensione media del bid.
        steps (int): Numero di passi per la media mobile.
    
    Returns:
        pd.Series: Sbilanciamento del volume.
    """
    return ((average_size_ask.rolling(steps).mean()) - (average_size_bid.rolling(steps).mean())) / ((average_size_ask.rolling(steps).mean()) + (average_size_bid.rolling(steps).mean()))

def microprice(price_ask: pd.Series, price_bid: pd.Series, average_size_ask: pd.Series, average_size_bid: pd.Series, steps: int) -> pd.Series:
    """
    Calcola il microprezzo.
    
    Args:
        price_ask (pd.Series): Serie con i prezzi ask.
        price_bid (pd.Series): Serie con i prezzi bid.
        average_size_ask (pd.Series): Serie con la dimensione media dell'ask.
        average_size_bid (pd.Series): Serie con la dimensione media del bid.
        steps (int): Numero di passi per la media mobile.
    
    Returns:
        pd.Series: Microprezzo calcolato.
    """
    # Calcolo dei termini intermedi
    mean_price_bid = price_bid.rolling(steps).mean()
    mean_price_ask = price_ask.rolling(steps).mean()
    mean_size_bid = average_size_bid.rolling(steps).mean()
    mean_size_ask = average_size_ask.rolling(steps).mean()

    # Calcolo dei pesi
    weight_ask = mean_size_ask / (mean_size_bid + mean_size_ask)
    weight_bid = mean_size_bid / (mean_size_bid + mean_size_ask)

    # Calcolo finale
    P_micro = (mean_price_bid * weight_ask + mean_price_ask * weight_bid) / \
            (mean_price_bid * mean_size_ask + mean_price_ask * mean_size_bid)

    return P_micro

def cost(shares: pd.Series, price: pd.Series, steps: int) -> pd.Series:
    """
    Calcola il costo.
    
    Args:
        shares (pd.Series): Serie con il numero di azioni.
        price (pd.Series): Serie con i prezzi.
        steps (int): Numero di passi per lo shift.
    
    Returns:
        pd.Series: Costo calcolato.
    """
    return np.sign(price - price.shift(steps)) * (shares * price - shares.shift(steps) * price.shift(steps))

def calculate_vpin(buy_volume: pd.Series, sell_volume: pd.Series, window_size: int) -> pd.Series:
    """
    Calcola il Volume-Synchronized Probability of Informed Trading (VPIN).
    
    Args:
        buy_volume (pd.Series): Serie con i volumi di acquisto.
        sell_volume (pd.Series): Serie con i volumi di vendita.
        window_size (int): Dimensione della finestra per la media mobile.
    
    Returns:
        pd.Series: VPIN calcolato.
    """
    vpin = buy_volume.sub(sell_volume).abs().rolling(window_size).mean()
    vpin /= buy_volume + sell_volume
    return vpin

def price_imbalance(price_ask: pd.Series, price_bid: pd.Series, price_mid: pd.Series, steps: int) -> pd.Series:
    """
    Calcola lo sbilanciamento del prezzo.
    
    Args:
        price_ask (pd.Series): Serie con i prezzi ask.
        price_bid (pd.Series): Serie con i prezzi bid.
        price_mid (pd.Series): Serie con i prezzi medi.
        steps (int): Numero di passi per la media mobile.
    
    Returns:
        pd.Series: Sbilanciamento del prezzo calcolato.
    """
    return (price_ask.rolling(steps).max() + price_bid.rolling(steps).min()) / price_mid.rolling(steps).mean()

def autocorrelation(signal: pd.Series, steps: int) -> pd.Series:
    """
    Calcola l'autocorrelazione di una serie.
    
    Args:
        signal (pd.Series): Serie di input.
        steps (int): Numero di passi per la finestra mobile.
    
    Returns:
        pd.Series: Autocorrelazione calcolata.
    """
    autocorr = signal.rolling(window=steps).apply(lambda x: x.autocorr())
    return autocorr

def shannon_entropy(price: np.ndarray, steps: int) -> np.ndarray:
    """
    Calcola l'entropia di Shannon per una serie di prezzi.
    
    Args:
        price (np.ndarray): Array di prezzi.
        steps (int): Numero di passi per il calcolo dell'entropia.
    
    Returns:
        np.ndarray: Entropia di Shannon calcolata.
    """
    shannon_entropy = np.zeros(len(price))
    for i in range(steps, len(price), 1):
        price_changes = (price[i] - price[i-steps]) / price[i-steps:i]
        p, bins = np.histogram(price_changes, bins='auto', density=True)
        shannon_entropy[i] = entropy(p)
    return shannon_entropy

def momentum_normalized(data: List[float], length: int) -> List[float]:
    """
    Calcola il Momentum Normalized Indicator con la trasformazione del cubo.
    
    Args:
        data (List[float]): Lista di dati dei prezzi.
        length (int): Periodo di lookback per il Momentum Indicator.
    
    Returns:
        List[float]: Momentum Normalized Indicator con la trasformazione del cubo.
    """
    momentum = []
    norm_mom = []

    for i, _ in enumerate(data):
        if i < length:
            momentum.append(0)
        else:
            momentum.append(data[i] - data[i - length])

    smoothed_momentum = super_smoother(momentum, length)

    for i, _ in enumerate(smoothed_momentum):
        if i > 1:
            norm_mom.append(smoothed_momentum[i] - smoothed_momentum[i - 1])
        else:
            norm_mom.append(0)

    agc_norm_mom = agc(norm_mom)
    cube_anm = cube_transform(agc_norm_mom)

    return cube_anm

def agc(data: List[float]) -> List[float]:
    """
    Implementazione in Python dell'Automatic Gain Control creato da John Ehlers.
    
    Args:
        data (List[float]): Lista di dati dei prezzi.
    
    Returns:
        List[float]: Dati con l'Automatic Gain Control applicato.
    """
    real = []
    peak = []
    for i, _ in enumerate(data):
        if i < 1:
            real.append(0)
            peak.append(.0000001)
        else:
            peak.append(0.991 * peak[i - 1])
            if abs(data[i]) > peak[i]:
                peak[i] = abs(data[i])

            if peak[i] != 0:
                real.append(data[i] / peak[i])

    return real

def cube_transform(data: List[float]) -> List[float]:
    """
    Implementazione in Python della Cube Transform creata da John Ehlers.
    
    Args:
        data (List[float]): Lista di dati dei prezzi.
    
    Returns:
        List[float]: Dati con la Cube Transform applicata.
    """
    cube = []
    for i, _ in enumerate(data):
        c = data[i]**3
        cube.append(c)
    return cube

def volume_heat(data: List[float], ma_length: int) -> List[float]:
    data = np.array(data)
    vh = np.zeros(len(data))  # Inizializza l'array con zeri
    rolling_mean = np.convolve(data, np.ones(ma_length + 1) / (ma_length + 1), mode='valid')
    rolling_std = np.sqrt(np.convolve((data - np.mean(data))**2, np.ones(ma_length + 1), mode='valid') / (ma_length + 1))
    vh[ma_length:] = (data[ma_length:] - rolling_mean) / rolling_std > 1
    vh[ma_length:] = vh[ma_length:].astype(float)  # Converti booleano in float (0.0 o 1.0)
    return vh.tolist()

def custom_trendflex(data: List[float], length: int, s_length: int) -> List[float]:
    """
    Implementazione in Python dell'indicatore TrendFlex con lunghezza del SuperSmoother personalizzabile, creato da John Ehlers.
    
    Args:
        data (List[float]): Lista di dati dei prezzi.
        length (int): Periodo di lookback.
        s_length (int): Periodo del SuperSmoother.
    
    Returns:
        List[float]: Dati dell'indicatore TrendFlex.
    """
    ssf = super_smoother(data, s_length)

    tf = []
    ms = []
    sums = []
    for i, _ in enumerate(ssf):
        if i < length:
            tf.append(0)
            ms.append(0)
            sums.append(0)
        else:
            sum = 0
            for t in range(1, length + 1):
                sum = sum + ssf[i] - ssf[i - t]
            sum = sum / length
            sums.append(sum)

            ms.append(0.04 * sums[i] * sums[i] + 0.96 * ms[i - 1])
            if ms[i] != 0:
                tf.append(round(sums[i] / math.sqrt(ms[i]), 2))

    return tf

def ebsw(data: List[float], hp_length: int, ssf_length: int) -> List[float]:
    """
    Implementazione in Python dell'indicatore Even Better Sine Wave creato da John Ehlers.
    
    Args:
        data (List[float]): Lista di dati dei prezzi.
        hp_length (int): Periodo.
        ssf_length (int): Predict.
    
    Returns:
        List[float]: Dati dell'indicatore Even Better Sine Wave.
    """
    pi = 3.14159
    alpha1 = (1 - math.sin(2 * pi / hp_length)) / math.cos(2 * pi / hp_length)

    hpf = []

    for i, _ in enumerate(data):
        if i < hp_length:
            hpf.append(0)
        else:
            hpf.append((0.5 * (1 + alpha1) * (data[i] - data[i - 1])) + (alpha1 * hpf[i - 1]))

    ssf = super_smoother(hpf, ssf_length)

    wave = []
    for i, _ in enumerate(data):
        if i < ssf_length:
            wave.append(0)
        else:
            w = (ssf[i] + ssf[i - 1] + ssf[i - 2]) / 3
            p = (pow(ssf[i], 2) + pow(ssf[i - 1], 2) + pow(ssf[i - 2], 2)) / 3
            if p == 0:
                wave.append(0)
            else:
                wave.append(w / math.sqrt(p))
    return wave

def szladx(data: List[List[float]], length: int) -> List[float]:
    """
    Un upgrade a bassa latenza dell'indicatore ADX.
    
    Args:
        data (List[List[float]]): Lista di dati che include [high, low, close].
        length (int): Periodo di lookback per l'ADX.
    
    Returns:
        List[float]: Dati dell'indicatore Low Lag ADX.
    """
    lag = (length - 1) / 2
    ssf = []
    smoothed_true_range = []
    smoothed_directional_movement_plus = []
    smoothed_directional_movement_minus = []
    dxi = []
    szladxi = []

    for i, _ in enumerate(data):
        if i < round(lag):
            ssf.append(1)
            smoothed_true_range.append(1)
            smoothed_directional_movement_minus.append(1)
            smoothed_directional_movement_plus.append(1)
            dxi.append(1)
            szladxi.append(1)
        else:
            high = data[i][0]
            high1 = data[i-1][0]
            low = data[i][1]
            low1 = data[i-1][1]
            close1 = data[i-1][2]

            trng = max(max(high - low, abs(high - close1)), abs(low - close1))
            if high - high1 > low1 - low:
                directional_movement_plus = max(high - high1, 0)
            else:
                directional_movement_plus = 0

            if low1 - low > high - high1:
                directional_movement_minus = max(low1 - low, 0)
            else:
                directional_movement_minus = 0

            smoothed_true_range.append(smoothed_true_range[i-1] - (smoothed_true_range[i-1] / length) + trng)
            smoothed_directional_movement_plus.append(smoothed_directional_movement_plus[i-1] - (smoothed_directional_movement_plus[i - 1] / length) + directional_movement_plus)
            smoothed_directional_movement_minus.append(smoothed_directional_movement_minus[i-1] - (smoothed_directional_movement_minus[i - 1] / length) + directional_movement_minus)

            di_plus = smoothed_directional_movement_plus[i] / smoothed_true_range[i] * 100
            di_minus = smoothed_directional_movement_minus[i] / smoothed_true_range[i] * 100
            dxi.append(abs(di_plus - di_minus) / (di_plus + di_minus) * 100)

            szladxi.append(dxi[i] + (dxi[i] - dxi[i-round(lag)]))

    ssf = super_smoother(szladxi, 10)
    return ssf

def voss(data: List[float], period: int, predict: int, bandwith: float) -> List[float]:
    """
    Implementazione in Python dell'indicatore Voss creato da John Ehlers.
    
    Args:
        data (List[float]): Lista di dati dei prezzi.
        period (int): Periodo.
        predict (int): Predict.
        bandwith (float): Bandwith.
    
    Returns:
        List[float]: Dati dell'indicatore Voss.
    """
    voss = []
    filt = []
    vf = []

    pi = 3.14159

    order = 3 * predict
    f1 = math.cos(2 * pi / period)
    g1 = math.cos(bandwith * 2 * pi / period)
    s1 = 1 / g1 - math.sqrt(1 / (g1 * g1) - 1)

    for i, _ in enumerate(data):
        if i <= period or i <= 5 or i <= order:
            filt.append(0)
        else:
            filt.append(0.5 * (1 - s1) * (data[i] - data[i - 2]) + f1 * (1 + s1) * filt[i - 1] - s1 * filt[i - 2])

    for i, _ in enumerate(data):
        if i <= period or i <= 5 or i <= order:
            voss.append(0)
        else:
            sumc = 0
            for count in range(order):
                sumc = sumc + ((count + 1) / float(order)) * voss[i - (order - count)]

            voss.append(((3 + order) / 2) * filt[i] - sumc)

    for i, _ in enumerate(data):
        vf.append(voss[i] - filt[i])
    return vf

def roofing_filter(data: List[float], hp_length: int, ss_length: int) -> List[float]:
    """
    Implementazione in Python dell'indicatore Roofing Filter creato da John Ehlers.
    
    Args:
        data (List[float]): Lista di dati dei prezzi.
        hp_length (int): Lunghezza del filtro passa-alto.
        ss_length (int): Periodo per il Super Smoother.
    
    Returns:
        List[float]: Dati con il Roofing Filter applicato.
    """
    hpf = []

    for i, _ in enumerate(data):
        if i < 2:
            hpf.append(0)
        else:
            alpha_arg = 2 * 3.14159 / (hp_length * 1.414)
            alpha1 = (math.cos(alpha_arg) + math.sin(alpha_arg) - 1) / math.cos(alpha_arg)
            hpf.append(math.pow(1.0-alpha1/2.0, 2)*(data[i]-2*data[i-1]+data[i-2]) + 2*(1-alpha1)*hpf[i-1] - math.pow(1-alpha1, 2)*hpf[i-2])
    ss = super_smoother(hpf, ss_length)
    return ss

def super_smoother(data: List[float], length: int) -> List[float]:
    """
    Implementazione in Python dell'indicatore Super Smoother creato da John Ehlers.
    
    Args:
        data (List[float]): Lista di dati dei prezzi.
        length (int): Periodo.
    
    Returns:
        List[float]: Dati con il Super Smoother applicato.
    """
    ssf = []
    for i, _ in enumerate(data):
        if i < 2:
            ssf.append(0)
        else:
            arg = 1.414 * 3.14159 / length
            a_1 = math.exp(-arg)
            b_1 = 2 * a_1 * math.cos(4.44/float(length))
            c_2 = b_1
            c_3 = -a_1 * a_1
            c_1 = 1 - c_2 - c_3
            ssf.append(c_1 * (data[i] + data[i-1]) / 2 + c_2 * ssf[i-1] + c_3 * ssf[i-2])
    return ssf

def getYX(series: pd.Series, constant: str, lags: Union[int, List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola Y e X per l'analisi di regressione.
    
    Args:
        series (pd.Series): Serie di dati.
        constant (str): Tipo di costante da includere ('nc', 'ct', 'ctt').
        lags (Union[int, List[int]]): Numero di ritardi o lista di ritardi.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Y e X per l'analisi di regressione.
    """
    series_ = series.diff().dropna()
    x = lagDF(series_, lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0]-1:-1, 0]  # lagged level
    y = series_.iloc[-x.shape[0]:].values
    if constant != 'nc':
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        if constant[:2] == 'ct':
            trend = np.arange(x.shape[0]).reshape(-1, 1)
            x = np.append(x, trend, axis=1)
        if constant == 'ctt':
            x = np.append(x, trend**2, axis=1)
    return y, x

def lagDF(df0: pd.DataFrame, lags: Union[int, List[int]]) -> pd.DataFrame:
    """
    Crea un DataFrame con ritardi di una serie di dati.
    
    Args:
        df0 (pd.DataFrame): DataFrame di input.
        lags (Union[int, List[int]]): Numero di ritardi o lista di ritardi.
    
    Returns:
        pd.DataFrame: DataFrame con ritardi.
    """
    df1 = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(lags + 1)
    else:
        lags = [int(lag) for lag in lags]
    for lag in lags:
        df_ = df0.shift(lag).copy(deep=True)
        df_.columns = [str(i) + '_' + str(lag) for i in df_.columns]
        df1 = df1.join(df_, how='outer')
    return df1

def getBetas(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola i coefficienti e le varianze per l'analisi di regressione.
    
    Args:
        y (np.ndarray): Array Y.
        x (np.ndarray): Array X.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Coefficienti e varianze.
    """
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    xxinv = np.linalg.inv(xx)
    bMean = np.dot(xxinv, xy)
    err = y - np.dot(x, bMean)
    bVar = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xxinv
    return bMean, bVar

def get_bsadf(logP: pd.Series, minSL: int, constant: str, lags: Union[int, List[int]]) -> Dict[str, Union[pd.Timestamp, float]]:
    """
    Calcola il BSADF (Bounded Seasonal Augmented Dickey-Fuller) test.
    
    Args:
        logP (pd.Series): Serie di dati dei prezzi.
        minSL (int): Minimo numero di ritardi.
        constant (str): Tipo di costante da includere ('nc', 'ct', 'ctt').
        lags (Union[int, List[int]]): Numero di ritardi o lista di ritardi.
    
    Returns:
        Dict[str, Union[pd.Timestamp, float]]: Risultati del test BSADF.
    """
    y, x = getYX(logP, constant=constant, lags=lags)
    startPoints = range(0, y.shape[0] + lags - minSL + 1)
    bsadf = None
    allADF = []
    for start in startPoints:
        y_, x_ = y[start:], x[start:]
        bMean_, bStd_ = getBetas(y_, x_)
        bMean_, bStd_ = bMean_[0, 0], bStd_[0, 0]**.5
        allADF.append(bMean_ / bStd_)
        if bsadf is None or allADF[-1] > bsadf:
            bsadf = allADF[-1]
    out = {'Time': logP.index[-1], 'gsadf': bsadf}
    return out