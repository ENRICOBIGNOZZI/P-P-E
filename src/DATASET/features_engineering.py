import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
from typing import Dict
from src.forecasting import rademacher_anti_serum_for_signals_ic
from src.DATASET.economic import create_dataframe
from src.DATASET.balance_sheet import balance_ticker_data
from src.DATASET.rating import analyze_rating
import src.DATASET.functions_LF as lf
import src.DATASET.functions_HF as hf
import src.DATASET.functions_TI as ti

def create_technical_indicators_all(raw_data) -> pd.DataFrame:
    """
    Crea gli indicatori tecnici e li restituisce in un DataFrame, eseguendo i calcoli in maniera sequenziale.
    """
    # Prepara il DataFrame di base
    df = raw_data
    # Incrementa volume e calcola segnali "up" e "down"
    df['volume'] += 1
    df['up'] = df['volume'].where(df['close'].diff() > 0, 1) + 1
    df['down'] = df['volume'].where(df['close'].diff() <= 0, 1) + 1

    new_columns = {}

    # Calcolo dei punti di cambiamento per diverse soglie
    def calculate_change_points(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        thresholds = np.array([0.01, 0.02, 0.03])
        cols = {}
        for value in thresholds:
            try:
                result = ti.cusum(df['close'], value)
                cols[f'change_points_{value}'] = result
            except Exception as exc:
                print(f"Errore per soglia={value}: {exc}")
        return cols

    # Calcolo degli indicatori tecnici
    def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
        windows = [5, 10, 20, 50, 100]
        indicators = {}
        for window in windows:
            try:
                # Si assume che ti.get_my_ta_windows ritorni un DataFrame e che si prenda la prima colonna
                ta_result = ti.get_my_ta_windows(df, [window]).iloc[:, 0]
                indicators[f'ta_{window}'] = ta_result
            except Exception as exc:
                print(f"Errore nel calcolo degli indicatori tecnici per la finestra={window}: {exc}")
        return indicators

    # Calcolo dell'RSI e delle decisioni RSI
    def calculate_rsi(df: pd.DataFrame) -> Dict[str, pd.Series]:
        windows = [5, 10, 20, 50, 100,200]
        rsi_indicators = {}
        rsi_decisions = {}
        for window in windows:
            try:
                rsi_result = ti.get_rsi(df['close'], [window])
                rsi_indicators.update(rsi_result)
            except Exception as exc:
                print(f"Errore nel calcolo dell'RSI per la finestra={window}: {exc}")
            try:
                decision_result = ti.get_rsi_decision(df['close'], [window])
                rsi_decisions.update(decision_result)
            except Exception as exc:
                print(f"Errore nel calcolo delle decisioni RSI per la finestra={window}: {exc}")
        return {**rsi_indicators, **rsi_decisions}

    # Calcolo della volatilità basata su log diff (mom_std)
    def calculate_log_diff_volatility(df: pd.DataFrame) -> pd.DataFrame:
        windows = [5, 10, 20, 50, 100]
        results = {}
        for window in windows:
            try:
                result = ti.mom_std(df, [window], [window])
                results[window] = result
            except Exception as exc:
                print(f"Errore nel calcolo della volatilità per la finestra={window}: {exc}")
        # Combina i risultati in un DataFrame finale
        df_mom_std = pd.concat(results.values(), axis=1)
        return df_mom_std

    # Calcolo delle medie mobili semplici ed esponenziali
    def calculate_moving_averages(df: pd.DataFrame) -> Dict[str, pd.Series]:
        windows = [5, 10, 20, 50, 100,200]
        cols = {}
        for window in windows:
            try:
                sma = (df['close'] - df['close'].rolling(window=window).mean()) / df['close']
                alpha = 2 / (window + 1)
                ema = (df['close'] - df['close'].ewm(alpha=alpha, adjust=False).mean()) / df['close']
                cols[f'sma_{window}'] = sma
                cols[f'ema_{window}'] = ema
            except Exception as exc:
                print(f"Errore per finestra={window}: {exc}")
        return cols

    # Aggiorna il dizionario con i risultati dei calcoli
    new_columns.update(calculate_change_points(df))
    new_columns.update(calculate_technical_indicators(df))
    new_columns.update(calculate_rsi(df))
    new_columns.update(calculate_log_diff_volatility(df))
    new_columns.update(calculate_moving_averages(df))

    # Assicuriamoci che tutti i valori in new_columns siano serie unidimensionali
    for key, value in list(new_columns.items()):
        if isinstance(value, pd.DataFrame):
            for col in value.columns:
                new_columns[f"{key}_{col}"] = value[col]
            del new_columns[key]
        elif isinstance(value, np.ndarray) and value.ndim > 1:
            for i in range(value.shape[1]):
                new_columns[f"{key}_{i}"] = value[:, i]
            del new_columns[key]

    # Crea un DataFrame con le nuove colonne e lo unisce a quello originale
    new_df = pd.DataFrame(new_columns, index=df.index)
    result_df = pd.concat([df, new_df], axis=1)

    # Se il check è attivo, esegue un'ulteriore analisi

    return result_df

def create_high_frequency_all(raw_data) -> pd.DataFrame:
    """
    Crea tutte le features ad alta frequenza in maniera sequenziale e le restituisce in un DataFrame.
    """
    # Prepara il DataFrame di base
    df = raw_data
    
    # Calcola i segnali "up" e "down"
    df['up'] = df['volume'].where(df['close'].diff() > 0, 1) + 1
    df['down'] = df['volume'].where(df['close'].diff() <= 0, 1) + 1
    
    # Definisci le finestre da considerare
    finestra = [1, 5, 10, 20, 50, 100,200]
    new_columns = {}

    # Per ciascuna finestra, calcola gli indicatori ad alta frequenza
    for steps in finestra:
        try:
            # Indicatori di base
            new_columns[f'ask_mediato_{steps}'] = hf.volume_ask_mediato(df['down'], df['up'], steps)
            new_columns[f'bid_mediato_{steps}'] = hf.volume_bid_mediato(df['down'], df['up'], steps)
            new_columns[f'volume_imbalance_{steps}'] = hf.volume_imbalance(df['down'], df['up'], steps)
            new_columns[f'microprice_{steps}'] = hf.microprice(df['high'], df['low'], df['down'], df['up'], steps)
            
            # Se steps >= 5, calcola ulteriori indicatori: momentum e volume heat
            if steps >= 5:
                new_columns[f'momentum_normalized_{steps}'] = hf.momentum_normalized(df['close'].values, steps)
                new_columns[f'volume_heat_{steps}'] = hf.volume_heat(df['close'].values, steps)
            
            # Se necessario, qui potresti anche calcolare altri indicatori (es. autocorrelazione)
            # Il codice originale era commentato, quindi lo manteniamo commentato:
            #
            # if 5 <= steps <= 20:
            #     try:
            #         autocorr = {}
            #         autocorr[f'volume_autocorrelation_{steps}'] = hf.autocorrelation(df['volume'], steps)
            #         autocorr[f'returns_autocorrelation_{steps}'] = hf.autocorrelation(df['close'].pct_change().fillna(0), steps)
            #         new_columns.update(autocorr)
            #     except Exception as exc:
            #         print(f"Errore nel calcolo dell'autocorrelazione per finestra={steps}: {exc}")
        except Exception as exc:
            print(f"Errore per finestra={steps}: {exc}")

    # Converte il dizionario in DataFrame (usando lo stesso indice di df)
    new_df = pd.DataFrame(new_columns, index=df.index)
    
    # Unisce il DataFrame originale con quello delle nuove features
    result_df = pd.concat([df, new_df], axis=1)
    
    # Se la modalità di check è attiva, esegue un'analisi aggiuntiva

    return result_df

def create_low_frequency_all(df) -> pd.DataFrame:
    """
    Crea tutte le features a bassa frequenza (stationary TA, wavelet coefficients e slope)
    in maniera sequenziale e le restituisce in un DataFrame.
    """
    # Copia del DataFrame originale
    #df =raw_data.reset_index(drop=True).copy()
    new_columns = {}

    # 1. Calcolo delle caratteristiche di analisi tecnica (stationary TA)
    windows_ta = [ 20, 50, 100]
    for window in windows_ta:
        try:
            # Assumiamo che lf.get_stationary_ta_windows ritorni un DataFrame
            features = lf.get_stationary_ta_windows(df, [window]).to_dict(orient='series')
            new_columns.update(features)
        except Exception as exc:
            print(f"Errore nel calcolo delle caratteristiche per la finestra={window}: {exc}")

    # 2. Calcolo dei coefficienti wavelet
    '''try:
        close_signal = df['close'].pct_change(10).values
        volume_signal = df['volume'].values

        for i, coeff in enumerate(lf.filter_bank(close_signal)):
            new_columns[f'wavelet_coeff_{i}'] = coeff

        for i, coeff in enumerate(lf.filter_bank(volume_signal)):
            new_columns[f'wavelet_coeff_vol_{i}'] = coeff
    except Exception as exc:
        print(f"Errore nel calcolo dei coefficienti wavelet: {exc}")'''

    # 3. Calcolo della pendenza della regressione lineare
    columns_lr = ['close', 'volume']
    windows_lr = [ 20, 60, 100,200]
    for col in columns_lr:
        for window in windows_lr:
            try:
                slope_col_name = f'{col}_slope_{window}'
                new_columns[slope_col_name] = df[col].rolling(window).apply(lf.linear_regression_slope, raw=False)
            except Exception as exc:
                print(f"Errore nel calcolo della pendenza per la colonna {col} con window={window}: {exc}")
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df


def build_dataset(ticker,prices: pd.DataFrame,
                  forecast_horizon: int = 10,
                  windows_features: list = [20,50,100,200],
                  ) -> pd.DataFrame:
    # Calcolo dei rendimenti e della cumulata (usata per i target)
    base_features=prices
    base_features['Returns'] = prices['close'].pct_change()
    cum_returns = base_features['Returns'].cumsum()
    
    # Per le features, definiamo un indice comune: da max(window_features) - 1 a len(returns) - forecast_horizon
    max_w = max(windows_features)
    valid_index = base_features['Returns'].index[max_w - 1: len(base_features['Returns']) - forecast_horizon]
    
    # 1. Calcolo vettoriale delle features base per ciascuna finestra
  
    for w in windows_features:
        base_features[f'volatility_{w}'] = base_features['Returns'].rolling(window=w).std().loc[valid_index]
        base_features[f'skewness_{w}']   = base_features['Returns'].rolling(window=w).skew().loc[valid_index]
        base_features[f'kurtosis_{w}']   = base_features['Returns'].rolling(window=w).kurt().loc[valid_index]
    
    # 2. Calcolo dei target: per ogni orizzonte h, target = cum_returns[t+h] - cum_returns[t]
    for h in range(1, forecast_horizon + 1):
        base_features[f'target_{h}'] = (cum_returns.shift(-h) - cum_returns).loc[valid_index]
       

    #dataset =pd.concat([base_features, all_stationary_ta, ta_windows, vol_estimators_df, vsa_df], axis=1)
    dataset =pd.concat([base_features], axis=1)

    
    dataset=create_technical_indicators_all(dataset)

    dataset=create_low_frequency_all(dataset)

    dataset=create_high_frequency_all(dataset)



    #print("date",dataset['date'])
    dataset.index = pd.to_datetime(dataset.index)
    dataset['day_of_year'] = dataset.index.dayofyear
    dataset['day_of_week'] = dataset.index.dayofweek
    dataset['month'] = dataset.index.month



    dataset=balance_ticker_data(ticker,dataset)
    dataset=analyze_rating(dataset,ticker)
    dataset=dataset.dropna()
    dataset_economic= create_dataframe()
    # Assumendo che l'indice di dataset e dataset_economic sia di tipo datetime
    dataset_economic_aligned = dataset_economic.reindex(dataset.index, method='ffill')
    dataset = pd.concat([dataset, dataset_economic_aligned], axis=1)


    
    non_stationary_cols = []
    constant_cols = []
    
    dataset = dataset.T.drop_duplicates().T
    dataset= dataset.loc[:, ~dataset.columns.duplicated()]
   

    dropped_cols = []
    for col in list(dataset.columns):
        col_data = dataset[col]
        
        # Se col_data è un DataFrame (ovvero ha ndim=2) con più colonne
        if isinstance(col_data, pd.DataFrame):
            # Se vuoi separarli: itera sulle sottocolonne
            print(f"🔍 La colonna '{col}' è multidimensionale, processa le sue sottocolonne:")
            for subcol in col_data.columns:
                try:
                    series = col_data[subcol].dropna()
                    result = adfuller(series)
                    #print(f"Risultato per {col}_{subcol}: {result}")
                except Exception as e:
                    print(f"⚠️ Errore per {col}_{subcol}: {e}")
        else:
            # Per le Series, verifica che non siano costanti
            if col_data.dropna().nunique() == 1:
                #print(f"⚠️ La colonna '{col}' è costante. La droppo.")
                dataset = dataset.drop(columns=[col])
                dropped_cols.append(col)
                continue

            try:
                result = adfuller(col_data.dropna())
                #print(f"Risultato per {col}: {result}")
            except Exception as e:
                print(f"⚠️ Errore per {col}: {e}")

    if dropped_cols:
        print(f"\nColonne droppate: {dropped_cols}")
    for col in dataset.columns:
        col_data = dataset[col]
        # Se col_data è un DataFrame, controlla se tutte le sue colonne sono costanti
        if isinstance(col_data, pd.DataFrame):
            # Rimuove i NaN e calcola nunique per ogni colonna, quindi verifica se tutte sono uguali a 1
            if (col_data.dropna().nunique() == 1).all():
                constant_cols.append(col)
        else:
            # Per una Series, il risultato di nunique() è uno scalare
            if col_data.dropna().nunique() == 1:
                constant_cols.append(col)
    dataset=dataset.dropna()
    
    dataset = dataset.drop(columns=constant_cols)
    print(dataset)
    if dataset.empty:
        # Se il dataset è vuoto, restituisci il DataFrame così com'è
        return dataset
    #print(f"Colonne costanti eliminate: {constant_cols}")
    
    numeric_columns = dataset.select_dtypes(include='number').columns
    
    for col in numeric_columns:
        result = adfuller(dataset[col])
        p_value = result[1]
        if p_value > 0.15:
            #print(f"Colonna {col} ha un p-value = {p_value:.3f} (non stazionaria)")
            non_stationary_cols.append(col)

    # Droppa le colonne non stazionarie dal dataset
    dataset = dataset.drop(columns=non_stationary_cols)
    dataset = dataset.T.drop_duplicates().T
    
    feature_columns = [col for col in dataset.columns if "target" not in col.lower()]
    target_columns = [f'target_{i+1}' for i in range(forecast_horizon)]
    pvalue_threshold=0.01
    selected_features = []
    for h in range(forecast_horizon):
        for feature in feature_columns:
            ic_orig, ic_flips = rademacher_anti_serum_for_signals_ic(
                forecast_horizon, dataset, signal_col=feature, ret_col=f'target_{h+1}', B=10
            )
            # Calcola il p-value: frazione delle permutazioni in cui l'IC flipped è >= IC originale
            p_value = (ic_flips >= ic_orig).mean()
            if p_value < pvalue_threshold and not np.isnan(ic_orig):
                selected_features.append(feature)
                
                print("feature:",feature)
                print("p-value",p_value)
    print(selected_features)
    dataset=dataset[list(selected_features)+target_columns]
    dataset=dataset.dropna()
    dataset = dataset.T.drop_duplicates().T

    return dataset





