import yfinance as yf
import pandas as pd

# src/data_download.py
import yfinance as yf
import pandas as pd
import os
import requests
from src.DATASET.features_engineering import build_dataset
import time
api_key = 'c0b27f4ee2a77f6df0e3c4f1a559c841'

def get_stock_data_segment(symbol: str, start: str, end: str, max_retries: int = 100, retry_delay: int = 1) -> pd.DataFrame:
    """
    Scarica un segmento di dati storici per il simbolo dato,
    dall'intervallo start-end utilizzando l'endpoint per i grafici giornalieri.
    
    In caso di errore, esegue un retry fino a max_retries volte.
    """
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/1day/{symbol}?from={start}&to={end}&apikey={api_key}"
    retries = 0
    success = False
    data = None

    while not success and retries < max_retries:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                success = True
            else:
                print(f"Errore: Status code {response.status_code} per {symbol} da {start} a {end}")
                retries += 1
                time.sleep(retry_delay)
        except Exception as e:
            print(f"Errore: {e} (tentativo {retries + 1}/{max_retries})")
            retries += 1
            time.sleep(retry_delay)
            
    if success and data:
        df_segment = pd.DataFrame(data)
        # Assicuriamoci che la colonna data esista e sia in formato datetime
        if 'date' in df_segment.columns:
            df_segment['date'] = pd.to_datetime(df_segment['date'])
        return df_segment
    else:
        print(f"Impossibile scaricare i dati per {symbol} da {start} a {end} dopo {max_retries} tentativi.")
        return pd.DataFrame()  # Ritorna DataFrame vuoto se non ci sono dati

def download_data(symbol: str, start_date: str, end_date: str, interval_days: int = 90, skip_download_if_exists: bool = True) -> pd.DataFrame:
    """
    Scarica i dati storici per il simbolo dato e li salva in ENRICO/data/raw/<symbol>.csv.
    Se il file esiste e skip_download_if_exists è True, carica i dati dal file locale.
    
    Il download viene effettuato per segmenti:
    - Si parte da end_date e si scarica un intervallo di 'interval_days' giorni.
    - Alla fine di ogni iterazione, la nuova data di fine viene impostata come il giorno precedente alla data minima ottenuta.
    - Si continua fino a che la data di inizio del segmento è precedente o uguale a start_date.
    """
    raw_path = os.path.join("ENRICO", "data", "raw", f"{symbol}.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    if skip_download_if_exists and os.path.exists(raw_path):
        print(f"[data_download] File {raw_path} già esistente. Carico i dati locali.")
        df = pd.read_csv(raw_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index("date")
        return df

    print(f"[data_download] Scarico i dati per {symbol} dall'intervallo {start_date} a {end_date}...")
    
    # Conversione delle stringhe in datetime
    start_date_dt = pd.to_datetime(start_date)
    current_end = pd.to_datetime(end_date)
    
    all_data = pd.DataFrame()

    # Ciclo finché il current_end è successivo o uguale a start_date
    while current_end >= start_date_dt:
        # Calcola la data di inizio per il segmento attuale
        current_start = current_end - pd.Timedelta(days=interval_days)
        if current_start < start_date_dt:
            current_start = start_date_dt

        # Formatta le date in stringa
        current_start_str = current_start.strftime("%Y-%m-%d")
        current_end_str = current_end.strftime("%Y-%m-%d")

        print(f"Download dei dati da {current_start_str} a {current_end_str} per {symbol}...")
        df_segment = get_stock_data_segment(symbol, current_start_str, current_end_str)

        if df_segment.empty:
            print("Nessun dato restituito per questo segmento, interrompo il ciclo.")
            break

        # Concatena i dati: i nuovi dati vengono aggiunti in testa (poiché si procede a ritroso)
        all_data = pd.concat([df_segment, all_data], ignore_index=True)

        # Trova la data minima ottenuta in questo segmento
        if 'date' in df_segment.columns:
            earliest_date = df_segment['date'].min()
            # Imposta il nuovo current_end come il giorno precedente alla data minima ottenuta
            current_end = pd.to_datetime(earliest_date) - pd.Timedelta(days=1)
        else:
            print("La colonna 'date' non è presente nei dati, interrompo il ciclo.")
            break

    # Se abbiamo raccolto dei dati, li ordiniamo per data e impostiamo l'indice
    if not all_data.empty:
        all_data['date'] = pd.to_datetime(all_data['date'])
        all_data = all_data.sort_values("date")
        all_data = all_data.set_index("date")
        # Salvataggio su file CSV
        all_data.to_csv(raw_path)
        print(f"[data_download] Dati salvati in {raw_path}")
    else:
        print("Nessun dato è stato scaricato.")

    return all_data


def load_or_build_dataset(ticker, start_date, end_date, forecast_horizon, load_processed=True,LOAD_DATASETS=True):
    processed_path = os.path.join("ENRICO", "data", "processed", f"{ticker}_forecasting_{forecast_horizon}_processed.csv")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    if load_processed and os.path.exists(processed_path):
        print(f"[load_or_build_dataset] Carico il dataset processato per {ticker} da {processed_path}")
        dataset = pd.read_csv(processed_path, index_col="date", parse_dates=True)
        return dataset

    # Se non esiste il dataset processato, scarica i dati raw
    df_prices = download_data(ticker, start_date, end_date, skip_download_if_exists=LOAD_DATASETS)
    df_prices = df_prices.dropna()
    if df_prices.empty or df_prices.shape[0] < 1000:
        print(f"⚠️ df_prices per {ticker} è vuoto o con pochi dati. Salto questo ticker...")
        return None

    # Costruisci il dataset con feature engineering
    dataset = build_dataset(ticker,df_prices, forecast_horizon=forecast_horizon)
    if dataset.empty or dataset.shape[0] < 700:
        print(f"⚠️ Il dataset per {ticker} è vuoto o con pochi dati dopo il feature engineering. Salto questo ticker...")
        return None

    # Salva il dataset processato
    dataset.to_csv(processed_path)
    print(f"✅ Dataset per {ticker} costruito e salvato con {dataset.shape[0]} campioni.")
    return dataset

def download_data_factors(symbol: str, start_date: str, end_date: str, skip_download_if_exists: bool = True) -> pd.DataFrame:
    """
    Scarica i dati storici per il simbolo dato usando yfinance, 
    salvandoli in data/raw/<symbol>.csv se non già presenti (opzionale).
    
    Parametri:
      - symbol: es. "AAPL"
      - start_date, end_date: stringhe data "YYYY-MM-DD"
      - skip_download_if_exists: se True, non riscarica i dati se il file esiste.
    """
    import os

    
    raw_path = os.path.join("ENRICO", "data_factors", "raw", f"{symbol}.csv")
    
    # Crea la cartella ENRICO/data/raw/ se non esiste
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
   
    
    if skip_download_if_exists and os.path.exists(raw_path):
        print(f"[data_download] File {raw_path} già esistente. Carico i dati locali.")
        df = pd.read_csv(raw_path)
        df=df.set_index("date")

        return df
    
    print(f"[data_download] Scarico i dati per {symbol} da {start_date} a {end_date}...")
    # Conversione delle stringhe in datetime
    start_date_dt = pd.to_datetime(start_date)
    current_end = pd.to_datetime(end_date)
    
    all_data = pd.DataFrame()

    # Ciclo finché il current_end è successivo o uguale a start_date
    while current_end >= start_date_dt:
        # Calcola la data di inizio per il segmento attuale
        current_start = current_end - pd.Timedelta(days=90)
        if current_start < start_date_dt:
            current_start = start_date_dt

        # Formatta le date in stringa
        current_start_str = current_start.strftime("%Y-%m-%d")
        current_end_str = current_end.strftime("%Y-%m-%d")

        print(f"Download dei dati da {current_start_str} a {current_end_str} per {symbol}...")
        df_segment = get_stock_data_segment(symbol, current_start_str, current_end_str)

        if df_segment.empty:
            print("Nessun dato restituito per questo segmento, interrompo il ciclo.")
            break

        # Concatena i dati: i nuovi dati vengono aggiunti in testa (poiché si procede a ritroso)
        all_data = pd.concat([df_segment, all_data], ignore_index=True)

        # Trova la data minima ottenuta in questo segmento
        if 'date' in df_segment.columns:
            earliest_date = df_segment['date'].min()
            # Imposta il nuovo current_end come il giorno precedente alla data minima ottenuta
            current_end = pd.to_datetime(earliest_date) - pd.Timedelta(days=1)
        else:
            print("La colonna 'date' non è presente nei dati, interrompo il ciclo.")
            break
    df=all_data
    #df = df[::-1]
    #df.reset_index(inplace=True)
    print(df)
    df.to_csv(raw_path, index=True)
    
    return df
def load_preprocessed_data(ticker):
    filename = f"processed_data_{ticker}.pkl"
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    return None

def save_preprocessed_data(ticker, data_df):
    filename = f"processed_data_{ticker}.pkl"
    data_df.to_pickle(filename)
    print(f"📂 Dataset preprocessato salvato: {filename}")