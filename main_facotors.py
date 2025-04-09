import pandas as pd
import json
import os
import sys
import numpy as np

import cvxpy as cp
from src.DATASET import data_download
from src import factor_optimization
import matplotlib.pyplot as plt

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Se hai altri tipi non serializzabili, li gestisci qui
        return super().default(obj)
def main():
    LOAD_DATASETS=True
    LOAD_MODEL =False
    model_filepath = "best_factor_model_output.json"
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]

    # Estrae la colonna dei simboli
    tickers = df['Symbol'].tolist()[0:200]
    print(tickers )
    
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    datasets = {}

    for ticker in tickers:
        print(f"\n== 📥 Elaborazione per {ticker} ==")
        df_prices = data_download.download_data_factors(ticker, start_date, end_date,skip_download_if_exists=LOAD_DATASETS )
        
        datasets[ticker] = df_prices['close']
        print(f"✅ Dataset per {ticker} costruito con {df_prices['close'].shape} campioni.")
    df_all = pd.concat(datasets, axis=1)
    df_all.fillna(method='ffill', inplace=True)
    df_all.fillna(method='bfill', inplace=True)
    df_all=df_all.dropna(axis=1)
    df_all=df_all.dropna(axis=0)
    df_all=df_all.pct_change().dropna()
    '''corr_matrix = df_all.corr()
    print(corr_matrix.shape)
    # Creazione di una maschera per escludere la diagonale
    mask = np.eye(corr_matrix.shape[0], dtype=bool)

    # Applichiamo la maschera per ottenere solo le correlazioni non diagonali
    corr_no_diag = corr_matrix.where(~mask)

    # Calcoliamo la media delle correlazioni non-nulle (ignorando i NaN)
    avg_corr = corr_no_diag.stack().mean()

    print("Media delle correlazioni (escludendo la diagonale):", avg_corr)
    np.fill_diagonal(corr_matrix.values, np.nan)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title('Matrice di correlazione degli α (senza diagonale)')
    plt.xticks(range(corr_matrix.shape[1]), corr_matrix.columns, rotation=90)
    plt.yticks(range(corr_matrix.shape[0]), corr_matrix.index)
    plt.show()'''
    
    if LOAD_MODEL and os.path.exists(model_filepath):
        try:
            with open(model_filepath, "r") as f:
                best_output = json.load(f)
            print("Output del modello caricato da file.")
        except json.JSONDecodeError:
            print("Errore nel decodificare il file JSON. Il file potrebbe essere vuoto o corrotto.")
            best_output = None  # oppure gestisci il caso in altro modo
    else:
        # Esegui la grid search per ottenere i migliori parametri
        results_df, best_params = factor_optimization.grid_search_factor_model(df_all)
        stocks = list(df_all.columns)
        best_output = factor_optimization.run_factor_model(
            df_all,
            stocks,
            int(best_params["p"]),
            int(best_params["m"]),
            int(best_params["tau_f"]),
            int(best_params["tau_s"])
        )
        def convert_obj(obj):
            if isinstance(obj, dict):
                return {k: convert_obj(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_obj(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            else:
                return obj

        best_output_converted = convert_obj(best_output)

        with open(model_filepath, "w") as f:
            json.dump(best_output_converted, f, indent=4)



    # Visualizzazione della matrice di correlazione degli α (senza diagonale)
    
    '''alpha_df = pd.DataFrame(best_output['alpha_list']).T
    print(alpha_df)
    
    corr_matrix = alpha_df.corr()
    print(corr_matrix.shape)
    # Creazione di una maschera per escludere la diagonale
    mask = np.eye(corr_matrix.shape[0], dtype=bool)

    # Applichiamo la maschera per ottenere solo le correlazioni non diagonali
    corr_no_diag = corr_matrix.where(~mask)

    # Calcoliamo la media delle correlazioni non-nulle (ignorando i NaN)
    avg_corr = corr_no_diag.stack().mean()

    print("Media delle correlazioni (escludendo la diagonale):", avg_corr)
    np.fill_diagonal(corr_matrix.values, np.nan)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title('Matrice di correlazione degli α (senza diagonale)')
    plt.xticks(range(corr_matrix.shape[1]), corr_matrix.columns, rotation=90)
    plt.yticks(range(corr_matrix.shape[0]), corr_matrix.index)
    plt.show()'''

if __name__ == "__main__":
    main()


