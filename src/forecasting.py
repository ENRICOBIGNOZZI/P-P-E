import numpy as np
import pandas as pd
import os
import joblib
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from src.evaluation import plot_equity_curve
import json
from src import forecasting
from src.evaluation import print_sharpe_ras, print_ras_results
from src.optimization import run_optimization
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from joblib import Parallel, delayed
from src.optimization import optimize_portfolio_sharpe
from src.utils import estimate_covariance_from_returns

def process_forecast_for_h(i, h, data_df, train_window, forecast_horizon, feature_columns, target_columns, k_neighbors):
    # Il training set varia in lunghezza in funzione di h:
    # per h=0, include i-train_window fino a i; per h>0 esclude le ultime h righe
    train = data_df.iloc[i - train_window: i - h]
    y_train = train[target_columns].values
    pvalue_threshold = 0.2

    # Selezione delle feature applicando il test RAS
    selected_features = []
    for feature in feature_columns:
        ic_orig, ic_flips = rademacher_anti_serum_for_signals_ic(
            forecast_horizon, train, signal_col=feature, ret_col=f'target_{h+1}', B=50
        )
        # Calcola il p-value: frazione delle permutazioni in cui l'IC flipped è >= IC originale
        p_value = (ic_flips >= ic_orig).mean()
        if p_value < pvalue_threshold and not np.isnan(ic_orig):
            selected_features.append(feature)

    # Se non viene selezionata nessuna feature, usa le feature originali (oppure gestisci il caso diversamente)
    if not selected_features:
        #print("jsndjn")
        #selected_features = feature_columns.copy()
        return 0,1

    # Pre-processing: scaling e PCA
    X_train = train[selected_features].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    n_features = X_train.shape[1]
    n_components = min(n_features, 5)
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)

    # Inizializzazione e training del modello KNN
    model = KNeighborsRegressor(n_neighbors=k_neighbors)
    model.fit(X_train, y_train[:, h])

    # Prepara il test set: il test viene estratto da data_df.iloc[i]
    test = data_df.iloc[i]
    X_test = test[selected_features].values.reshape(1, -1)
    X_test = scaler.transform(X_test)
    X_test = pca.transform(X_test)

    # Previsione
    pred = model.predict(X_test)[0]

    # Calcolo della varianza basata sui vicini
    distances, indices = model.kneighbors(X_test)
    neighbor_targets = y_train[indices.flatten(), h]
    pred_var = np.var(neighbor_targets) + 0.01  # Offset per la stabilizzazione
    #print("pred",pred.shape)
    #print("pred_var",pred_var.shape)
    return pred, pred_var
def walk_forward_multi_step_kelly(data_df: pd.DataFrame, train_window: int, k_neighbors: int, forecast_horizon: int):
    """
    Applica il metodo walk-forward con KNN per generare previsioni di ritorni e stimare il rischio.
    Ritorna:
      - Sharpe Ratio stimato
      - Serie di rendimenti realizzati
      - Previsioni di ritorni aggregati
      - Previsioni di varianza aggregata
    """
    #feature_columns = ['volatility', 'avg_return', 'prev_return', 'skewness', 'kurtosis']
    feature_columns = [col for col in data_df.columns if "target" not in col.lower()]
    target_columns = [f'target_{i+1}' for i in range(forecast_horizon)]
    
    aggregated_kelly_fractions = []
    realized_aggregated_returns = []
    predicted_returns = {}
    real_return = {}
    predicted_variances = {}

    for i in range(train_window, len(data_df) - forecast_horizon + 1):
        # Estrai il test una sola volta per il dato i
        test = data_df.iloc[i]
        
        
        
        #print("inzio",forecast_horizon)
        '''results = Parallel(n_jobs=(forecast_horizon))(
            delayed(process_forecast_for_h)(
                i, h, data_df, train_window, forecast_horizon, feature_columns, target_columns, k_neighbors
            )
            for h in range(forecast_horizon)
        )
        #print("fine")
        # Separa le previsioni e le varianze per ciascun orizzonte
        forecasts, forecast_vars = zip(*results)'''
        
        # da qui in poi '''
        forecasts = []
        
        forecast_vars = []
        # Ciclo per ogni orizzonte di forecast
        for h in range(forecast_horizon):
            # Il training set varia in lunghezza in funzione di h:
            # per h=0, include i-train_window fino a i, per h>0 esclude le ultime h righe
            train = data_df.iloc[i - train_window: i - h]  
            X_train = train[feature_columns].values
            y_train = train[target_columns].values
            pvalue_threshold=0.2

            selected_features = []
            '''for feature in feature_columns:
                # Applica il test RAS per la feature corrente sul training set
                ic_orig, ic_flips = rademacher_anti_serum_for_signals_ic(
                    forecast_horizon, train, signal_col=feature, ret_col=f'target_{h+1}', B=50
                )
                # Calcola il p-value: frazione delle permutazioni in cui l'IC flipped è >= IC originale
                p_value = (ic_flips >= ic_orig).mean()
                
                # Se il p-value è sotto la soglia, seleziona la feature
                if p_value < pvalue_threshold and not np.isnan(ic_orig):
                    #print("forecasting:",h+1)
                    #print("feature:",feature)
                    selected_features.append(
                        feature,
                    )
                    # Stampa i risultati per la feature selezionata
                    #print_ras_results(signal_name=feature, target=target_columns[h], ic_orig=ic_orig, ic_flips=ic_flips)
            
            #rint(list(selected_features))'''
            selected_features=feature_columns
            X_train = train[list(selected_features)].values
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            n_features = X_train.shape[1]
            n_components = min(n_features, 5)
            pca = PCA(n_components=n_components)
            X_train= pca.fit_transform(X_train)




            # Inizializza e addestra il modello
            model = KNeighborsRegressor(n_neighbors=k_neighbors)
            model.fit(X_train, y_train[:, h])
            

            
            X_test = test[list(selected_features)].values.reshape(1, -1)
            X_test = scaler.transform(X_test)
            X_test= pca.transform(X_test)

            pred = model.predict(X_test)[0]
            forecasts.append(pred)
            
            # Utilizza il metodo kneighbors del modello addestrato per ottenere gli indici dei vicini
            distances, indices = model.kneighbors(X_test)
            neighbor_targets = y_train[indices.flatten(), h]
            
            #pred_var = np.var(neighbor_targets) + 0.01  # Offset per stabilizzazione
            #forecast_vars.append(pred_var)
        
        # Aggrega le previsioni
        mu_agg = np.sum(forecasts)
        sigma2_agg = estimate_covariance_from_returns(mu_agg.reshape(1, -1))#np.sum(forecast_vars)
        kelly_fraction = optimize_portfolio_sharpe((mu_agg).reshape(-1,1),(sigma2_agg).reshape(-1,1))#mu_agg / sigma2_agg if sigma2_agg > 1e-6 else 0
        #kelly_fraction = max(kelly_fraction, 0)
        #mu_agg= max(mu_agg, 0)
        
        aggregated_kelly_fractions.append(kelly_fraction)
        realized = test[target_columns[0]]
        realized_portfolio_return = kelly_fraction * realized
        realized_aggregated_returns.append(realized_portfolio_return)
        
        date = data_df.index[i-1]
        predicted_returns[date] = mu_agg
        predicted_variances[date] = sigma2_agg
        real_return[date] =realized 




        

    realized_returns = np.array(realized_aggregated_returns)
    predicted_returns = predicted_returns#np.array(predicted_returns)
    
    predicted_variances = predicted_variances#np.array(predicted_variances)

    #if realized_returns.std() < 1e-9:
    #    return 0, realized_returns, predicted_returns, predicted_variances
    
    sharpe = realized_returns.mean() / realized_returns.std() * np.sqrt(252)
    return sharpe, realized_returns, predicted_returns, predicted_variances,real_return

# 📌 Calcola l'Information Coefficient (IC)
def compute_information_coefficient(data_df: pd.DataFrame, signal_col: str = 'avg_return', ret_col: str = 'target_1') -> float:
    x = data_df[signal_col].values
    y = data_df[ret_col].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oggetto = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
    return oggetto
def get_predicted_metrics(data_df, train_window, k_neighbors, forecast_horizon):
    """
    Genera i ritorni previsti e la varianza stimata usando il modello di forecasting.
    Restituisce due array: predicted_returns e predicted_variances.
    """
    #print(data_df)
    _, effettivi, predicted_returns, predicted_variances,real_returns = walk_forward_multi_step_kelly(
        data_df, train_window=train_window, k_neighbors=k_neighbors, forecast_horizon=forecast_horizon
    )
    #plot_equity_curve(effettivi,"Equity Curve")

    return predicted_returns, predicted_variances,real_returns
# 📌 Test di robustezza per IC (Information Coefficient)
def rademacher_anti_serum_for_signals_ic(forecast_horizon, data_df: pd.DataFrame, signal_col: str = 'avg_return', ret_col: str = 'target_1', B: int = 100):
    """
    Testa se il segnale è robusto attraverso permutazioni casuali (Rademacher).
    """
    
    ic_orig = compute_information_coefficient(data_df, signal_col=signal_col, ret_col=ret_col)
    ic_flips = np.array([
        compute_information_coefficient(
            data_df.assign(**{signal_col: data_df[signal_col] * np.random.choice([1, -1], size=len(data_df))}),
            signal_col=signal_col, ret_col=ret_col
        )
        for _ in range(B)
    ])
    return ic_orig, ic_flips

# 📌 Test di robustezza per Sharpe Ratio
def rademacher_anti_serum_for_sharpe(forecast_horizon, data_df: pd.DataFrame, train_window: int, k_neighbors: int, target_cols: list = None, B: int = 100):
    """
    Testa se lo Sharpe Ratio è robusto attraverso permutazioni casuali (Rademacher).
    """
    if target_cols is None:
        target_cols = [f'target_{i+1}' for i in range(forecast_horizon)]
    
    sharpe_orig, _, _, _,_ = walk_forward_multi_step_kelly(
        data_df, train_window=train_window, k_neighbors=k_neighbors, forecast_horizon=forecast_horizon
    )

    sharpe_flips = []
    for _ in range(B):
        df_flipped = data_df.copy()
        for col in target_cols:
            df_flipped[col] = df_flipped[col].values * np.random.choice([1, -1], size=len(df_flipped))
        
        sharpe_flip, _, _, _,_ = walk_forward_multi_step_kelly(
            df_flipped, train_window=train_window, k_neighbors=k_neighbors, forecast_horizon=forecast_horizon
        )
        sharpe_flips.append(sharpe_flip)

    return sharpe_orig, np.array(sharpe_flips)


# 📌 Funzioni di salvataggio/caricamento modelli
def train_final_models(data_df: pd.DataFrame, train_window: int, k_neighbors: int, forecast_horizon: int = 5):
    """
    Allena i modelli finali kNN per ogni orizzonte temporale.
    """
    
    feature_columns = [col for col in data_df.columns if "target" not in col.lower()]
    target_columns = [f'target_{i+1}' for i in range(forecast_horizon)]
    
    
    models = {}

    for h in range(forecast_horizon):
        #train = data_df.iloc[-train_window:]
        if h > 0:
            train = data_df.iloc[-train_window:-h]
        else:
            train = data_df.iloc[-train_window:]

        #print(train.shape)
        #train = data_df.iloc[i - train_window:i-h]
        X_train = train[feature_columns].values
        y_train = train[target_columns[h]].values
        model = KNeighborsRegressor(n_neighbors=k_neighbors)
        model.fit(X_train, y_train)
        models[f'model_horizon_{h+1}'] = model
    
    return models

def save_models(models: dict, ticker: str):
    """
    Salva il dizionario dei modelli.
    """
    model_path = os.path.join("models", f"{ticker}_models.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(models, model_path)
    print(f"✅ Modelli salvati in {model_path}")

def load_models(ticker: str):
    """
    Carica il dizionario dei modelli.
    """
    model_path = os.path.join("models", f"{ticker}_models.pkl")
    if os.path.exists(model_path):
        print(f"📂 Modelli caricati da {model_path}")
        return joblib.load(model_path)
    else:
        print(f"❌ Nessun modello trovato per {ticker}.")
        return None
def load_optimization_params(ticker):
    filename = f"optimization_params_{ticker}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            params = json.load(f)
        print(f"📂 Parametri di ottimizzazione caricati da {filename}")
        return params
    else:
        print(f"❌ Nessun file di parametri di ottimizzazione trovato per {ticker}.")
        return None
def save_forecast_results(ticker, forecast_df,FORECASTS_DIR):
    """
    Salva le previsioni su file CSV con timestamp.
    """

    filename = os.path.join(FORECASTS_DIR, f"{ticker}_forecast.csv")
    forecast_df.to_csv(filename, index=False)
    print(f"📊 Previsioni salvate: {filename}")
def load_forecast_data(tickers,FORECASTS_DIR):
    """
    Carica i file CSV corrispondenti ai tickers specificati, li unisce lungo l'asse delle colonne
    e li ordina per data.
    
    Args:
        tickers (list): Lista di ticker da caricare.

    Returns:
        pd.DataFrame: DataFrame combinato con i dati di previsione.
    """
    all_data = []
    
    for ticker in tickers:
        file_path = os.path.join(FORECASTS_DIR, f"{ticker}_forecast.csv")
        
        if not os.path.exists(file_path):
            print(f"⚠️ File non trovato per {ticker}: {file_path}")
            continue  # Salta il ticker se il file non esiste

        try:
            df = pd.read_csv(file_path, parse_dates=["date"])  # Parsing automatico della data
            df.set_index("date", inplace=True)  # Usa la data come indice
            
            # Rinominiamo le colonne per evitare conflitti nei merge
            df = df.add_prefix(f"{ticker}_")  
            
            all_data.append(df)
            print(f"✅ Caricato: {file_path} - Shape: {df.shape}")
        
        except Exception as e:
            print(f"❌ Errore nel caricamento di {file_path}: {e}")
    
    if not all_data:
        print("❌ Nessun file valido caricato.")
        return None
    
    # Unire tutti i DataFrame lungo l'asse delle colonne usando la data come chiave
    combined_df = pd.concat(all_data, axis=1)
    #print(combined_df)
    
    # Ordinare per data
    combined_df.sort_index(inplace=True)

    # Reintegra la colonna "date" come prima colonna
    combined_df.reset_index(inplace=True)

    return combined_df



def carica_allena_modello(ticker,LOAD_MODELS,data_df,forecast_horizon,lista_parametri1,lista_parametri2,FORECASTS_DIR):
    if LOAD_MODELS:
        models = forecasting.load_models(ticker)
        opt_params = load_optimization_params(ticker)
        if models is not None and opt_params is not None:
            print(f"✅ Modello pre-caricato per {ticker}")
            best_tw = opt_params["train_window"]
            best_kn = opt_params["k_neighbors"]
            skip_tests = True  # Non rifare i test se carichiamo il modello
        else:
            print("❌ Nessun modello trovato. Allenamento necessario...")
            skip_tests = False  # Facciamo i test solo se ri-ottimizziamo il modello
    else:
        skip_tests = False

    if not skip_tests:
        print("🛠 Allenamento modello...")
        best_tw, best_kn, best_sharpe = run_optimization(data_df, forecast_horizon,lista_parametri1,lista_parametri2)
        opt_params = {"train_window": best_tw, "k_neighbors": best_kn, "sharpe": best_sharpe}
        models = forecasting.train_final_models(data_df, train_window=best_tw, k_neighbors=best_kn, forecast_horizon=forecast_horizon)
        forecasting.save_models(models, ticker)
        
        with open(f"optimization_params_{ticker}.json", "w") as f:
            json.dump(opt_params, f)
        print(f"📂 Parametri di ottimizzazione salvati per {ticker}")


    # 📌 Test di robustezza: SOLO SE IL MODELLO È STATO RIALLENATO
    if not skip_tests:
        print(f"\n🔍 Testing Overfitting per {ticker}")
        signal_columns = [col for col in data_df.columns if not col.startswith("target_")]
        target_columns = [col for col in data_df.columns if col.startswith("target_")]
        for signal_col in signal_columns:
            for target_col in target_columns:
                ic_orig, ic_flips = rademacher_anti_serum_for_signals_ic(
                    forecast_horizon, data_df, signal_col=signal_col, ret_col=target_col, B=500
                )
                print_ras_results(signal_col, target_col, ic_orig, ic_flips)
        sharpe_orig, sharpe_flips = rademacher_anti_serum_for_sharpe(
            forecast_horizon, data_df, train_window=best_tw, k_neighbors=best_kn, B=10
        )
        print_sharpe_ras(sharpe_orig, sharpe_flips)
    predicted_returns, predicted_variances,real_returns = forecasting.get_predicted_metrics(data_df, best_tw, best_kn, forecast_horizon)
    predicted_returns= pd.Series(predicted_returns)
    predicted_variances= pd.Series(predicted_variances)
    real_returns=pd.Series(real_returns)
    print(predicted_returns.shape)
    print(real_returns.shape)

    forecast_df = pd.DataFrame({
        "date": predicted_returns.index, 
        "predicted_return": predicted_returns.values, 
        "predicted_variance": predicted_variances.values,
        "ticker": ticker,  # Aggiungiamo il ticker
        "real_returns_step_after":real_returns.values,
    })
    save_forecast_results(ticker, forecast_df,FORECASTS_DIR)