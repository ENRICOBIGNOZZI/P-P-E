import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from src.optimization import optimize_portfolio_sharpe
from src.utils import estimate_covariance_from_returns,make_positive_definite



def plot_equity_curve(returns: np.array, title: str = "Equity Curve"):
    """
    Plotta la curva di rendimento cumulativo.
    """
    equity_curve = (1 + returns).cumprod()
    plt.figure()
    plt.plot(equity_curve, label="Equity Curve")
    plt.xlabel("Passo walk-forward")
    plt.ylabel("Rendimento cumulativo")
    plt.title(title)
    plt.legend()
    plt.show()
    return

def obtain_weights_and_final_dataset(forecast_df,tickers,FORECASTS_DIR):
    all_allocations = []
    forecast_df=forecast_df.dropna()



    unique_dates = forecast_df["date"].unique()

    for date in unique_dates:
        sub_df = forecast_df[forecast_df["date"] == date]
        pred_returns = sub_df.filter(like="predicted_return").values[0]
        covariance_matrix = estimate_covariance_from_returns(pred_returns)#+0.01
        
        covariance_matrix=make_positive_definite(covariance_matrix)
        weights = optimize_portfolio_sharpe(pred_returns, covariance_matrix)
        #print(sub_df)
        
        allocation = {"date": date}

        # Itera su ogni ticker e relativo peso
        for t, w in zip(tickers, weights):
            allocation[f"w_{t}"] = w
            allocation[t] = sub_df[f"{t}_real_returns_step_after"].values

        # Aggiungi il dizionario alla lista delle allocazioni
        all_allocations.append(allocation)


    allocations_df = pd.DataFrame(all_allocations)
    print(allocations_df)
    allocations_df.to_csv(os.path.join(FORECASTS_DIR, "portfolio_allocations.csv"), index=False)
    '''if not pd.api.types.is_datetime64_any_dtype(dataset_for_back .index):
        dataset_for_back .index = pd.to_datetime(dataset_for_back .index)
    if not pd.api.types.is_datetime64_any_dtype(allocations_df ['date']):
        allocations_df ['date'] = pd.to_datetime(allocations_df ['date'])


    df_merged = dataset_for_back .merge(allocations_df , left_index=True, right_on='date', how='outer')
    df_merged.set_index('date', inplace=True)
    df_merged=df_merged.dropna()'''

    df_backtest = backtest_portfolio(allocations_df.dropna(),tickers)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(df_backtest.index, df_backtest['cumulative_return'], label='Cumulative Return', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Portfolio Cumulative Return Over Time')
    plt.legend()
    plt.grid()
    plt.show()


def backtest_portfolio(df, tickers):
    """
    Calcola il ritorno giornaliero del portafoglio come somma ponderata
    dei ritorni per tutti i tickers presenti nella lista 'tickers'
    e calcola il rendimento cumulativo del portafoglio.
    """
    # Calcola il ritorno giornaliero del portafoglio in modo dinamico
    df['portfolio_return'] = sum(df[ticker] *3*df[f'w_{ticker}'] for ticker in tickers)
    ritorni=np.array(df['portfolio_return'].dropna().values)
    sharpe = np.mean(ritorni)/ritorni.std() * np.sqrt(252)



    print("SHARPE RATIO PORTOFLIO:",sharpe)
    
    # Calcola il valore cumulativo del portafoglio assumendo un capitale iniziale di 1
    df['cumulative_return'] = (1 + df['portfolio_return']).cumprod()
    
    return df

def print_ras_results(signal_name: str, target: str, ic_orig: float, ic_flips: np.array):
    """
    Stampa i risultati del test RAS per un segnale.
    """
    p_value = (ic_flips >= ic_orig).mean()
    print(f"RAS per il segnale '{signal_name}' e target '{target}':")
    print("  IC originale:", ic_orig)
    print("  Media IC flips:", ic_flips.mean())
    print("  p-value:", p_value, "\n")
    
def print_sharpe_ras(sharpe_orig: float, sharpe_flips: np.array):
    p_value_sharpe = (sharpe_flips >= sharpe_orig).mean()
    print("RAS per Sharpe (flipping rendimenti):")
    print("  Sharpe originale:", sharpe_orig)
    print("  Media Sharpe flips:", sharpe_flips.mean())
    print("  p-value:", p_value_sharpe)


def analyze_equity_with_factor_model(equity_returns, factor_returns, exposures, Sigma_f, confidence=0.95, plot=True):
    """
    Analizza e visualizza la performance della equity utilizzando un modello fattoriale.

    Parametri:
      - equity_returns: pandas Series, rendimenti osservati della equity (PnL istantaneo)
                        con indice temporale.
      - factor_returns: pandas DataFrame, rendimenti dei fattori con indice temporale identico
                        a quello di equity_returns e colonne denominate (es. 'F1', 'F2', ...).
      - exposures: pandas DataFrame, esposizioni (β) del portafoglio per ciascun fattore con
                   lo stesso indice e colonne di factor_returns.
      - Sigma_f: numpy array di dimensione (k, k), matrice di covarianza costante dei fattori.
      - confidence: livello di confidenza per gli intervalli (default 0.95).
      - plot: booleano, se True genera i grafici della performance.

    Restituisce:
      - df_results: DataFrame contenente, per ogni istante:
            * partial contributions per ogni fattore,
            * total_factor_pnl (somma dei contributi),
            * residual (equity_return - total_factor_pnl),
            * rendimenti cumulativi sia per l’equity che per il modello fattoriale,
            * le esposizioni per ogni fattore.
    """
    # Verifica che gli indici e le colonne siano compatibili
    if not (equity_returns.index.equals(factor_returns.index) and equity_returns.index.equals(exposures.index)):
        raise ValueError("Gli indici di equity_returns, factor_returns e exposures devono essere uguali.")
    
    if not (list(factor_returns.columns) == list(exposures.columns)):
        raise ValueError("Le colonne di factor_returns e exposures devono essere identiche (stessi fattori).")
    
    dates = equity_returns.index
    k = factor_returns.shape[1]
    
    # Calcola il valore critico z in base al livello di confidenza
    z = norm.ppf(0.5 + confidence / 2.0)
    
    # Liste per salvare i risultati per ogni istante
    total_factor_pnl_list = []
    residual_list = []
    partial_contributions_dict = {col: [] for col in factor_returns.columns}
    # Salva le esposizioni (β) per ogni fattore
    exposures_dict = {col: [] for col in exposures.columns}
    
    # Itera per ogni data
    for date in dates:
        # Estrai i vettori per il periodo t
        f_t = factor_returns.loc[date].values        # vettore (k,)
        beta_t = exposures.loc[date].values            # vettore (k,)
        r_t = equity_returns.loc[date]                 # rendimento osservato (scalare)
        
        # Contributi parziali: beta_t * f_t (elementwise)
        partial_contrib = beta_t * f_t
        total_factor_pnl = np.sum(partial_contrib)
        residual = r_t - total_factor_pnl
        
        # Salva i risultati
        total_factor_pnl_list.append(total_factor_pnl)
        residual_list.append(residual)
        for i, col in enumerate(factor_returns.columns):
            partial_contributions_dict[col].append(partial_contrib[i])
            exposures_dict[col].append(beta_t[i])
    
    # Crea il DataFrame dei risultati
    df_results = pd.DataFrame({
        "equity_return": equity_returns,
        "total_factor_pnl": total_factor_pnl_list,
        "residual": residual_list
    }, index=dates)
    
    # Aggiunge le colonne dei contributi parziali e delle esposizioni
    for col in factor_returns.columns:
        df_results[f"{col}_contrib"] = partial_contributions_dict[col]
        df_results[f"{col}_beta"] = exposures_dict[col]
    
    # Calcola i rendimenti cumulativi
    df_results["cumulative_equity_return"] = df_results["equity_return"].cumsum()
    df_results["cumulative_factor_pnl"] = df_results["total_factor_pnl"].cumsum()
    
    # Se richiesto, genera i plot
    if plot:
        # Plot 1: Contributi istantanei per ciascun fattore e residuo idiosincratico
        plt.figure(figsize=(12, 6))
        for col in factor_returns.columns:
            plt.plot(df_results.index, df_results[f"{col}_contrib"], label=f"Contrib {col}")
        plt.plot(df_results.index, df_results["residual"], label="Idiosincratico", linestyle="--", color="black")
        plt.xlabel("Data")
        plt.ylabel("Contributo giornaliero")
        plt.title("Contributi istantanei dei Fattori e Componente Idiosincratica")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # Plot 2: Confronto istantaneo tra PnL fattoriale e rendimento osservato
        plt.figure(figsize=(10, 5))
        plt.plot(df_results.index, df_results["total_factor_pnl"], label="PnL Fattoriale", marker='o')
        plt.plot(df_results.index, df_results["equity_return"], label="Rendimento Osservato", marker='s')
        plt.xlabel("Data")
        plt.ylabel("Valore giornaliero")
        plt.title("Confronto istantaneo: PnL Fattoriale vs Rendimento Osservato")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # Plot 3: Andamento cumulativo della performance
        plt.figure(figsize=(10, 5))
        plt.plot(df_results.index, df_results["cumulative_equity_return"], label="Rendimento Osservato Cumulativo")
        plt.plot(df_results.index, df_results["cumulative_factor_pnl"], label="PnL Fattoriale Cumulativo")
        plt.xlabel("Data")
        plt.ylabel("Rendimento Cumulativo")
        plt.title("Andamento cumulativo della Performance")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # Plot 4: Evoluzione delle esposizioni (β) per ciascun fattore
        plt.figure(figsize=(12, 6))
        for col in exposures.columns:
            plt.plot(df_results.index, df_results[f"{col}_beta"], label=f"Beta {col}")
        plt.xlabel("Data")
        plt.ylabel("Esposizione (β)")
        plt.title("Evoluzione delle Esposizioni del Portafoglio")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    return df_results



def analyze_selection_sizing(returns_universe, selected_indices, weights_real):
    """
    Analizza la performance di stock picking (selection) e del sizing del portafoglio.

    Parametri:
      - returns_universe: array NumPy di dimensione (T, N) con i rendimenti storici 
                          dell'universo dei titoli.
      - selected_indices: lista o array degli indici (o posizioni) dei titoli selezionati dal gestore.
      - weights_real: array NumPy di dimensione (M,) con i pesi reali assegnati dal gestore 
                      ai titoli selezionati (che devono sommare a 1).

    La funzione calcola:
      - Portafoglio Universo (equi-pesato su tutti i N titoli)
      - Portafoglio di Selezione (equi-pesato sui titoli selezionati)
      - Portafoglio Reale (con i pesi reali)
      - IR per ogni portafoglio
      - Selection Skill = IR(Selezione) - IR(Universo)
      - Sizing Skill = IR(Reale) - IR(Selezione)

    Vengono stampati i valori IR, le skill e dei consigli su cosa migliorare.
    """

    # Verifica input e converte in array se necessario
    returns_universe = np.array(returns_universe)  # (T, N)
    T, N = returns_universe.shape
    M = len(selected_indices)
    
    # Controlla che weights_real abbia dimensione corretta e somma a 1 (entro un piccolo errore)
    weights_real = np.array(weights_real)
    if weights_real.shape[0] != M:
        raise ValueError("La lunghezza di weights_real deve essere uguale al numero di titoli selezionati.")
    if not np.isclose(weights_real.sum(), 1.0, atol=1e-4):
        raise ValueError("I pesi reali devono sommare a 1.")
    
    # 1) Calcolo dei rendimenti giornalieri dei portafogli
    # Portafoglio Universo: equi-pesato su tutti i titoli
    weights_universe = np.ones(N) / N  # array di dimensione (N,)
    portf_universe = returns_universe @ weights_universe  # (T,)
    
    # Portafoglio di Selezione: equi-pesato sui titoli selezionati
    weights_selection = np.zeros(N)
    weights_selection[selected_indices] = 1.0 / M
    portf_selection = returns_universe @ weights_selection  # (T,)
    
    # Portafoglio Reale: usa weights_real per i titoli selezionati
    weights_real_full = np.zeros(N)
    weights_real_full[selected_indices] = weights_real
    portf_real = returns_universe @ weights_real_full  # (T,)
    
    # 2) Calcolo dell'Information Ratio (IR)
    def information_ratio(portf_rets):
        # Calcola la media e la deviazione standard (campionaria) dei rendimenti
        mean_ret = np.mean(portf_rets)
        std_ret = np.std(portf_rets, ddof=1)
        return mean_ret / std_ret if std_ret != 0 else 0.0
    
    IR_universe = information_ratio(portf_universe)
    IR_selection = information_ratio(portf_selection)
    IR_real = information_ratio(portf_real)
    
    # 3) Calcolo di Selection e Sizing Skill
    selection_skill = IR_selection - IR_universe
    sizing_skill = IR_real - IR_selection
    
    # 4) Stampa dei risultati
    print("=== RISULTATI DELL'ATTRIBUZIONE SELECTION vs SIZING ===\n")
    print(f"IR (Portafoglio Universo)  = {IR_universe:.4f}")
    print(f"IR (Portafoglio Selezione) = {IR_selection:.4f}")
    print(f"IR (Portafoglio Reale)     = {IR_real:.4f}\n")
    
    print(f"Selection Skill = {selection_skill:.4f}")
    print(f"Sizing Skill    = {sizing_skill:.4f}\n")
    
    # Commenti sulla Selection Skill
    if selection_skill > 0:
        print("La Selection Skill è POSITIVA: la selezione dei titoli è efficace, in media batte l'universo.")
    else:
        print("La Selection Skill NON è POSITIVA: la selezione dei titoli non migliora la performance rispetto all'universo.")
        print("Suggerimento: rivedi i criteri di stock picking, potresti considerare di ampliare l'universo o affinare i segnali.")
    
    print()  # riga vuota
    # Commenti sulla Sizing Skill
    if sizing_skill > 0:
        print("La Sizing Skill è POSITIVA: i pesi attivi migliorano la performance rispetto a un equi-pesato sui titoli selezionati.")
        print("Suggerimento: enfatizza ulteriormente le posizioni con maggiore conviction.")
    else:
        print("La Sizing Skill NON è POSITIVA: la differenziazione dei pesi peggiora la performance rispetto a un equi-pesato.")
        print("Suggerimento: valuta di ridurre la concentrazione dei pesi (ovvero, avvicinarti a un equi-pesato) o rivedere il processo di sizing.")
    
    # Consigli finali
    print("\n=== CONSIGLI FINALI ===")
    if selection_skill <= 0 and sizing_skill <= 0:
        print("- Rivedi i criteri di selezione (stock picking) e considera un approccio di sizing più uniforme.")
    elif selection_skill > 0 and sizing_skill <= 0:
        print("- La selezione è buona, ma il sizing è un punto debole: prova a equi-pesare i titoli selezionati.")
    elif selection_skill <= 0 and sizing_skill > 0:
        print("- Il sizing è positivo, ma la selezione dei titoli non batte l'universo: focalizzati maggiormente sul miglior stock picking.")
    else:
        print("- Ottima combinazione: la selezione e il sizing contribuiscono entrambi a migliorare la performance. Continua così!")
    
    # Ritorna anche i portafogli e gli IR per eventuali ulteriori analisi
    results = {
        "portf_universe": portf_universe,
        "portf_selection": portf_selection,
        "portf_real": portf_real,
        "IR_universe": IR_universe,
        "IR_selection": IR_selection,
        "IR_real": IR_real,
        "selection_skill": selection_skill,
        "sizing_skill": sizing_skill
    }
    return results
