import itertools
import numpy as np
import pandas as pd
from numpy.linalg import norm, inv
from scipy.optimize import minimize

def beta_stability(loadings_list):
    """Calcola il BETARER: media delle differenze relative fra loadings consecutivi.
       Più basso è il valore, maggiore è la stabilità dei beta nel tempo."""
    diffs = []
    for i in range(1, len(loadings_list)):
        prev = loadings_list[i-1]
        curr = loadings_list[i]
        # Calcola la differenza relativa in norma Frobenius
        diff = norm(np.array(list(curr.values())) - np.array(list(prev.values())), 'fro') / (norm(np.array(list(prev.values())), 'fro') + 1e-8)
        diffs.append(diff)
    return np.mean(diffs) if len(diffs) > 0 else np.nan
def reduce_turnover(B_t, B_t1):
    A = np.dot(B_t.T,  B_t1)
    U, _, Vt = np.linalg.svd(A)
    X_star = np.dot(Vt.T, U.T)
    B_t1_transformed = np.dot(B_t1, X_star)
    return B_t1_transformed

import numpy as np
from numpy.linalg import norm

import numpy as np
from numpy.linalg import norm

import numpy as np
from numpy.linalg import norm

import numpy as np
from numpy.linalg import norm

def dict_to_array(d):
    # Ordina le chiavi per righe e colonne
    keys = sorted(d.keys())
    return np.array([[d[i][j] for j in keys] for i in keys])

def convert_to_matrix(x):
    # Se x è un dict, convertilo direttamente
    if isinstance(x, dict):
        return dict_to_array(x)
    # Se x è un array e il suo primo elemento è un dict, converti x.item() (assumendo che x contenga un singolo dict)
    elif isinstance(x, np.ndarray):
        if x.size > 0 and isinstance(x.flat[0], dict):
            # Assumiamo che x contenga un solo dizionario
            return dict_to_array(x.item())
    return x

def frobenius_error(est_cov, emp_cov):
    # Converte gli input, gestendo sia dict che array contenenti dict
    est_cov = convert_to_matrix(est_cov)
    emp_cov = convert_to_matrix(emp_cov)
    
    return norm(est_cov - emp_cov, 'fro') / norm(emp_cov, 'fro')






def random_portfolio_test(window_returns, est_cov, var_real, num_trials=100):
    """Calcola il RPAVT: media dei quadrati dell'errore tra varianza predetta e realizzata su portafogli casuali"""
    N = window_returns.shape  # window_returns: (T x N)
    errors = []
    for _ in range(num_trials):
        w = np.random.uniform(0, 1, size=N)
        if np.sum(np.abs(w)) == 0:
            continue
        w = w / np.sum(np.abs(w))
        portf_returns = window_returns @ w
        var_pred = w.T @ est_cov @ w

        var_real=np.sum(var_real)
        errors.append((var_pred - var_real)**2)
    return np.mean(errors)

def min_var_portfolio_error(est_cov, emp_cov):
    """Calcola la varianza realizzata del portafoglio a minima varianza (stimata con est_cov) rispetto a emp_cov"""
    N = est_cov.shape[0]
    def objective(w):
        return w.T @ est_cov @ w
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bnds = [(0, 1) for _ in range(N)]
    w0 = np.ones(N) / N
    res = minimize(objective, w0, method='SLSQP', bounds=bnds, constraints=cons)
    if not res.success:
        return np.nan
    w = res.x
    return w.T @ emp_cov @ w
def run_factor_model(prices, stocks, p, m, tau_f, tau_s):
    """
    Esegue il modello fattoriale in due stadi per i parametri specificati.

    Parametri:
      - prices: DataFrame con i prezzi.
      - stocks: lista degli stock (utilizzata per definire num_stocks e per etichettare i risultati).
      - p: numero di fattori principali per il primo stadio.
      - m: numero di fattori per il secondo stadio.
      - tau_f: half-life per il primo decadimento.
      - tau_s: half-life per il secondo stadio (deve essere maggiore di tau_f).

    Ritorna un dizionario in cui ogni chiave è una data (ottenuta dall'ultima osservazione
    della finestra) e i valori sono i risultati associati, con le componenti relative agli stock
    etichettate con il nome dell'azione.
    """
    import numpy as np
    import pandas as pd

    num_stocks = len(stocks)
    window_size = 3 * tau_s

    # Inizializzo i dizionari in cui salvare i risultati per ogni data (finestra)
    factors_dict = {}
    loadings_dict = {}
    volatility_factors_dict = {}
    volatility_dict = {}
    empirical_cov_dict = {}
    volatility_idio_dict = {}
    window_returns_dict = {}
    explained_variance_dict = {}
    alpha_dict = {}
    estimated_ret = {}
    betas={}

    # Calcolo degli esponenti per il decadimento
    window_returns = prices.iloc[0:window_size].T
    esponente = np.exp(-np.arange(window_returns.shape[1]) / tau_f)[::-1]
    esponente_secondo_stage = np.exp(-np.arange(window_returns.shape[1]) / tau_s)[::-1]
    
    n = 0
    B_hat_vecchio = None  # Memorizza il B_hat della finestra precedente
    B_vecchio = None
    

    for t in range(window_size, len(prices), 1):
        # Estrai la data corrispondente all'ultima osservazione della finestra
        data = prices.iloc[t-1].name

        # Seleziona la finestra dei ritorni e trasponi (l'indice saranno i nomi degli stock)
        window_returns = prices.iloc[t-window_size:t].T

        # Primo stadio: applica il decadimento esponenziale
        W_tau_f = np.diag(esponente)
        R_tilde = window_returns @ W_tau_f

        # Decomposizione SVD
        U, S, Vt = np.linalg.svd(R_tilde, full_matrices=False)
        explained_ratio = np.sum(S[:p]**2) / np.sum(S**2)
        explained_variance_dict[data] = explained_ratio

        # Regolarizzazione degli autovalori
        tau_max = (1 + np.sqrt(num_stocks / window_size))**2 * np.median(S)
        S = np.where(S > tau_max, S, tau_max)
        U_p = U[:, :p]
        S_p = np.diag(S[:p])
        Vt_p = Vt[:p, :]
        F = S_p @ Vt_p  # Fattori

        # Calcolo dei loadings (B)
        if n > 0:
            B = reduce_turnover(B_vecchio, U_p)
            B_vecchio = B
        else:
            B = U_p
            B_vecchio = B

        # Stima della volatilità fattoriale
        Sigma_hat = U_p @ S_p @ S_p @ U_p.T

        # Calcolo della volatilità idiosincratica iniziale
        sigma_idio_start = np.sqrt(np.sum((R_tilde - (U_p @ S_p @ Vt_p))**2, axis=1))
        W_sigma = np.diag(1 / sigma_idio_start)
        W_inv_sigma = np.linalg.inv(W_sigma)

        # Secondo stadio: applica il decadimento esponenziale al secondo stadio
        W_tau_s = np.diag(esponente_secondo_stage)
        R_hat = W_sigma @ window_returns @ W_tau_s
        U_hat, S_hat, Vt_hat = np.linalg.svd(R_hat, full_matrices=False)

        tau_max_2 = (1 + np.sqrt(num_stocks / window_size))**2 * np.median(S_hat)
        S_hat = np.where(S_hat > tau_max_2, S_hat, tau_max_2)
        U_m = W_inv_sigma @ U_hat[:, :m]
        S_m = np.diag(S_hat[:m])
        Vt_m = Vt_hat[:m, :]
        F_hat = S_m @ Vt_m  # Nuovi fattori

        # Calcolo dei loadings del secondo stadio
        if n > 0:
            B_hat = reduce_turnover(B_hat_vecchio, U_m)
            B_hat_vecchio = B_hat
        else:
            B_hat = U_m
            B_hat_vecchio = B_hat
            n += 1

        # Calcola le stime di volatilità complessiva e idiosincratica
        Sigma_hat = U_m @ S_m @ S_m @ U_m.T
        sigma_idio = np.mean(S_hat[m:]**2) * np.eye(num_stocks)
        sigma = Sigma_hat + sigma_idio

        # Calcolo di alpha (residuo)
        alpha = R_tilde.iloc[:, -1] - (B_hat @ F_hat)[:, -1]

        # Salva i risultati associati alla data corrente

        # Fattori: si considerano come vettore (senza etichettatura per stock)
        factors_dict[data] = F_hat[:, -1].tolist()

        # Empirical covariance: matrice con righe e colonne etichettate con i nomi degli stock
        emp_cov = R_tilde.shape[1] * np.cov(window_returns.T.values, rowvar=False)
        emp_cov_df = pd.DataFrame(emp_cov, index=stocks, columns=stocks)
        empirical_cov_dict[data] = emp_cov_df.to_dict()

        # Loadings: matrice con righe etichettate con i nomi degli stock
        loadings_df = pd.DataFrame(B_hat, index=stocks)
        loadings_dict[data] = loadings_df.to_dict(orient='list')

        # Volatility factors: matrice con righe e colonne etichettate con i nomi degli stock
        vol_fac_df = pd.DataFrame(Sigma_hat, index=stocks, columns=stocks)
        volatility_factors_dict[data] = vol_fac_df.to_dict()

        # Volatility idiosincratica:
        vol_idio_df = pd.DataFrame(sigma_idio, index=stocks, columns=stocks)
        volatility_idio_dict[data] = vol_idio_df.to_dict()

        # Volatility complessiva:
        vol_df = pd.DataFrame(sigma, index=stocks, columns=stocks)
        volatility_dict[data] = vol_df.to_dict()

        # Window returns: la Series ha già come indice i nomi degli stock
        window_returns_dict[data] = window_returns.iloc[:, -1].to_dict()

        # Alpha: converto in Series con indice = stocks
        alpha_series = pd.Series(alpha, index=stocks)
        alpha_dict[data] = alpha_series.to_dict()


        #print(B_hat.shape)
        betas[data] = {i: pd.Series(B_hat[:, i], index=stocks).to_dict() for i in range(B_hat.shape[1])}

        
        estimated_ret[data]= pd.Series((B_hat @ F_hat)[:, -1], index=stocks).to_dict()
        

    return {
        #"factors_list": factors_dict,
        "loadings_list": loadings_dict,
        "volatility_list": volatility_dict,
        "volatility_factors_list": volatility_factors_dict,
        "volatility_idio_list": volatility_idio_dict,
        "empirical_cov_list": empirical_cov_dict,
        #"window_returns_list": window_returns_dict,
        "explained_variance_list": explained_variance_dict,
        #"alpha_list": alpha_dict,
        "estimatated_factors_return":estimated_ret,
        "betas":betas,
    }

'''p   m  tau_f  tau_s  mean_frobenius_error  mean_RP_error  mean_MinVar_error  beta_stability  explained_variance  avg_corr_residua
0   3   3    250    251              1.870761   1.400018e+07           0.079446        0.032051            0.394294          0.001445
1   3   5    250    251              1.831358   1.400018e+07           0.079544        0.040539            0.394294          0.000606
2   3  10    250    251              1.797175   1.400019e+07           0.079422        0.075657            0.394294          0.000749
3   5   3    250    251              2.042739   1.400017e+07           0.080358        0.026434            0.454658          0.000284'''


def grid_search_factor_model(prices,
                             lista_p=[30],
                             lista_m=[30],
                             lista_tau_f=[50],
                             lista_tau_s=[250]):
    """
    Esegue una grid search sul modello dei fattori utilizzando le liste di parametri specificate.
    
    Parameters:
    - prices: pd.DataFrame
        DataFrame contenente i prezzi con colonne rappresentanti gli stock.
    - lista_p: list, default=[1]
        Lista dei possibili valori per il parametro p (es. fattori principali nel primo stadio).
    - lista_m: list, default=[1]
        Lista dei possibili valori per il parametro m (es. fattori nel secondo stadio).
    - lista_tau_f: list, default=[100, 250]
        Lista dei possibili half-life per il primo decadimento.
    - lista_tau_s: list, default=[100, 250]
        Lista dei possibili half-life per il secondo decadimento.
    
    Returns:
    - results_df: pd.DataFrame
        DataFrame con tutti i risultati della grid search.
    - best_params: pd.Series
        Serie contenente la migliore combinazione di parametri basata sulla minimizzazione dell'errore totale.
    
    Nota: La funzione si basa su altre funzioni già definite nel contesto:
        run_factor_model, frobenius_error, random_portfolio_test, min_var_portfolio_error, beta_stability.
    """
    import itertools
    import numpy as np
    import pandas as pd

    stocks = list(prices.columns)
    grid_results = []
    
    for p, m, tau_f, tau_s in itertools.product(lista_p, lista_m, lista_tau_f, lista_tau_s):
        # Se tau_s non è maggiore di tau_f, passa alla combinazione successiva
        if tau_s <= tau_f:
            continue
        
        print(f"Eseguo per p={p}, m={m}, tau_f={tau_f}, tau_s={tau_s}")
        output = run_factor_model(prices, stocks, p, m, tau_f, tau_s)
        
        # Ora le chiavi dei dizionari rappresentano le date (le finestre)
        vol_dict = output["volatility_list"]
        emp_cov_dict = output["empirical_cov_list"]
        #win_returns_dict = output["window_returns_list"]
        loadings_dict = output["loadings_list"]
        explained_variance_dict = output["explained_variance_list"]
        '''alpha_dict = output["alpha_list"]
        
        # Per alpha, costruiamo un DataFrame in cui le righe sono le finestre (indicizzate per data)
        alpha_df = pd.DataFrame.from_dict(alpha_dict, orient="index")
        corr_matrix = alpha_df.corr()
        # Creazione di una maschera per escludere la diagonale
        mask = np.eye(corr_matrix.shape[0], dtype=bool)
        # Applichiamo la maschera per ottenere solo le correlazioni non-diagonali
        corr_no_diag = corr_matrix.where(~mask)
        # Calcoliamo la media delle correlazioni non-nulle (ignorando i NaN)
        avg_corr = corr_no_diag.stack().mean()'''
        
        # Inizializza le liste per le metriche
        frob_errors = []
        rp_errors = []
        minvar_errors = []
        
        # Itera sulle finestre (ordinando le date per coerenza)
        for date in sorted(vol_dict.keys()):
            est_cov = np.array(vol_dict[date])
            emp_cov = np.array(emp_cov_dict[date])
            # win_returns è stato salvato come lista; convertiamola in array
            #w_returns = np.array(win_returns_dict[date])
            frob_errors.append(frobenius_error(est_cov, emp_cov))

            #rp_errors.append(random_portfolio_test(np.array(list(w_returns.T.item().values())), convert_to_matrix(est_cov),(convert_to_matrix(emp_cov)), num_trials=100))
            #array = list(dizionario.values())
            minvar_errors.append(min_var_portfolio_error(convert_to_matrix(est_cov), convert_to_matrix(emp_cov)))
        
        # Calcola le medie delle metriche
        mean_frob = np.nanmean(frob_errors)
        #mean_rp = np.nanmean(rp_errors)
        mean_minvar = np.nanmean(minvar_errors)
        beta_stab = beta_stability(list(loadings_dict.values()))
        explained_variance_avg = np.mean(list(explained_variance_dict.values()))
        
        grid_results.append({
            "p": p,
            "m": m,
            "tau_f": tau_f,
            "tau_s": tau_s,
            "mean_frobenius_error": mean_frob,
            #"mean_RP_error": mean_rp,
            "mean_MinVar_error": mean_minvar,
            "beta_stability": beta_stab,
            "explained_variance": explained_variance_avg,
            #"avg_corr_residua": avg_corr,
        })
    
    results_df = pd.DataFrame(grid_results)
    print("\nRisultati della Grid Search:")
    print(results_df)
    
    # Calcola un errore totale (ad esempio, somma di mean_MinVar_error e beta_stability)
    results_df["total_error"] = results_df["beta_stability"]#results_df["mean_MinVar_error"] + r
    best_idx = results_df["total_error"].idxmin()
    best_params = results_df.loc[best_idx]
    
    print("\nMigliore combinazione di parametri:")
    print(best_params)
    print("\nVarianza spiegata media dai fattori:", best_params["explained_variance"])
    
    return results_df, best_params


# Esempio di utilizzo:
# results_df, best_params = grid_search_factor_model(prices)
