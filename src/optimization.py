import cvxpy as cp
import numpy as np
from itertools import product
from src import  forecasting

def estimate_covariance_from_returns(predicted_returns: np.ndarray):
    """
    Stima la matrice di covarianza come il prodotto esterno del vettore dei rendimenti predetti.
    """
    predicted_returns = predicted_returns.reshape(-1, 1)
    covariance_matrix = predicted_returns @ predicted_returns.T
    return covariance_matrix

def optimize_portfolio_sharpe(predicted_returns, predicted_cov_matrix, max_volatility=0.2):
    """
    Ottimizza il portafoglio massimizzando il ritorno con un vincolo sulla volatilità.
    """
    n = len(predicted_returns)
    w = cp.Variable(n)
    
    objective = cp.Maximize(predicted_returns @ w)
    constraints = [
        w >= 0,
        cp.norm1(w) <= 1,
        cp.quad_form(w, predicted_cov_matrix) <= max_volatility**2
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return w.value #if prob.status == cp.OPTIMAL else np.ones(n) / n
    '''n = len(predicted_returns)
    if predicted_returns>0:
        return np.ones(n) / n
    elif predicted_returns<0:
        return -np.ones(n) / n
    else:
        return -np.zeros(n) / n'''



def run_optimization(data_df, forecast_horizon,train_window_list,k_neighbors_list):
    
    results = []
    print("\n🔎 Ricerca dei parametri migliori...")
    for tw, kn in product(train_window_list, k_neighbors_list):
        if tw < kn:
            continue
        sharpe, _ , _ , _,_ = forecasting.walk_forward_multi_step_kelly(
            data_df, train_window=tw, k_neighbors=kn, forecast_horizon=forecast_horizon
        )
        print(f" Train window: {tw}, k_neighbors: {kn}, Sharpe: {sharpe:.4f}")
        results.append((tw, kn, sharpe))
    if results:
        best = max(results, key=lambda x: x[2])
        best_tw, best_kn, best_sharpe = best
        print("\n🏆 Migliori parametri trovati:")
        print(f" - train_window = {best_tw}")
        print(f" - k_neighbors = {best_kn}")
        print(f" - Sharpe ratio = {best_sharpe:.4f}")
        return best_tw, best_kn, best_sharpe
    else:
        print("❌ Nessuna combinazione valida trovata.")
        return 20, 1, 0.0