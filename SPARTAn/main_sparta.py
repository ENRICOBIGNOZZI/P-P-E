import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Import delle funzioni se sono in un file esterno
from testing_function import adf_test
from stationary_SPARTAn import spartan_regression_fast, forecast_y
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

def plot_3d_cluster_assignment(X_torch, Y_torch, gamma_prob, cluster_idx=(0,1)):
    """
    Plot 3D colorato secondo il cluster dominante.
    """
    X_np = X_torch.detach().numpy()
    Y_np = Y_torch.detach().numpy().reshape(-1)
    gamma_np = gamma_prob.detach().numpy()

    feat1, feat2 = cluster_idx
    x_vals = X_np[:, feat1]
    y_vals = X_np[:, feat2]

    # Cluster dominante per ogni punto
    cluster_assign = np.argmax(gamma_np, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x_vals, y_vals, Y_np, c=cluster_assign, alpha=0.6)
    ax.set_xlabel(f"Feature {feat1}")
    ax.set_ylabel(f"Feature {feat2}")
    ax.set_zlabel("Target Y")
    plt.title("Color by Cluster Assignment")
    plt.show()


def plot_3d_local_l2(X_torch, Y_torch, gamma_prob, Lambda, cluster_idx=(0,1)):
    """
    Plot 3D colorato secondo la 'forza' della regolarizzazione L2 in ciascun cluster.
    Calcoliamo la somma dei quadrati di Lambda per ogni cluster (norma L2).
    Poi per ogni punto t usiamo la norma L2 del cluster dominante.
    """
    X_np = X_torch.detach().numpy()
    Y_np = Y_torch.detach().numpy().reshape(-1)
    gamma_np = gamma_prob.detach().numpy()

    feat1, feat2 = cluster_idx
    x_vals = X_np[:, feat1]
    y_vals = X_np[:, feat2]

    cluster_assign = np.argmax(gamma_np, axis=1)

    # Calcoliamo la norma L2 dei parametri di regressione in ogni cluster
    # Lambda ha shape (K, D, M) ipotizzando quell'ordine
    Lambda_np = Lambda.detach().numpy()
    K = Lambda_np.shape[0]
    local_l2 = []
    for k in range(K):
        # Somma dei quadrati di tutti i coefficienti di quel cluster
        local_l2.append(np.sum(Lambda_np[k,:,:]**2))
    local_l2 = np.array(local_l2)

    # Assegniamo a ogni punto la "norma L2" del cluster che lo domina
    color_l2 = local_l2[cluster_assign]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x_vals, y_vals, Y_np, c=color_l2, alpha=0.6)
    ax.set_xlabel(f"Feature {feat1}")
    ax.set_ylabel(f"Feature {feat2}")
    ax.set_zlabel("Target Y")
    plt.title("Color by L2 magnitude of local regression")
    plt.colorbar(sc, label="Sum of squares of Lambda_k")
    plt.show()


def plot_3d_weighted_sum(X_torch, Y_torch, w_prob, cluster_idx=(0,1)):
    """
    Plot 3D colorato secondo la somma pesata delle feature con i w_prob globali.
    Per ogni punto t calcoliamo sum_d [w_prob[d] * X[t,d]].
    """
    X_np = X_torch.detach().numpy()
    Y_np = Y_torch.detach().numpy().reshape(-1)
    w_np = w_prob.detach().numpy().reshape(-1)

    feat1, feat2 = cluster_idx
    x_vals = X_np[:, feat1]
    y_vals = X_np[:, feat2]

    # somma pesata
    weighted_sum = np.sum(X_np * w_np[np.newaxis,:], axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x_vals, y_vals, Y_np, c=weighted_sum, alpha=0.6)
    ax.set_xlabel(f"Feature {feat1}")
    ax.set_ylabel(f"Feature {feat2}")
    ax.set_zlabel("Target Y")
    plt.title("Color by global weighted sum (w_prob · x)")
    plt.colorbar(sc, label="∑ w[d]*X[t,d]")
    plt.show()


def main_plots(X_torch, Y_torch, gamma_prob, Lambda, w_prob, cluster_idx=(0,1)):
    # 1) Colori per cluster
    plot_3d_cluster_assignment(X_torch, Y_torch, gamma_prob, cluster_idx)

    # 2) Colori secondo l'ampiezza dei parametri (dove incide L2)
    plot_3d_local_l2(X_torch, Y_torch, gamma_prob, Lambda, cluster_idx)

    # 3) Colori secondo la somma pesata (w_prob)
    plot_3d_weighted_sum(X_torch, Y_torch, w_prob, cluster_idx)


# ESEMPIO D'USO (dopo aver calcolato spartan_regression_fast):
# main_plots(X_torch, Y_torch, gamma_prob, Lambda, w_prob, cluster_idx=(0,1))


def main():
    # Carica il dataset
    '''df = pd.read_csv('time_series_features.csv')
    df = df.T.drop_duplicates().T
    df = df.dropna()

    # Creiamo dei lag su 'target'
    for i in range(1, 10):
        df[f'target_shif{i}'] = df['target'].shift(i)

    # Esegui un test di stazionarietà ADF su ogni colonna, salvi i risultati
    stationarity_results = {}
    stationary_columns = []
    for col in df.columns:
        series = df[col]
        p_value = adf_test(series)  # adf_test deve essere definito o importato
        stationarity_results[col] = {"p_value": p_value, "stationary": p_value < 0.05}
        if p_value < 0.05:
            stationary_columns.append(col)

    # Al momento utilizziamo tutto 'df' (o df_stationary = df[stationary_columns])
    df_stationary = df  # se volessi filtrare con le colonne stazionarie, usa [stationary_columns]
    df_stationary['target'] = df['target']

    # Copia, rimuove duplicati e dropna
    df = df_stationary.copy()
    df = df.T.drop_duplicates().T
    df = df.dropna()

    # Scala i dati
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    df = df_scaled
    print(df.head())

    # Definiamo quante previsioni in avanti (forecasting=1 => target_forecast1)
    forecasting = 1
    target_columns = []
    for i in range(1, forecasting+1):
        df[f'target_forecast{i}'] = df['target'].shift(-i)
        target_columns.append(f'target_forecast{i}')

    # Convertiamo in tensori PyTorch
    input_columns = df.columns  # Attenzione: se vuoi escludere 'target_forecast1' dalle feature, rimuovila da input_columns
    X = torch.tensor(df[input_columns].values, dtype=torch.float32)
    Y = torch.tensor(df[target_columns].values, dtype=torch.float32)

    n_samples = X.shape[0]
    n_train = int(n_samples * 0.6)

    # Train/test split
    X_train = X[:n_train]
    Y_train = Y[:n_train]
    X_test  = X[n_train:]
    Y_test  = Y[n_train:]

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)

    # Numero di 'box' per la discretizzazione
    K = 3'''





    torch.manual_seed(0)
    np.random.seed(0)

    T = 1000       # Number of timesteps
    D = 20         # Number of features (changed from 1 to 20)
    M = 1          # Number of outputs (same as before)

    # Initialize the arrays
    X_np = np.zeros((T, D), dtype=np.float32)
    Y_np = np.zeros((T, M), dtype=np.float32)

    # Populate X and Y
    for t in range(T):
        # For each time step t, create 20 features
        for d in range(D):
            X_np[t, d] = np.cos((t + d) / 50) + np.random.normal(0, 0.1)
        
        # Single target value
        Y_np[t, 0] = np.cos((t + 1) / 50) + np.random.normal(0, 0.1)

    print("X shape:", X_np.shape)  # (1000, 20)
    print("Y shape:", Y_np.shape)  # (1000, 1)


    X_torch = torch.from_numpy(X_np)
    Y_torch = torch.from_numpy(Y_np)

    Lambda, gamma_prob, w_prob, C = spartan_regression_fast(
        X_torch,
        Y_torch,
        K=10,                
        epsilon_d=0.0001,
        epsilon_r=1,
        epsilon_l2=0.0,
        epochs=10000,
        lr=1e-1,
        verbose=True
    )

    print("\n=== Fine addestramento ===")
    print("w_prob =", w_prob)
    print("gamma_prob shape =", gamma_prob.shape)
    print("Lambda shape =", Lambda.shape)
    print("C shape =", C.shape)

    Y_hat_torch = forecast_y(X_torch, Lambda, w_prob, C)
    Y_hat_np = Y_hat_torch.detach().numpy().reshape(-1)

    Y_np_flat = Y_np.reshape(-1)
    mse_train = np.mean((Y_hat_np - Y_np_flat) ** 2)
    print(f"MSE train = {mse_train:.6f}")

    '''plt.figure()
    plt.plot(Y_np_flat, label="True series")
    plt.plot(Y_hat_np, label="Forecast (SPARTAn)", linestyle='--')
    plt.title("Forecast vs test")
    plt.legend()
    plt.show()



    error_np = Y_hat_np - Y_np_flat
    plt.figure()
    plt.hist(error_np, bins=50)
    plt.title("Histogram of Error (Y_hat - Y_true)")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()

    # Histogram of weights (W)
    # Assuming 'w_prob' is a torch Tensor containing the weights
    w_prob_np = w_prob.detach().numpy().reshape(-1)

    plt.figure()
    plt.hist(w_prob_np, bins=50)
    plt.title("Distribution of Weights (W)")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.show()'''
    main_plots(X_torch, Y_torch, gamma_prob, Lambda, w_prob, cluster_idx=(0,1))


if __name__ == "__main__":
    main()
