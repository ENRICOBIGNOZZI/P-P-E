import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from main_kernel import rbf_kernel
from TIME_SPARTA import spartan_regression_fast_time, forecast_y_time
from stationary_SPARTAn import spartan_regression_fast, forecast_y

# ======================
# 1. Caricamento dati
# ======================
df = pd.read_csv("sinusoidal_features.csv")[:10000]
df["target"] = df["signal"].shift(-1)
df = df.dropna()
df=df.drop(['time_squared'],axis=1)
print(df.columns)
'''plt.figure(figsize=(12,6))
for col in df.columns:
    plt.plot(df[col], label=col)

plt.xlabel("Indice")
plt.ylabel("Valore")
plt.title("Serie Temporali - Tutte le Colonne")
plt.legend()
plt.grid(True)

# Salva il grafico in formato PNG
plt.savefig("all_time_series.png")
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df["target"], label="Target", linestyle="--")
plt.xlabel("Indice")
plt.ylabel("Valore")
plt.title("Serie Temporale: Signal e Target")
plt.legend()
plt.grid(True)

# Salva il plot in una immagine PNG
plt.savefig("target_series_plot.png")
plt.show()'''
X_np = df.drop(columns=["signal"]).values.astype(np.float32)
Y_np = df["target"].values.reshape(-1, 1).astype(np.float32)

scaler = StandardScaler()
X_np = scaler.fit_transform(df.drop(columns=["signal"]).values.astype(np.float32))
Y_np = df["target"].values.reshape(-1, 1).astype(np.float32)

# Riduzione per velocità (facoltativo)
X_np = X_np  # [:1000]
Y_np = Y_np  # [:1000]

X = torch.tensor(X_np)
Y = torch.tensor(Y_np)

n = X.shape[0]
n_train = int(0.60 * n)
X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]

# ======================
# 2. Kernel Ridge Regression Full (KRR Full)
# ======================
lambda_ = 1e-3
gamma = 0.000001
K_train = rbf_kernel(X_train, X_train, gamma=gamma)

n_tr = X_train.shape[0]
K_train_reg = K_train + lambda_ * torch.eye(n_tr)

# Solve per alpha in: (K + lambda I) alpha = Y
alpha = torch.linalg.solve(K_train_reg, Y_train)
Y_train_pred = K_train @ alpha
mse_train = torch.mean((Y_train_pred - Y_train)**2)
print("KRR Full Train MSE:", mse_train.item())
K_test_train = rbf_kernel(X_test, X_train, gamma=gamma)
Y_test_pred = K_test_train @ alpha

mse_test = torch.mean((Y_test_pred - Y_test)**2)
print("KRR Full Test MSE:", mse_test.item())

# ======================
# 3. KRR rank-1 (KRR Deep)
# ======================
# Implementazione basata su rete neurale per apprendere un kernel
Y_pred_krr_rank1 = Y_test_pred 


# ======================
# 4. SPARTAn standard e SPARTAn Time
# ======================
Lambda, gamma_prob, w_prob, C = spartan_regression_fast(
    X_train,
    Y_train,
    K=3,
    epsilon_d=0.001,
    epsilon_r=1,
    epsilon_l2=0.,
    epochs=10000,
    lr=1e-2,
    verbose=True
)
Y_hat_torch = forecast_y(X_test, Lambda, w_prob, C)
Y_pred_spartan = Y_hat_torch.detach().numpy().reshape(-1)

Lambda, gamma_prob, w_prob, C = spartan_regression_fast_time(
    X_train,
    Y_train,
    K=3,
    epsilon_d=0.001,
    epsilon_r=1,
    epsilon_l2=0.00,
    epochs=10000,
    lr=1e-2,
    verbose=True
)
Y_hat_torch = forecast_y_time(X_test, Lambda, w_prob, C)
Y_pred_spartan_time = Y_hat_torch.detach().numpy().reshape(-1)

Y_test_np = Y_test.detach().numpy().flatten()

# ======================
# 5. LSTM Model
# ======================
# Funzione per creare sequenze dai dati (con finestra mobile)
window_size = 10
def create_sequences(X, Y, window_size):
    Xs = []
    Ys = []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        Ys.append(Y[i+window_size])
    return np.array(Xs), np.array(Ys)

# Creiamo le sequenze utilizzando i dati già scalati (X_np e Y_np)
X_seq, Y_seq = create_sequences(X_np, Y_np, window_size)
X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
Y_seq_tensor = torch.tensor(Y_seq, dtype=torch.float32)

# Divisione in training e test per le sequenze (60% training)
n_seq = X_seq_tensor.shape[0]
n_train_seq = int(0.70 * n_seq)
X_train_seq = X_seq_tensor[:n_train_seq]
X_test_seq = X_seq_tensor[n_train_seq:]
Y_train_seq = Y_seq_tensor[:n_train_seq]
Y_test_seq = Y_seq_tensor[n_train_seq:]

# Definizione del modello LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)       # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]         # consideriamo l'output dell'ultimo timestep
        out = self.fc(out)
        return out

input_size = X_np.shape[1]  # Numero di feature per ogni timestep
hidden_size = 64
num_layers = 1
output_size = 1

lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=1e-2)
epochs_lstm = 1000

# Ciclo di training per LSTM
for epoch in range(epochs_lstm):
    lstm_model.train()
    optimizer_lstm.zero_grad()
    output = lstm_model(X_train_seq)
    loss = criterion(output, Y_train_seq)
    loss.backward()
    optimizer_lstm.step()
    if epoch % 20 == 0 or epoch == epochs_lstm - 1:
        print(f"LSTM Epoch {epoch}: Train Loss = {loss.item():.6f}")

lstm_model.eval()
with torch.no_grad():
    Y_pred_lstm = lstm_model(X_test_seq)
    mse_lstm = torch.mean((Y_pred_lstm - Y_test_seq) ** 2)
    print("LSTM Test MSE:", mse_lstm.item())

Y_pred_lstm_np = Y_pred_lstm.detach().numpy().flatten()
print(Y_pred_lstm_np)
print(Y_pred_lstm_np.shape)
Y_true_seq = Y_test_np[-len(Y_test_seq):]

# ======================
# 6. Confronto e Visualizzazione
# ======================
preds = {
    "KRR Full": Y_pred_krr_rank1.detach().numpy().flatten(),
    #"KRR Deep": Y_pred_krr_rank2.detach().numpy().flatten(),
    "SPARTAn": Y_pred_spartan.flatten(),
    #"SPARTAn Time": Y_pred_spartan_time.flatten(),
    "LSTM": Y_pred_lstm_np
}

for name, pred in preds.items():
    print(name)
    if name=='LSTM':
        mse = mean_squared_error(Y_test_np[-len(pred):], pred)
    else:
        mse = mean_squared_error(Y_test_np, pred[:len(Y_test_np)])
    print(f"{name} MSE (completo o adattato): {mse:.6f}")

# Calcolo degli MSE per ogni modello
mse_values = {}
for name, pred in preds.items():
    print(name)
    if name == 'LSTM':
        mse = mean_squared_error(Y_test_np[-len(pred):], pred)
    else:
        mse = mean_squared_error(Y_test_np, pred[:len(Y_test_np)])
    mse_values[name] = mse
    print(f"{name} MSE (completo o adattato): {mse:.6f}")

# Creazione di un DataFrame con i risultati
df_mse = pd.DataFrame(list(mse_values.items()), columns=['Model', 'MSE'])
df_mse['MSE'] = df_mse['MSE'].apply(lambda x: f"{x:.6f}")

# Creazione della tabella usando matplotlib
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_mse.values,
                 colLabels=df_mse.columns,
                 cellLoc='center',
                 loc='center')

plt.title("MSE Table")
plt.savefig("mse_table.png")  # Salva la tabella in formato PNG
plt.show()
window_size = 5  # Dimensione della finestra usata per generare le sequenze

# Creiamo un asse x per la serie di target "True" e per le previsioni degli altri modelli.
x_true = np.arange(len(Y_true_seq))

plt.figure(figsize=(15,6))
plt.plot(x_true, Y_true_seq, label="True", linewidth=2)

for name, pred in preds.items():
    if name == "LSTM":
        # Per LSTM, creiamo un asse x che tenga conto dello shift (window_size)
        x_lstm = np.arange(window_size, window_size + len(pred))
        plt.plot(x_lstm, pred, label=name)
    else:
        # Gli altri modelli sono allineati con x_true
        plt.plot( pred[-len(Y_test_seq):], label=name)

plt.legend()
plt.title("Forecast Comparison")
plt.xlabel("Indice Temporale")
plt.ylabel("Valore")
plt.grid(True)
plt.savefig("forecast_comparison.png")  # Salva il grafico in PNG
plt.show()


plt.figure(figsize=(10,6))
for name, pred in preds.items():
    plt.hist(pred[-len(Y_test_seq):] - Y_true_seq, bins=50, alpha=0.5, label=name)
plt.legend()
plt.title("Error Histogram")
plt.grid(True)
plt.savefig("error_histogram.png")  # Salva il grafico in PNG
plt.show()
