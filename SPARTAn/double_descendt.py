import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ====================================================
# 1. Caricamento e preparazione del dataset ECMWF
# ====================================================
mat_data = scipy.io.loadmat('/Users/enrico/PHD/ECMWF_EU_temperatures.mat')
if 'X' in mat_data:
    data = mat_data['X']
    print("Shape of X:", data.shape)
    # Converte l'array in un DataFrame
    df_data = pd.DataFrame(data)
    print("Dataset completo:")
    print(df_data)
else:
    raise ValueError("La variabile 'X' non è presente nel file .mat")
df_data=df_data.iloc[:,:30]
# Usiamo tutte le righe disponibili e impostiamo il target come il valore della prima feature al passo successivo
df = df_data.copy()
df["target"] = df.iloc[:, 0].shift(-1)
df = df.dropna()

# Selezione delle feature e del target
X_np = df.drop(columns=["target"]).values.astype(np.float32)
Y_np = df["target"].values.reshape(-1, 1).astype(np.float32)

# Scaling delle feature
scaler = StandardScaler()
X_np_scaled = scaler.fit_transform(X_np)

# Conversione in tensori PyTorch
X_tensor = torch.tensor(X_np_scaled)
Y_tensor = torch.tensor(Y_np)

# Suddivisione in training e test (per questo esempio usiamo 50% training e 50% test)
n_samples = X_tensor.shape[0]
n_train = int(0.70 * n_samples)
X_train = X_tensor[:n_train]
Y_train = Y_tensor[:n_train]
X_test = X_tensor[n_train:]
Y_test = Y_tensor[n_train:]

# Dimensione delle feature
D = X_train.shape[1]
print("Numero di feature (D):", D)

# ====================================================
# 2. Definizione del modello SPARTAn non lineare
# ====================================================
class NonlinearSPARTAnRegressor(nn.Module):
    def __init__(self, input_dim, K):
        """
        Parametri del modello:
         - u ∈ ℝ^(input_dim): da cui si ottengono i pesi delle feature: w = softmax(u)
         - beta ∈ ℝ^(K x input_dim): coefficienti locali per ciascun box
         - C ∈ ℝ^(K x input_dim): centri dei box per la discretizzazione dello spazio delle feature
         
        Il numero totale di parametri (senza eventuali bias) è:
             P(K) = input_dim + 2 * K * input_dim
        """
        super(NonlinearSPARTAnRegressor, self).__init__()
        self.input_dim = input_dim
        self.K = K
        # Parametro non vincolato per calcolare w
        self.u = nn.Parameter(torch.zeros(input_dim))
        # Coefficienti locali per ciascun box
        self.beta = nn.Parameter(torch.randn(K, input_dim))
        # Centri dei box
        self.C = nn.Parameter(torch.randn(K, input_dim))
        
    def forward(self, X):
        """
        X: batch di input, shape (batch_size, input_dim)
        Restituisce:
         - y_pred: predizione finale (batch_size, 1)
         - w: vettore dei pesi delle feature, shape (input_dim,)
         - gamma: soft-assignment per ciascun box, shape (batch_size, K)
         - theta: coefficienti locali pesati (K, input_dim)
        """
        # Calcola w da u tramite softmax
        w = torch.softmax(self.u, dim=0)  # shape (input_dim,)
        # Calcola i coefficienti locali per ciascun box: θₖ = w ⊙ betaₖ
        theta = self.beta * w.unsqueeze(0)  # shape (K, input_dim)
        
        # Calcola l'output locale per ciascun box: outₖ = x · θₖ per ogni sample
        out = X.matmul(theta.T)  # shape (batch_size, K)
        
        # Calcola le distanze euclidee al quadrato tra ogni sample e il centro di ciascun box
        dists = torch.sum((X.unsqueeze(1) - self.C.unsqueeze(0)) ** 2, dim=2)  # shape (batch_size, K)
        
        # Calcola la soft-assignment: gamma = softmax(-dists)
        gamma = torch.softmax(-dists, dim=1)  # shape (batch_size, K)
        
        # Previsione finale: media pesata degli output locali
        y_pred = torch.sum(gamma * out, dim=1, keepdim=True)  # shape (batch_size, 1)
        return y_pred, w, gamma, theta

def entropy_penalty(w):
    # Penalizzazione entropica per evitare distribuzioni troppo piatte
    return torch.sum(w * torch.log(w + 1e-8))

# ====================================================
# 3. Funzione di training per il modello non lineare SPARTAn
# ====================================================
def train_spartan_nonlinear(model, X_train, Y_train, X_test, Y_test,
                            eps_w=0.1, eps_beta=1.0, num_epochs=3000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred, w, gamma, theta = model(X_train)
        loss_mse = mse_loss(y_pred, Y_train)
        loss_entropy = entropy_penalty(w)
        loss_l2 = torch.sum(model.beta ** 2)  # regolarizzazione sui coefficienti locali
        
        loss = loss_mse + eps_w * loss_entropy #+ eps_beta * loss_l2
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                y_test_pred, _, _, _ = model(X_test)
                test_loss = mse_loss(y_test_pred, Y_test).item()
            test_losses.append(test_loss)
    return train_losses, test_losses

# ====================================================
# 4. Esperimenti: variazione di K e calcolo del numero di parametri
# ====================================================
# In questa parte variare K (da 1 a 20) per studiare come varia il numero di parametri:
#      P = D + 2*K*D
K_values = [1,2,5,8,10,50,200,300,400,800,1000]

train_loss_list = []
test_loss_list = []
param_count_list = []

eps_w = 0.001
eps_beta = 1.0
num_epochs = 20000
lr = 0.01

for K in K_values:
    model = NonlinearSPARTAnRegressor(input_dim=D, K=K)
    train_losses, test_losses = train_spartan_nonlinear(model, X_train, Y_train, X_test, Y_test,
                                                         eps_w=eps_w, eps_beta=eps_beta,
                                                         num_epochs=num_epochs, lr=lr)
    final_train_loss = train_losses[-1]
    final_test_loss = test_losses[-1] if len(test_losses) > 0 else None
    param_count = D + 2 * K * D
    param_count_list.append(param_count)
    train_loss_list.append(final_train_loss)
    test_loss_list.append(final_test_loss)
    
    print(f"K = {K:2d} | Parametri totali = {param_count:4d} | Train Loss = {final_train_loss:.4f} | Test Loss = {final_test_loss:.4f}")

plt.figure(figsize=(10,6))
plt.plot(param_count_list, train_loss_list, label="Train Loss", marker="o")
plt.plot(param_count_list, test_loss_list, label="Test Loss", marker="o")
plt.xlabel("Numero totale di parametri (P = D + 2*K*D)")
plt.ylabel("Loss (MSE + penalità)")
plt.title("Double Descent nel modello SPARTAn non lineare al variare di K")
plt.legend()
plt.grid(True)
plt.show()

# ====================================================
# 5. Visualizzazione del Forecasting sul Test set
# ====================================================
# Per visualizzare il forecasting scegliamo un valore di K (ad es. K = 10)
K_forecast = 2
model_forecast = NonlinearSPARTAnRegressor(input_dim=D, K=K_forecast)

# Addestriamo il modello per il forecasting
train_losses_forecast, test_losses_forecast = train_spartan_nonlinear(model_forecast, X_train, Y_train, X_test, Y_test,
                                                                       eps_w=eps_w, eps_beta=eps_beta,
                                                                       num_epochs=num_epochs, lr=lr)
                                                                       
# Effettua le previsioni sul test set
model_forecast.eval()
with torch.no_grad():
    y_pred_forecast, _, _, _ = model_forecast(X_test)
    
# Convertiamo le previsioni e il target in array NumPy per la visualizzazione
y_pred_forecast_np = y_pred_forecast.cpu().numpy().flatten()
y_test_np = Y_test.cpu().numpy().flatten()

# Visualizza il forecasting: confronto tra target reale e previsione
plt.figure(figsize=(12,6))
plt.plot(y_test_np, label='Target Reale', marker='o', linestyle='-')
plt.plot(y_pred_forecast_np, label='Forecast', marker='x', linestyle='--')
plt.xlabel("Indice dei campioni del Test set")
plt.ylabel("Valore del Target")
plt.title("Forecasting sul Test set (K = {})".format(K_forecast))
plt.legend()
plt.grid(True)
plt.show()
