import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Modifica SPARTAn con peso dipendente dal tempo
def spartan_regression_fast_time(
    X, Y, K=10,
    epsilon_d=1e-4, epsilon_r=1e-5, epsilon_l2=1e-4,
    epochs=1000, lr=1e-2, verbose=True
):
    T, D = X.shape
    T2, M = Y.shape
    assert T == T2, f"Mismatch T in X({T}) vs Y({T2})"

    Lambda = nn.Parameter(torch.randn(K, D, M)*0.01)
    C      = nn.Parameter(torch.randn(K, D)*0.01)
    w_0    = nn.Parameter(torch.zeros(T, D))  # ora dipende dal tempo
    gamma_0= nn.Parameter(torch.randn(T, K)*0.01)

    optimizer = optim.Adam([Lambda, C, w_0, gamma_0], lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        w = torch.softmax(w_0, dim=1)  # (T,D), dipendenza temporale
        gamma = torch.softmax(gamma_0, dim=1)  # (T,K)

        # Discretizzazione
        diff = X.unsqueeze(1) - C.unsqueeze(0)  # (T,K,D)
        w_exp = w.unsqueeze(1)  # (T,1,D)
        gamma_exp = gamma.unsqueeze(2)  # (T,K,1)
        loss_discret = torch.sum(w_exp * gamma_exp * diff**2) / (T*K)

        loss_entropy = epsilon_d * torch.sum(w * torch.log(w + 1e-12))
        loss_l2 = epsilon_l2 * torch.sum(Lambda**2)

        # Regression con peso tempo-dipendente
        dot = torch.einsum('td,td,kdm->tkm', X, w, Lambda)
        resid = (Y.unsqueeze(1) - dot)**2  # (T,K,M)
        loss_reg = epsilon_r * torch.sum(gamma_exp * resid) / (T*M)

        total_loss = loss_discret + loss_entropy + loss_l2 + loss_reg
        total_loss.backward()
        optimizer.step()

        if verbose and epoch % 100 == 0:
            print(f"[Epoch {epoch}] Loss: {total_loss.item():.6f}")

    with torch.no_grad():
        w_out = torch.softmax(w_0, dim=1)
        gamma_out = torch.softmax(gamma_0, dim=1)

    return Lambda.detach(), gamma_out, w_out, C.detach()

# Forecast aggiornato per il peso temporale
def forecast_y_time(X_new, Lambda, w_prob, C):
    F, D = X_new.shape
    K, D2, M = Lambda.shape
    assert D == D2, "Dimensioni incoerenti!"

    distances = []
    for i in range(F):
        diff = X_new[i] - C
        dist_i = torch.sum(w_prob[i]*(diff**2), dim=1)
        distances.append(-dist_i)
    distances = torch.stack(distances, dim=0)
    gamma_forecast = torch.softmax(distances, dim=1)

    preds_k = torch.stack([torch.einsum('d,dm,fd->fm', w_prob.mean(0), Lambda[k_], X_new) for k_ in range(K)], dim=1)
    Y_hat = torch.sum(gamma_forecast.unsqueeze(2) * preds_k, dim=1)
    return Y_hat
'''import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generazione dati sintetici
T, D, M = 500, 10, 1
np.random.seed(0)
torch.manual_seed(0)

X_np = np.zeros((T, D), dtype=np.float32)
Y_np = np.zeros((T, M), dtype=np.float32)

for t in range(T):
    for d in range(D):
        X_np[t, d] = np.sin((t + d) / 25) + np.random.normal(0, 0.05)
    Y_np[t, 0] = np.sin((t + 1) / 25) + np.random.normal(0, 0.05)

X_torch = torch.from_numpy(X_np)
Y_torch = torch.from_numpy(Y_np)

# Applicazione modello SPARTAn con peso dipendente dal tempo
Lambda, gamma_prob, w_prob, C = spartan_regression_fast(
    X_torch,
    Y_torch,
    K=1,
    epsilon_d=0.01,
    epsilon_r=1,
    epsilon_l2=0.000,
    epochs=10000,
    lr=0.05,
    verbose=True
)

# Forecast con il modello addestrato
Y_hat_torch = forecast_y(X_torch, Lambda, w_prob, C)
Y_hat_np = Y_hat_torch.detach().numpy().reshape(-1)

# Visualizzazione risultati
plt.figure(figsize=(12,6))
plt.plot(Y_np, label="True Series", linewidth=2)
plt.plot(Y_hat_np, label="Forecast (SPARTAn Temporal)", linestyle='--', linewidth=2)
plt.title("Forecast con SPARTAn (Peso Temporale)")
plt.legend()
plt.grid(True)
plt.show()

# Istogramma degli errori
error_np = Y_hat_np - Y_np.flatten()
plt.figure(figsize=(8,5))
plt.hist(error_np, bins=50, color='skyblue', edgecolor='black')
plt.title("Histogram of Errors")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()'''