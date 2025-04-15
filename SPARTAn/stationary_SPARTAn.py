import torch
import torch.nn as nn
import torch.optim as optim



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def spartan_regression_fast(
    X,                   # shape: (T,D)
    Y,                   # shape: (T,M)
    K=10,
    epsilon_d=1e-4,
    epsilon_r=1e-5,
    epsilon_l2=1e-4,
    epochs=1000,
    lr=1e-2,
    verbose=True
):
    T, D = X.shape
    T2, M = Y.shape
    assert T==T2, f"Mismatch T in X({T}) vs Y({T2})"

    Lambda = nn.Parameter(torch.randn(K, D, M)*0.01)
    C      = nn.Parameter(torch.randn(K, D)*0.01)
    w_0    = nn.Parameter(torch.zeros(D))
    gamma_0= nn.Parameter(torch.randn(T, K)*0.01)

    optimizer = optim.Adam([Lambda,C,w_0,gamma_0], lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        w = torch.softmax(w_0, dim=0)          # (D,)
        gamma = torch.softmax(gamma_0, dim=1)  # (T,K)

        # 1) Discretizzazione
        diff = X.unsqueeze(1) - C.unsqueeze(0)       # (T,K,D)
        w_exp = w.view(1,1,D)                        # (1,1,D)
        gamma_exp = gamma.view(T,K,1)                # (T,K,1)
        loss_discret = torch.sum(w_exp * gamma_exp * diff**2) / (T*K)

        loss_entropy = epsilon_d * torch.sum(w * torch.log(w + 1e-12))

        loss_l2 = epsilon_l2 * torch.sum(Lambda**2)

        # dot(t,k,m) = sum_d [ w[d]*Lambda[k,d,m]*X[t,d] ]
        dot = torch.einsum('td,d,kdm->tkm', X, w, Lambda)  # (T,K,M)
        gamma_exp2 = gamma.unsqueeze(2)                   # (T,K,1)
        resid = (Y.unsqueeze(1) - dot)**2                 # (T,K,M)
        loss_reg = epsilon_r * torch.sum(gamma_exp2 * resid) / (T*M)

        total_loss = loss_discret + loss_entropy + loss_l2 + loss_reg
        total_loss.backward()
        optimizer.step()

        if verbose and (epoch % 100 == 0):
            print(f"[Epoch {epoch}] Loss: {total_loss.item():.6f}")

    with torch.no_grad():
        w_out = torch.softmax(w_0, dim=0)
        gamma_out = torch.softmax(gamma_0, dim=1)

    return Lambda.detach(), gamma_out, w_out, C.detach()

def forecast_y(X_new, Lambda, w_prob, C):
    F, D = X_new.shape
    K, D2, M = Lambda.shape
    assert D==D2, "Dimensioni incoerenti tra X e Lambda!"
    distances = []
    for i in range(F):
        diff = X_new[i] - C 
        dist_i = torch.sum(w_prob*(diff**2), dim=1)  
        distances.append(-dist_i)
    distances = torch.stack(distances, dim=0)  
    gamma_forecast = torch.softmax(distances, dim=1)  

    
    preds_k = []
    for k_ in range(K):
        lam_k = Lambda[k_]          
        dot_k = torch.einsum('d,dm,fd->fm', w_prob, lam_k, X_new)
        preds_k.append(dot_k)        
    preds_k = torch.stack(preds_k, dim=1)  
    gamma_exp = gamma_forecast.unsqueeze(2) 
    Y_hat = torch.sum(gamma_exp * preds_k, dim=1) 
    return Y_hat

if __name__ == "__main__":
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

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

    plt.figure()
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
    plt.show()
