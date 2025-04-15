def rbf_kernel(X1, X2, gamma=0.5):
    X1_sq = (X1 ** 2).sum(dim=1).view(-1, 1)  # shape: (n1, 1)
    X2_sq = (X2 ** 2).sum(dim=1).view(1, -1)  # shape: (1, n2)
    dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T   # shape: (n1, n2)
    return torch.exp(-gamma * dist_sq)


def model_f(x, X_train, alpha, gamma=0.5):
    # x: (m x D)
    # X_train: (n x D)
    # alpha: (n x 1)
    # Return f(x): (m x 1)
    K_xX = rbf_kernel(x, X_train, gamma=gamma)  # (m x n)
    return K_xX @ alpha  # (m x 1)

def partial_derivative(x, d, X_train, alpha, gamma=0.5):
    K_xX = rbf_kernel(x, X_train, gamma=gamma) 
    xd = x[:, d].unsqueeze(1)  # shape (m x 1)
    X_train_d = X_train[:, d].unsqueeze(0)  # shape (1 x n)
    # broadcast to (m x n)
    diff_d = xd - X_train_d  # (m x n)
    derivative_val = -2 * gamma * diff_d * K_xX  # (m x n)
    return derivative_val @ alpha
import torch
import numpy as np
import matplotlib.pyplot as plt
def main():
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
    lambda_ = 1e-2


    n_train = 800  # for example, first 800 for training, last 200 for testing

    X_train = X_torch[:n_train]
    Y_train = Y_torch[:n_train]
    X_test  = X_torch[n_train:]
    Y_test  = Y_torch[n_train:]


    K_train = rbf_kernel(X_train, X_train, gamma=0.5)

    n_tr = X_train.shape[0]
    K_train_reg = K_train + lambda_ * torch.eye(n_tr)

    # Solve for alpha in: (K + lambda I) alpha = Y
    alpha = torch.linalg.solve(K_train_reg, Y_train)
    Y_train_pred = K_train @ alpha
    mse_train = torch.mean((Y_train_pred - Y_train)**2)
    print("Train MSE:", mse_train.item())
    K_test_train = rbf_kernel(X_test, X_train, gamma=0.5)
    Y_test_pred = K_test_train @ alpha
    mse_test = torch.mean((Y_test_pred - Y_test)**2)
    print("Test MSE:", mse_test.item())


    Y_hat_torch = Y_test_pred
    Y_hat_np = Y_hat_torch.detach().numpy().reshape(-1)

    Y_np_flat = Y_test.detach().numpy().reshape(-1)
    mse_train = np.mean((Y_hat_np - Y_np_flat) ** 2)
    print(f"MSE train = {mse_train:.6f}")

    '''plt.figure()
    plt.plot(Y_np_flat, label="True series")
    plt.plot(Y_hat_np, label="Forecast (Kernel)", linestyle='--')
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


    eigenvals, eigenvecs = torch.linalg.eigh(K_train)
    idx_desc = torch.argsort(eigenvals, descending=True)
    evals_desc = eigenvals[idx_desc]
    evecs_desc = eigenvecs[:, idx_desc]
    plt.plot(evals_desc.detach().cpu().numpy())
    plt.title("Eigenvalues in descending order")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.show()



    num_evecs_to_plot = 5  # ad es., i primi 5
    plt.figure()
    for i in range(num_evecs_to_plot):
        # evecs_desc[:, i] è il (i)-esimo autovettore in ordine decrescente
        plt.plot(evecs_desc[:, i].detach().cpu().numpy(), label=f"Eigenvec {i}")

    plt.legend()
    plt.title("First 5 eigenvectors")
    plt.xlabel("Training sample index")
    plt.ylabel("Value")
    plt.show()'''
    eigenvals, eigenvecs = torch.linalg.eigh(K_train)
    idx_desc = torch.argsort(eigenvals, descending=True)
    evals_desc = eigenvals[idx_desc]
    evecs_desc = eigenvecs[:, idx_desc]



    idx_desc = torch.argsort(eigenvals, descending=True)
    evals_desc = eigenvals[idx_desc]
    evecs_desc = eigenvecs[:, idx_desc]

    # 2) Take the largest eigenvalue/vector
    lambda1 = evals_desc[0]
    u1 = evecs_desc[:, 0].reshape(-1,1)  # (n,1)

    # 3) Build rank-1 approximation
    K_rank1 = lambda1 * (u1 @ u1.T)  # (n x n)

    # 4) Solve for alpha^(1) in (K_rank1 + lambda I)*alpha = y
    lambda_ = 1e-2  # reg param
    n = K_train.shape[0]
    K_rank1_reg = K_rank1 + lambda_ * torch.eye(n)

    alpha_rank1 = torch.linalg.solve(K_rank1_reg, Y_train)
    K_test_train = rbf_kernel(X_test, X_train)  # (m x n)
    Y_test_pred_rank1 = K_test_train @ alpha_rank1  # (m x 1)

    # === 7) Evaluate error
    mse_rank1 = torch.mean((Y_test_pred_rank1 - Y_test)**2)
    print("Test MSE (rank-1 approximation):", mse_rank1.item())

    # === 8) Optional: Compare with the full kernel ridge solution
    # Just for reference, let's do the full KRR once (no rank-1 approximation)
    K_train_reg = K_train + lambda_ * torch.eye(n_train)
    alpha_full = torch.linalg.solve(K_train_reg, Y_train)
    K_test_train_full = rbf_kernel(X_test, X_train, gamma=0.5)
    Y_test_pred_full = K_test_train_full @ alpha_full
    mse_full = torch.mean((Y_test_pred_full - Y_test)**2)
    print("Test MSE (full KRR):", mse_full.item())

    # === 9) Visualize the test predictions vs. true values
    Y_test_np = Y_test.detach().cpu().numpy().flatten()
    Y_pred_rank1_np = Y_test_pred_rank1.detach().cpu().numpy().flatten()
    Y_pred_full_np = Y_test_pred_full.detach().cpu().numpy().flatten()

    plt.plot(Y_test_np, label="True", linewidth=1.0)
    plt.plot(Y_pred_rank1_np, label="Rank-1 Pred", linestyle="--")
    plt.plot(Y_pred_full_np, label="Full KRR Pred", linestyle=":")
    plt.legend()
    plt.title("Comparison: Rank-1 KRR vs. Full KRR")
    plt.xlabel("Test sample index")
    plt.ylabel("Output value")
    plt.show()
    

    # 5) Then alpha_rank1 is your partial solution. For any x,
    #    prediction ~ sum_i alpha_rank1[i]*K(x, x_i).
    #    We'll do it in vector form for test data:
    # Suppose X_test is (m x D)


    gamma = 0.5
    m = X_test.shape[0]
    D = X_test.shape[1]

    grad_importance = torch.zeros(D)
    for d in range(D):
        # partial derivatives for each sample in X_test
        pd = partial_derivative(X_test, d, X_train, alpha, gamma=gamma)  # (m x 1)
        # measure magnitude
        grad_importance[d] = pd.abs().mean()

    print("Gradient-based importance:", grad_importance)
    

    # Calcolo degli autovalori e autovettori
    eigenvals, eigenvecs = torch.linalg.eigh(K_train)
    idx_desc = torch.argsort(eigenvals, descending=True)
    evals_desc = eigenvals[idx_desc]
    evecs_desc = eigenvecs[:, idx_desc]

    # Plot degli autovalori
    plt.figure()
    plt.plot(evals_desc.detach().cpu().numpy())
    plt.title("Autovalori in ordine decrescente")
    plt.xlabel("Indice")
    plt.ylabel("Autovalore")
    plt.savefig("autovalori.png")  # Salva il grafico come PNG
    plt.close()

    # Plot dei primi 5 autovettori
    num_evecs_to_plot = 5  # Numero di autovettori da plottare
    plt.figure()
    for i in range(num_evecs_to_plot):
        plt.plot(evecs_desc[:, i].detach().cpu().numpy(), label=f"Autovettore {i+1}")

    plt.legend()
    plt.title("Primi 5 autovettori")
    plt.xlabel("Indice del campione di training")
    plt.ylabel("Valore")
    plt.savefig("autovettori.png")  # Salva il grafico come PNG
    plt.close()
if __name__ == "__main__":
    main()
