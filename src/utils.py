import numpy as np

def make_positive_definite(matrix):
    """
    Rende una matrice definita positiva e simmetrica per l'uso in CVXPY.
    """
    matrix = np.array(matrix, dtype=float)  # Convertiamo a float
    matrix = (matrix + matrix.T) / 2  # Forziamo la simmetria
    
    try:
        np.linalg.cholesky(matrix)
        return matrix  
    except np.linalg.LinAlgError:
        pass  

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-3)  # Forziamo valori positivi
    matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return (matrix + matrix.T) / 2
def estimate_covariance_from_returns(predicted_returns: np.ndarray):
    """
    Stima la matrice di covarianza come il prodotto esterno del vettore dei rendimenti predetti.
    """
    predicted_returns = predicted_returns.reshape(-1, 1)  # Vettore colonna
    covariance_matrix = predicted_returns @ predicted_returns.T  # Prodotto esterno
    return covariance_matrix
