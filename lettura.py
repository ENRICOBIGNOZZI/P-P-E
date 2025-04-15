import scipy.io
import pandas as pd

# Carica il file .mat
mat_data = scipy.io.loadmat('/Users/enrico/PHD/ECMWF_EU_temperatures.mat')

# Visualizza le chiavi principali del dizionario caricato
print("Chiavi del file .mat:", mat_data.keys())

# Usa la variabile 'X' presente nel file
if 'X' in mat_data:
    data = mat_data['X']
    # Visualizza le dimensioni dell'array
    print("Dimensioni dell'array 'X':", data.shape)
    # Visualizza i primi 5 valori (se array numerico)
    print("Primi 5 valori:", data.flatten())
    
    # Converti l'array in un DataFrame
    df_data = pd.DataFrame(data)
    
    # Mostra le prime 5 righe del DataFrame
    print("Prime 5 righe del DataFrame:")
    print(df_data)
else:
    print("La variabile 'X' non è presente nel file .mat")

