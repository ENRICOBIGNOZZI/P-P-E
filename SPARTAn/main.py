import numpy as np
import pandas as pd
from testing_function import adf_test
df=pd.read_csv('time_series_features.csv').set_index('date')


stationarity_results = {}
stationary_columns = []
for col in df.columns:
    series = df[col]
    p_value = adf_test(series)
    # Salviamo il risultato del test nel dizionario
    stationarity_results[col] = {"p_value": p_value, "stationary": p_value < 0.05}
    
    # Se la serie è stazionaria, salviamo il nome della colonna
    if p_value < 0.05:
        stationary_columns.append(col)

# Creazione di un nuovo DataFrame contenente solo le colonne stazionarie
df_stationary = df[stationary_columns]

# Stampa dei risultati dei test per ogni colonna
'''print("Risultati del test di Dickey-Fuller:")
for col, result in stationarity_results.items():
    if result["stationary"]:
        print(f"Colonna '{col}': stazionaria (p-value = {result['p_value']:.3f})")
    else:
        print(f"Colonna '{col}': NON stazionaria (p-value = {result['p_value']:.3f})")'''

df=df_stationary.copy()
forecasting=1
for i in range(1,forecasting,1):
    df[f'target_forecast{i}']=df['target'].shift(-i)
print(df)


