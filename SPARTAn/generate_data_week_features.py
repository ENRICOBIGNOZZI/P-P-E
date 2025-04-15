import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Imposta il seme per la riproducibilità
np.random.seed(0)

# Numero di osservazioni
n = 500

# Crea una sequenza di date giornaliere
dates = pd.date_range(start='2020-01-01', periods=n, freq='D')

# Costruisci le componenti della serie:
# 1. Trend: una crescita lineare
trend = np.linspace(0, 10, n)

# 2. Stagionalità: funzione sinusoidale per modellare la stagionalità
seasonal = 5 * np.sin(np.linspace(0, 3 * np.pi, n))

# 3. Rumore: componente casuale per rendere i dati più realistici
noise = np.random.normal(0, 1, n)

# Definisci il target come combinazione di trend, stagionalità e rumore
target = trend + seasonal + noise

# Creazione della feature a forte prevedibilità:
# Utilizziamo il valore ritardato (lag 1) del target.
strong_lag = np.roll(target, 1)
strong_lag[0] = target[0]  # il primo elemento non ha precedente, lo riempiamo col valore corrente

# Creazione delle feature a debole prevedibilità:
# weak_noise: combinazione di rumore esistente e ulteriore rumore casuale
weak_noise = noise + np.random.normal(0, 1, n)

# weak_trend: utilizza il trend ma aggiunge un livello di rumore più elevato
weak_trend = trend + np.random.normal(0, 2, n)

# Crea il DataFrame con il dataset simulato
df = pd.DataFrame({
    'date': dates,
    'target': target,
    'strong_lag': strong_lag,
    'weak_noise': weak_noise,
    'weak_trend': weak_trend
})

# Visualizza le prime righe del dataset
print(df.head())

# Salva il DataFrame in un file CSV (senza l'indice)
df.to_csv("weak_features_timeseries_data.csv", index=False)
print("Dati salvati su timeseries_data.csv")

# Opzionale: Visualizza il grafico del target e della feature a forte prevedibilità
plt.figure(figsize=(12,6))
plt.plot(df['date'], df['target'], label='Target')
plt.plot(df['date'], df['strong_lag'], label='Strong Lag', alpha=0.7)
plt.xlabel('Data')
plt.ylabel('Valore')
plt.title('Serie Temporale: Target vs Strong Lag')
plt.legend()
plt.show()

# Opzionale: Visualizza il grafico delle features a debole prevedibilità
plt.figure(figsize=(12,6))
plt.plot(df['date'], df['weak_noise'], label='Weak Noise')
plt.plot(df['date'], df['weak_trend'], label='Weak Trend', alpha=0.7)
plt.xlabel('Data')
plt.ylabel('Valore')
plt.title('Serie Temporale: Features a Debole Prevedibilità')
plt.legend()
plt.show()
