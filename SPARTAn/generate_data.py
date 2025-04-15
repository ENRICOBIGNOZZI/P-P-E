import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Impostazioni iniziali
np.random.seed(42)
n = 10_000
t = np.linspace(0, 4 * np.pi, n)

# Cambi di frequenza e ampiezza (5 cambi)
frequencies = [1, 3, 5, 2, 4]
amplitudes = [1, 0.5, 1.5, 1, 2]
signal = np.zeros(n)
segment_length = n // len(frequencies)

for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
    start = i * segment_length
    end = start + segment_length
    signal[start:end] = amp * np.sin(freq * t[start:end])

# Features utili
features_useful = {
    'signal': signal,
    'lag_1': np.roll(signal, 1),
    'lag_2': np.roll(signal, 2),
    'signal_squared': signal ** 2,
    'rolling_mean_5': pd.Series(signal).rolling(5).mean().fillna(0),
    'rolling_std_5': pd.Series(signal).rolling(5).std().fillna(0),
    'ewma_span10': pd.Series(signal).ewm(span=10, adjust=False).mean(),
    'fourier_sin': np.sin(2 * np.pi * t / max(t)),
    'fourier_cos': np.cos(2 * np.pi * t / max(t)),
    'time_squared': t ** 2,
}

# Features di rumore casuale
num_noise_features = 10 - len(features_useful)
features_noise = {
    f'noise_{i}': np.random.normal(0, 1, n)
    for i in range(num_noise_features)
}

# Unire tutte le features
features = {**features_useful, **features_noise}

df = pd.DataFrame(features)

# Visualizzazione del segnale
plt.figure(figsize=(15, 5))
plt.plot(t, signal, label="Sinusoide con variazioni")
plt.xlabel("Tempo")
plt.ylabel("Ampiezza")
plt.title("Segnale sinusoidale con cambi di frequenza e ampiezza")
plt.legend()
plt.grid(True)
plt.show()

# Salvare il DataFrame in CSV
df.to_csv("sinusoidal_features.csv", index=False)

# Visualizzare il DataFrame finale
print(df)
