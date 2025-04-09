import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set fixed seed for reproducibility
np.random.seed(42)

# Parameters for the series
n = 5000  # total number of observations
cut_point = 3500  # time index at which the series becomes (almost) stationary
date_range = pd.date_range(start='2020-01-01', periods=n, freq='D')

# Segment 1: Non-stationary series for the first "cut_point" observations
n1 = cut_point
t1 = np.arange(n1)

# Linear trend
trend1 = np.linspace(0, 10, n1)

# Annual seasonality
seasonal1 = 5 * np.sin(2 * np.pi * t1 / 365)

# High frequency oscillation to make the series more "volatile"
high_freq1 = 2 * np.sin(2 * np.pi * 20 * t1 / n1)

# Random noise
noise1 = np.random.normal(0, 1.5, n1)

# Non-stationary segment
segment1 = trend1 + seasonal1 + high_freq1 + noise1

# Segment 2: Almost stationary series for the remaining observations
n2 = n - cut_point
t2 = np.arange(n2)

# To ensure continuity, the base of segment2 is set to the last value of segment1
base_value = segment1[-1]

# Low amplitude seasonality and less noise, without trend
seasonal2 = 0.5 * np.sin(2 * np.pi * t2 / 50)
noise2 = np.random.normal(0, 0.8, n2)
segment2 = base_value + seasonal2 + noise2

# Concatenating both segments to form the full time series
target = np.concatenate([segment1, segment2])

# Create DataFrame with the series and the time index
df = pd.DataFrame({'date': date_range, 'target': target})
df.set_index('date', inplace=True)

# ---------------------------- Feature Engineering: Create 100 features ----------------------------

features = pd.DataFrame(index=df.index)

# --- Lag features (10 features) ---
for lag in range(1, 20):
    features[f'lag_{lag}'] = df['target'].shift(lag)

# --- Rolling means for different window sizes (6 features) ---
for window in range(3, 50):
    features[f'rolling_mean_{window}'] = df['target'].rolling(window=window).mean()

# --- Rolling standard deviation for the same windows (6 features) ---
for window in range(3, 50):
    features[f'rolling_std_{window}'] = df['target'].rolling(window=window).std()

# --- Rolling min and max for 4 window sizes (8 features) ---
for window in range(3, 50):
    features[f'rolling_min_{window}'] = df['target'].rolling(window=window).min()
    features[f'rolling_max_{window}'] = df['target'].rolling(window=window).max()

# --- First and second differences (2 features) ---
features['diff_1'] = df['target'].diff(1)
features['diff_2'] = df['target'].diff(2)

# --- Time index as a continuous variable (1 feature) ---
features['time_index'] = np.arange(n)

# --- Day of week as numeric [0-6] (1 feature) ---
features['day_of_week'] = df.index.dayofweek

# --- One-hot encoding for day of week (7 features) ---
dow_series = pd.Series(df.index.dayofweek, index=df.index)
dow_dummies = pd.get_dummies(dow_series, prefix='dow')
features = pd.concat([features, dow_dummies], axis=1)

# --- One-hot encoding for month (12 features) ---
month_series = pd.Series(df.index.month, index=df.index)
month_dummies = pd.get_dummies(month_series, prefix='month')
features = pd.concat([features, month_dummies], axis=1)

# --- One-hot encoding for quarter (4 features) ---
quarter_series = pd.Series(df.index.quarter, index=df.index)
quarter_dummies = pd.get_dummies(quarter_series, prefix='quarter')
features = pd.concat([features, quarter_dummies], axis=1)

# --- Fourier terms to capture seasonality: 10 frequencies (20 features: sine and cosine) ---
for k in range(1, 11):
    features[f'fourier_sin_{k}'] = np.sin(2 * np.pi * k * features['time_index'] / n)
    features[f'fourier_cos_{k}'] = np.cos(2 * np.pi * k * features['time_index'] / n)

# --- Polynomial time features: quadratic and cubic (2 features) ---
features['time_index_sq'] = features['time_index'] ** 2
features['time_index_cu'] = features['time_index'] ** 3

# Up to this point: 10 (lag) + 6 (rolling_mean) + 6 (rolling_std) + 8 (rolling min/max) + 2 (diff) +
# 1 (time_index) + 1 (day_of_week) + 7 (dow one-hot) + 12 (month one-hot) + 4 (quarter one-hot) +
# 20 (Fourier) + 2 (polynomial) = 78 features

# --- Lag of the first difference: first 5 lags (5 features) ---
for lag in range(1, 6):
    features[f'diff1_lag_{lag}'] = features['diff_1'].shift(lag)

# Total so far: 78 + 5 = 83

# --- Rolling skewness and kurtosis for windows 3 and 5 (4 features) ---
for window in [3, 5]:
    features[f'rolling_skew_{window}'] = df['target'].rolling(window=window).skew()
    features[f'rolling_kurt_{window}'] = df['target'].rolling(window=window).kurt()

# Total: 83 + 4 = 87

# --- Exponential weighted moving averages (EWMA) for different spans (3 features) ---
for span in [5, 10, 20]:
    features[f'ewma_mean_{span}'] = df['target'].ewm(span=span, adjust=False).mean()

# Total: 87 + 3 = 90

# --- Rolling range: difference between rolling_max and rolling_min for windows 7 and 14 (2 features) ---
for window in [7, 14]:
    features[f'rolling_range_{window}'] = df['target'].rolling(window=window).max() - df['target'].rolling(window=window).min()

# Total: 90 + 2 = 92

# --- Lag ratios: current value/lag_1 and lag_1/lag_2 (2 features) ---
features['ratio_current_lag1'] = df['target'] / df['target'].shift(1)
features['ratio_lag1_lag2'] = df['target'].shift(1) / df['target'].shift(2)

# Total: 92 + 2 = 94

# --- Additional temporal features: cumulative percentage change and week of year (2 features) ---
features['pct_change_from_start'] = df['target'] / df['target'].iloc[0] - 1
features['week_of_year'] = df.index.isocalendar().week.astype(int)

# Total: 94 + 2 = 96

# --- Day of year (1 feature) ---
features['day_of_year'] = df.index.dayofyear

# --- Sine/Cosine transformations on day of year (2 features) ---
features['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
features['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)

# Total: 96 + 1 + 2 = 99

# --- Weekend indicator (1 feature) ---
features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

# Final total: 99 + 1 = 100

# Combine the target with the features for convenience
data = features.join(df['target'])

# ---------------------------- Visualization, CSV, and Image Saving ----------------------------

# Display preview of the DataFrame
print("Preview of the generated time series and features:")
print(data.head(15))
data=data.dropna(axis=1)
data=data.dropna()
#data=data.set_index('date')
# Plot the time series with the regime change line
plt.figure(figsize=(14, 5))
plt.plot(df.index, df['target'], label='Time Series')
plt.axvline(x=df.index[cut_point], color='red', linestyle='--', label='Regime Change')
plt.xlabel('Date')
plt.ylabel('Target')
plt.title('Time Series with Regime Change')
plt.legend()

# Save the plot as a PNG image
plt.savefig("time_series.png", dpi=300)
plt.show()

# Save the DataFrame as a CSV file
data.to_csv("time_series_features.csv")
print("The files 'time_series_features.csv' and 'time_series.png' have been saved successfully.")
