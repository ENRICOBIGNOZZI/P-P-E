
from statsmodels.tsa.stattools import adfuller
def adf_test(series, signif=0.05, verbose=False):
    result = adfuller(series.dropna(), autolag='AIC')
    test_statistic, p_value, n_lags, n_obs, crit_values, icbest = result
    if verbose:
        print(f"Statistic: {test_statistic:.3f}")
        print(f"P-Value: {p_value:.3f}")
        print(f"#Lags Used: {n_lags}")
        print(f"Number of Observations Used: {n_obs}")
        print("Critical Values:")
        for key, value in crit_values.items():
            print(f"    {key}: {value:.3f}")
    return p_value