from urllib.request import urlopen
import certifi
import json
import pandas as pd
from functools import reduce
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hmmlearn.hmm as hmm
#from DATASET_COMPONENT.functions import clusterKMeansTop

# Inserisci qui la tua API key
YOUR_API_KEY="d9574579c0afd2f75e2ee33ccd41c35e"



# Funzione per ottenere i dati JSON da un URL
def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

def process_dataframe(url, desc, has_date_column=False):

    differentiate = ["GDP", "realGDP", "nominalPotentialGDP", "realGDPPerCapita", "retailSales", "durableGoods", "totalNonfarmPayroll", "retailMoneyFunds", "industrialProductionTotalIndex", "CPI"]
    data = get_jsonparsed_data(url)
    df = pd.DataFrame(data)
    df[desc] = df['value']
    df.drop(columns = "value", inplace = True)
    df = df.iloc[::-1].reset_index(drop=True)
    df.set_index("date", inplace = True)
    if desc in differentiate:
      df[df.columns[0]] = df[df.columns[0]].pct_change()
    if desc == "inflationRate":
      df.index = pd.to_datetime(df.index)
      monthly_avg = df.resample('M').mean()
      first_day_next_month = monthly_avg.index + pd.offsets.MonthBegin(1)
      df = pd.DataFrame(monthly_avg.values, index=first_day_next_month, columns=['inflationRate'])
      df.index = df.index.strftime('%Y-%m-%d')
    return df


def create_dataframe():
    feature_list = ["GDP", "realGDP", "nominalPotentialGDP", "realGDPPerCapita", "federalFunds", "CPI",
    "inflationRate", "inflation", "retailSales", "consumerSentiment", "durableGoods",
    "unemploymentRate", "totalNonfarmPayroll", "initialClaims", "industrialProductionTotalIndex",
    "newPrivatelyOwnedHousingUnitsStartedTotalUnits", "totalVehicleSales", "retailMoneyFunds",
    "smoothedUSRecessionProbabilities", #"3MonthOr90DayRatesAndYieldsCertificatesOfDeposit",
    "commercialBankInterestRateOnCreditCardPlansAllAccounts", "30YearFixedRateMortgageAverage",
    "15YearFixedRateMortgageAverage"]


    urls = {}

    for feature in feature_list:
        urls[feature] = f"https://financialmodelingprep.com/api/v4/economic?name={feature}&apikey={YOUR_API_KEY}"

    # Liste per i dataframes con e senza colonne di date
    dataframes_without_dates = []

    for description, url in urls.items():
        df = process_dataframe(url, description)
        dataframes_without_dates.append(df)

    first_date = min(df.index.min() for df in dataframes_without_dates)
    last_date = max(df.index.max() for df in dataframes_without_dates)

    trimmed_dataframes = [df.loc[first_date:last_date] for df in dataframes_without_dates]
    merged_df = reduce(lambda left, right: pd.merge(left, right, how='outer', left_index=True, right_index=True), trimmed_dataframes)
    filled_df = merged_df.ffill().dropna()
    filled_df.index = pd.to_datetime(filled_df.index)

    return filled_df



    


