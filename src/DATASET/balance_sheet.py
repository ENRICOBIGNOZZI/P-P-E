import pandas as pd
from urllib.request import urlopen
import certifi
import json
#from DATASET_COMPONENT.functions import clusterKMeansTop
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm


def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)


def process_dataframe(url, has_date=False, rome_tz=None):
    data = get_jsonparsed_data(url)
    df = pd.DataFrame(data)
    if has_date and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt#.tz_localize('UTC').dt.tz_convert(rome_tz)
    return df

def rename_duplicates(df, index):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [f"{dup}_{index}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def merge_dataframes(dfs):
    merged_df = dfs[0]
    for i, df in enumerate(dfs[1:], start=1):
        #df = rename_duplicates(df, i)
        merged_df = merged_df.merge(df, on=['date', 'symbol', 'period', 'calendarYear'], how='outer')
    return merged_df

def balance_ticker_data(ticker,df_apple):
    
    YOUR_API_KEY = "d9574579c0afd2f75e2ee33ccd41c35e"
    dataframes = []
    
    urls = [
        f"https://financialmodelingprep.com/api/v3/financial-statement-full-as-reported/{ticker}?period=quarter&limit=100&apikey={YOUR_API_KEY}",
        f"https://financialmodelingprep.com/api/v3/income-statement-growth/{ticker}?period=quarter&apikey={YOUR_API_KEY}",
        f"https://financialmodelingprep.com/api/v3/cash-flow-statement-growth/{ticker}?period=quarter&apikey={YOUR_API_KEY}",
        f"https://financialmodelingprep.com/api/v3/balance-sheet-statement-growth/{ticker}?period=quarter&apikey={YOUR_API_KEY}",
        f"https://financialmodelingprep.com/api/v3/financial-growth/{ticker}?period=quarter&apikey={YOUR_API_KEY}"
    ]
    
    df_1 = process_dataframe(urls[0])
    df_1 = df_1.dropna(axis=1)
    df_1['date'] = process_dataframe(urls[1])['date']
    df_1['calendarYear'] = process_dataframe(urls[1])['calendarYear']
    for col in df_1.select_dtypes(include='number').columns:
        df_1[col]=df_1[col].pct_change()
    dataframes.append(df_1)
    
    for url in urls[1:]:
        df = process_dataframe(url)
        dataframes.append(df)
    
    merged_df = merge_dataframes(dataframes)
    merged_df = merged_df.dropna().sort_values(by='date')#.reset_index(drop=True)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep='first')]
    merged_df.set_index('date', inplace=True)
    merged_df=merged_df.select_dtypes(include='number')
    colonne_da_rimuovere = [column for column in merged_df.columns if merged_df[column].nunique() == 1]
    merged_df.drop(columns=colonne_da_rimuovere, inplace=True)
    
    df_originale=merged_df

    df_originale.index = pd.to_datetime(df_originale.index)
    df_apple.index = pd.to_datetime(df_apple.index)


    df_concatenato = pd.concat([df_apple,df_originale], axis=1)
    df_concatenato.index = pd.to_datetime(df_concatenato.index)
   
    df_concatenato.index = df_concatenato.index.tz_localize(None)
    df_apple.index = df_apple.index.tz_localize(None)

    df_concatenato.fillna(method='ffill', inplace=True)
    df_concatenato = df_concatenato[df_concatenato.index.isin(df_apple.index)]
  
    df_concatenato=df_concatenato.dropna()
    df=df_concatenato.copy()
    return df

   