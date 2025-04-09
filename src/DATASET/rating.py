import pandas as pd
from urllib.request import urlopen
import certifi
import json
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder




def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

def process_dataframe(url):
    data = get_jsonparsed_data(url)
    df = pd.DataFrame(data)
    
    # Convert 'publishedDate' to datetime and format it
    if 'publishedDate' in df.columns:
        df['publishedDate'] = pd.to_datetime(df['publishedDate'])
        df['publishedDate'] = df['publishedDate'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df

def analyze_rating(df_finale,symbol):
    #df_finale.index = pd.to_datetime(df_finale.index)  # Imposta 'data' come indice e convertila in datetime
    #df_finale.index.name = 'date'  # Rinomina l'indice in 'date'

    # Rimuovere la colonna 'data' poiché ora è l'indice
    #df_finale = df_finale.drop(columns=['data'])

    
    api_key= "d9574579c0afd2f75e2ee33ccd41c35e"
    # Create the URL using the provided API key and symbol
    url = f"https://financialmodelingprep.com/api/v3/historical-rating/{symbol}?apikey={api_key}"
    
    # Process the DataFrame
    df = process_dataframe(url)

    # Specify columns to drop
    columns_to_drop = ['symbol']  # Add any additional columns as necessary
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Drop NaN values and sort by 'date'
    df = df.dropna().sort_values(by='date').reset_index(drop=True)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    df.set_index('date', inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column])

    # Ensure all data is in integer format
    df = df.astype(int)
    
    # Fit the HMM model
    df=df.dropna()
    df.index = pd.to_datetime(df.index)

    # Ora che l'indice è un DatetimeIndex, puoi normalizzarlo
    df.index = df.index.normalize()

    df_concatenato = pd.concat([df_finale,df], axis=1)

    df_concatenato.index = pd.to_datetime(df_concatenato.index)
   
    #df_concatenato.index = df_concatenato.index.tz_localize(None)
    

    df_concatenato.fillna(method='ffill', inplace=True)
    df_concatenato = df_concatenato[df_concatenato.index.isin(df_finale.index)]
  
    df_concatenato=df_concatenato.dropna()
    df=df_concatenato.copy()
    #df['states_balance_sheet']=df['states_balance_sheet'].astype(int)
    return df#['states']



