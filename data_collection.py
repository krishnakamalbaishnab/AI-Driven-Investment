import requests
import pandas as pd
from datetime import datetime

def get_historical_data(crypto_id='bitcoin', days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extracting prices and dates
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    
    # Convert timestamp to readable date
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.drop('timestamp', axis=1, inplace=True)
    
    return df

if __name__ == "__main__":
    crypto_data = get_historical_data('bitcoin', 365)
    crypto_data.to_csv('crypto_data.csv', index=False)
