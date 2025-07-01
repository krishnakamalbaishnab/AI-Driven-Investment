import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataCollectionConfig:
    BASE_URL = "https://api.coingecko.com/api/v3"
    DEFAULT_CRYPTO = 'bitcoin'
    DEFAULT_DAYS = 365
    DEFAULT_CURRENCY = 'usd'
    OUTPUT_FILE = 'crypto_data.csv'
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    REQUEST_TIMEOUT = 30  # seconds

class CryptoDataCollector:
    def __init__(self, config: DataCollectionConfig = None):
        self.config = config or DataCollectionConfig()
        self.session = requests.Session()
        
        # Set user agent to avoid rate limiting
        self.session.headers.update({
            'User-Agent': 'AI-Driven-Investment/1.0 (Educational Project)'
        })
    
    def get_historical_data(self, 
                          crypto_id: str = None, 
                          days: int = None, 
                          vs_currency: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical cryptocurrency data from CoinGecko API.
        
        Args:
            crypto_id: Cryptocurrency ID (e.g., 'bitcoin', 'ethereum')
            days: Number of days of historical data
            vs_currency: Currency to compare against (default: 'usd')
            
        Returns:
            DataFrame with historical price data or None if failed
        """
        crypto_id = crypto_id or self.config.DEFAULT_CRYPTO
        days = days or self.config.DEFAULT_DAYS
        vs_currency = vs_currency or self.config.DEFAULT_CURRENCY
        
        url = f"{self.config.BASE_URL}/coins/{crypto_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': 'daily'
        }
        
        logger.info(f"Fetching {days} days of {crypto_id} data...")
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.config.REQUEST_TIMEOUT
                )
                
                # Check for rate limiting
                if response.status_code == 429:
                    wait_time = int(response.headers.get('Retry-After', self.config.RETRY_DELAY))
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                # Validate response structure
                if 'prices' not in data:
                    raise ValueError("Invalid response format: missing 'prices' field")
                
                if not data['prices']:
                    raise ValueError("No price data returned")
                
                # Process the data
                df = self._process_price_data(data['prices'], crypto_id, vs_currency)
                
                logger.info(f"Successfully fetched {len(df)} data points for {crypto_id}")
                return df
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{self.config.MAX_RETRIES}): {str(e)}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY)
                else:
                    logger.error("Max retries exceeded. Data collection failed.")
                    return None
                    
            except (ValueError, KeyError) as e:
                logger.error(f"Data processing error: {str(e)}")
                return None
                
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return None
        
        return None
    
    def _process_price_data(self, prices_data: list, crypto_id: str, vs_currency: str) -> pd.DataFrame:
        """Process raw price data into a structured DataFrame."""
        try:
            df = pd.DataFrame(prices_data, columns=['timestamp', 'price'])
            
            # Convert timestamp to datetime
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.drop('timestamp', axis=1, inplace=True)
            
            # Add metadata
            df['crypto_id'] = crypto_id
            df['currency'] = vs_currency
            
            # Reorder columns
            df = df[['date', 'price', 'crypto_id', 'currency']]
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Validate data quality
            self._validate_data_quality(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing price data: {str(e)}")
            raise
    
    def _validate_data_quality(self, df: pd.DataFrame):
        """Validate the quality of collected data."""
        # Check for missing values
        if df['price'].isna().any():
            logger.warning("Missing price values detected")
        
        # Check for negative prices
        if (df['price'] < 0).any():
            logger.warning("Negative price values detected")
        
        # Check for duplicate dates
        if df['date'].duplicated().any():
            logger.warning("Duplicate date entries detected")
        
        # Check for reasonable price range
        price_std = df['price'].std()
        price_mean = df['price'].mean()
        
        # Flag extreme outliers (more than 5 standard deviations)
        outliers = df[abs(df['price'] - price_mean) > 5 * price_std]
        if not outliers.empty:
            logger.warning(f"Detected {len(outliers)} potential price outliers")
        
        logger.info(f"Data quality check completed. Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> bool:
        """Save DataFrame to CSV file."""
        filename = filename or self.config.OUTPUT_FILE
        
        try:
            # Create backup if file exists
            if os.path.exists(filename):
                backup_name = f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(filename, backup_name)
                logger.info(f"Created backup: {backup_name}")
            
            df.to_csv(filename, index=False)
            logger.info(f"Data saved to {filename}")
            
            # Verify the saved file
            test_df = pd.read_csv(filename)
            if len(test_df) != len(df):
                raise ValueError("Saved file verification failed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False
    
    def get_available_cryptocurrencies(self) -> Optional[Dict[str, Any]]:
        """Get list of available cryptocurrencies from CoinGecko."""
        url = f"{self.config.BASE_URL}/coins/list"
        
        try:
            response = self.session.get(url, timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            coins = response.json()
            logger.info(f"Retrieved {len(coins)} available cryptocurrencies")
            
            return coins
            
        except Exception as e:
            logger.error(f"Error fetching cryptocurrency list: {str(e)}")
            return None

def main():
    """Main function for data collection."""
    try:
        collector = CryptoDataCollector()
        
        # You can specify different cryptocurrencies here
        cryptocurrencies = ['bitcoin']  # Add more: ['bitcoin', 'ethereum', 'cardano']
        
        all_data = []
        
        for crypto in cryptocurrencies:
            logger.info(f"Collecting data for {crypto}...")
            
            df = collector.get_historical_data(
                crypto_id=crypto,
                days=365,  # 1 year of data
                vs_currency='usd'
            )
            
            if df is not None:
                all_data.append(df)
            else:
                logger.error(f"Failed to collect data for {crypto}")
        
        if not all_data:
            logger.error("No data collected for any cryptocurrency")
            return False
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save the data
        success = collector.save_data(combined_df)
        
        if success:
            logger.info("Data collection completed successfully!")
            logger.info(f"Total records: {len(combined_df)}")
            logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        else:
            logger.error("Failed to save collected data")
        
        return success
        
    except Exception as e:
        logger.error(f"Data collection failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
