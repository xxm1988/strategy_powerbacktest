from futu import *
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import logging
from src.utils.logger import setup_logger
from src.utils.timeframe import get_kl_type
from src.utils.data_resampler import resample_ohlcv

logger = setup_logger(__name__)

class FutuDataFetcher:
    def __init__(self, host: str = 'localhost', port: int = 11111):
        self.quote_ctx = OpenQuoteContext(host=host, port=port)
        logger.info("Initialized Futu OpenQuoteContext")
        
    def __del__(self):
        self.quote_ctx.close()
        
    def fetch_historical_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "DAY"
    ) -> pd.DataFrame:
        """
        Fetch historical data from Futu OpenAPI and resample if needed
        
        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            timeframe: K-line timeframe (e.g., "1M", "4H", "DAY")
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Map custom timeframes to available Futu timeframes
            futu_timeframe_map = {
                "4H": "60M",  # Fetch 1-hour data for 4H resampling
                "2H": "60M",  # Fetch 1-hour data for 2H resampling
                "3H": "60M",  # Fetch 1-hour data for 3H resampling
            }
            
            # Use mapped timeframe for Futu API
            fetch_timeframe = futu_timeframe_map.get(timeframe.upper(), timeframe)
            kl_type = get_kl_type(fetch_timeframe)
            
            ret_code, data, page_req_key = self.quote_ctx.request_history_kline(
                symbol,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                ktype=kl_type
            )
            
            if ret_code != RET_OK:
                logger.error(f"Failed to fetch data: {data}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Resample if needed
            if timeframe.upper() in futu_timeframe_map:
                df = resample_ohlcv(df, timeframe)
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise 

    def get_trading_calendar(self, start_date: datetime, end_date: datetime) -> pd.DatetimeIndex:
        """
        Get trading calendar between start and end dates using Futu OpenAPI
        
        Args:
            start_date: Start date for calendar
            end_date: End date for calendar
            
        Returns:
            DatetimeIndex of trading days
        """
        try:
            # Convert dates to string format required by Futu API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Request trading days from Futu API
            ret_code, data = self.quote_ctx.request_trading_days(
                market=TrdMarket.HK,  # Hong Kong market
                start_date=start_str,
                end_date=end_str
            )
            
            if ret_code != RET_OK:
                self.logger.error(f"Failed to fetch trading calendar: {data}")
                return pd.DatetimeIndex([])
            
            # Convert to DatetimeIndex and sort
            calendar = pd.DatetimeIndex(data['trading_days']).sort_values()
            
            return calendar
            
        except Exception as e:
            self.logger.error(f"Error getting trading calendar: {str(e)}")
            return pd.DatetimeIndex([])

