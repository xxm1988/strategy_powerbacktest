from futu import *
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import logging
from src.utils.logger import setup_logger

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
        interval: KLType = KLType.K_DAY
    ) -> pd.DataFrame:
        """
        Fetch historical data from Futu OpenAPI
        
        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            interval: Kline interval
            
        Returns:
            DataFrame with historical data
        """
        try:
            ret_code, data, page_req_key = self.quote_ctx.request_history_kline(
                symbol,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                ktype=interval
            )
            
            if ret_code != RET_OK:
                logger.error(f"Failed to fetch data: {data}")
                return pd.DataFrame()
                
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise 