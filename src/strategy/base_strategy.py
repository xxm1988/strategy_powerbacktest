from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime

class BaseStrategy(ABC):
    def __init__(self, parameters: Dict[str, Any] = None):
        self.parameters = parameters or {}
        self.position = 0
        self.signals = []
        self.lot_size = parameters.get('lot_size', 1)  # Default lot size
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def set_lot_size(self, lot_size: int):
        """Set the lot size for the stock"""
        self.lot_size = lot_size
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the strategy logic
        
        Args:
            data: Historical market data
            
        Returns:
            Series of trading signals (1: Buy, -1: Sell, 0: Hold)
        """
        pass
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators used by the strategy
        
        Args:
            data: Historical market data
            
        Returns:
            DataFrame with additional technical indicators
        """
        return data
        
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters
        
        Returns:
            Boolean indicating if parameters are valid
        """
        return True
        
    def calculate_position_size(self, capital: float, price: float) -> int:
        """Calculate position size in lots based on available capital"""
        max_shares = int(capital / price)
        return (max_shares // self.lot_size) * self.lot_size
        
    def get_required_warmup_period(self) -> int:
        """
        Get the required warmup period for the strategy.
        Each strategy should override this method based on its indicator requirements.
        
        Returns:
            int: Number of periods needed for warmup
        """
        return 0  # Base class returns 0, child classes should override
        
    def prepare_data(self, data: pd.DataFrame, start_date: datetime = None, end_date: datetime = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data by including warmup period for indicator calculation
        
        Args:
            data: Original market data
            start_date: Start date for trading period
            end_date: End date for trading period
            
        Returns:
            Tuple containing:
                - DataFrame with indicators (including warmup)
                - DataFrame with indicators (warmup removed)
        """
        # Calculate indicators on full dataset
        calculated = self.calculate_indicators(data)
        
        # If no warmup needed and no date filtering required, return the same data
        if self.get_required_warmup_period() == 0:
            return calculated, calculated
            
        # Convert time_key to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(calculated.index):
            calculated = calculated.set_index('time_key', drop=False)
            calculated.index = pd.to_datetime(calculated.index)

        # Get the trading period data based on user-specified dates
        mask = pd.Series(True, index=calculated.index)
        if start_date:
            mask &= (calculated.index >= pd.to_datetime(start_date))
        if end_date:
            mask &= (calculated.index <= pd.to_datetime(end_date))
        
        trading_period = calculated[mask]
        trading_period.reset_index(drop=True, inplace=True)
        
        return calculated, trading_period