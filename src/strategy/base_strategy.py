from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

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