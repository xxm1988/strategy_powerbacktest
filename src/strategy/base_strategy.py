from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    def __init__(self, parameters: Dict[str, Any] = None):
        self.parameters = parameters or {}
        self.position = 0
        self.signals = []
        
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

class MovingAverageCrossStrategy(BaseStrategy):
    def __init__(self, parameters: Dict[str, Any] = None):
        super().__init__(parameters)
        self.short_window = parameters.get('short_window', 20)
        self.long_window = parameters.get('long_window', 50)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data['SMA_short'] = data['close'].rolling(window=self.short_window).mean()
        data['SMA_long'] = data['close'].rolling(window=self.long_window).mean()
        return data
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = self.calculate_indicators(data)
        signals = pd.Series(0, index=data.index)
        signals[data['SMA_short'] > data['SMA_long']] = 1
        signals[data['SMA_short'] < data['SMA_long']] = -1
        return signals 