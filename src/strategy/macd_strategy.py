from typing import Dict
import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy
class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy
    Buy Signal: When MACD line crosses above Signal line (Golden Cross)
    Sell Signal: When MACD line crosses below Signal line (Death Cross)
    """
    
    def __init__(self, params: Dict):
        """
        Initialize MACD Strategy with parameters
        
        Args:
            params: Dictionary containing:
                - fast_period: Period for fast EMA (default: 12)
                - slow_period: Period for slow EMA (default: 26)
                - signal_period: Period for signal line (default: 9)
        """
        super().__init__(params)
        self.fast_period = params.get('fast_period', 12)
        self.slow_period = params.get('slow_period', 26)
        self.signal_period = params.get('signal_period', 9)
        
    def calculate_warmup_period(self) -> int:
        """Calculate required warmup period for MACD"""
        # Need enough data for slow EMA + signal line
        return self.slow_period + self.signal_period
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:   
        # Calculate MACD components
        fast_ema = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Generate signals based on crossovers
        # Golden Cross (Buy): MACD crosses above Signal
        # Death Cross (Sell): MACD crosses below Signal
        for i in range(1, len(data)):
            if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
                signals[i] = 1  # Buy signal
            elif macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
                signals[i] = -1  # Sell signal
        
        # Remove signals during warmup period
        return signals