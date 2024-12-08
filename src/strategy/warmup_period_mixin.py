from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
class WarmupPeriodMixin(ABC):
    def __init__(self):
        self._warmup_period = self.calculate_warmup_period()
    
    @abstractmethod
    def calculate_warmup_period(self) -> int:
        """Calculate required warmup period for the strategy"""
        pass
    
    def is_warmed_up(self, data_length: int) -> bool:
        """Check if enough data is available for warmup"""
        return data_length >= self._warmup_period
    
    def get_valid_signals(self, signals: pd.Series) -> pd.Series:
        """Remove signals during warmup period"""
        signals.iloc[:self._warmup_period] = 0
        return signals