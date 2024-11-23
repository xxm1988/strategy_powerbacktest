from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class Portfolio:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict] = []
        self.history = pd.DataFrame(columns=['timestamp', 'capital', 'holdings', 'total'])
        
    def update(self, timestamp: pd.Timestamp, prices: Dict[str, float]) -> None:
        """
        Update portfolio state with new prices
        
        Args:
            timestamp: Current timestamp
            prices: Dictionary of symbol -> price mappings
        """
        holdings_value = sum(
            self.positions.get(symbol, 0) * price 
            for symbol, price in prices.items()
        )
        
        total_value = self.current_capital + holdings_value
        
        self.history = pd.concat([
            self.history,
            pd.DataFrame([{
                'timestamp': timestamp,
                'capital': self.current_capital,
                'holdings': holdings_value,
                'total': total_value
            }])
        ], ignore_index=True)
        
    def execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: pd.Timestamp,
        commission: float = 0.0
    ) -> bool:
        """
        Execute a trade and update portfolio state
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares (positive for buy, negative for sell)
            price: Execution price
            timestamp: Trade timestamp
            commission: Commission rate
            
        Returns:
            Boolean indicating if trade was successful
        """
        trade_value = quantity * price
        commission 