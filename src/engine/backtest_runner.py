from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
import pandas as pd

from ..data.data_fetcher import FutuDataFetcher
from ..strategy.strategy_factory import StrategyFactory
from .backtest_engine import BacktestEngine
from ..utils.logger import setup_logger

@dataclass
class BacktestConfig:
    """Configuration for backtest run"""
    strategy_name: str
    strategy_params: Dict
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission: float

class BacktestRunner:
    """
    Handles the execution of backtests with different strategies and parameters
    
    Attributes:
        data_fetcher (FutuDataFetcher): Data fetcher instance
        logger: Logger instance
    """
    
    def __init__(self, host: str = 'localhost', port: int = 11111):
        """
        Initialize BacktestRunner
        
        Args:
            host (str): Futu OpenD host
            port (int): Futu OpenD port
        """
        self.data_fetcher = FutuDataFetcher(host=host, port=port)
        self.logger = setup_logger(__name__)
    
    def run(self, config: BacktestConfig) -> str:
        """
        Run backtest with given configuration
        
        Args:
            config (BacktestConfig): Backtest configuration
            
        Returns:
            str: Path to generated report
            
        Raises:
            ValueError: If strategy creation fails
        """
        # Fetch historical data
        data = self.data_fetcher.fetch_historical_data(
            symbol=config.symbol,
            start=config.start_date,
            end=config.end_date
        )
        
        # Create strategy
        strategy = StrategyFactory.create_strategy(
            config.strategy_name,
            config.strategy_params
        )
        
        # Initialize and run backtest engine
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=config.initial_capital,
            commission=config.commission
        )
        
        results = engine.run(data)
        return results.generate_report() 