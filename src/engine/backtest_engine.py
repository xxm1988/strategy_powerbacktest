from typing import Dict, Any
import pandas as pd
import numpy as np
from src.strategy.base_strategy import BaseStrategy
from src.data.data_fetcher import FutuDataFetcher
from src.utils.logger import setup_logger
from .portfolio import Portfolio

logger = setup_logger(__name__)

logger = setup_logger(__name__)

class BacktestEngine:
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 100000.0,
        commission: float = 0.001
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.portfolio = pd.DataFrame()
        self.positions = pd.DataFrame()
        self.trades = []
        
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest on historical data
        
        Args:
            data: Historical market data
            
        Returns:
            Dictionary containing backtest results
        """
        signals = self.strategy.generate_signals(data)
        self.portfolio = self._calculate_portfolio(data, signals)
        return self._calculate_metrics()
        
    def _calculate_portfolio(
        self,
        data: pd.DataFrame,
        signals: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate portfolio value over time
        """
        portfolio = pd.DataFrame(index=data.index)
        portfolio['holdings'] = signals.shift(1) * data['close']
        portfolio['cash'] = self.initial_capital - (
            signals.diff() * data['close'] * (1 + self.commission)
        ).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio
        
    def _calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics
        """
        returns = self.portfolio['returns']
        total_return = (
            self.portfolio['total'].iloc[-1] - self.initial_capital
        ) / self.initial_capital
        
        metrics = {
            'total_return': total_return,
            'annual_return': returns.mean() * 252,
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std(),
            'max_drawdown': (
                self.portfolio['total'] / self.portfolio['total'].cummax() - 1
            ).min(),
            'win_rate': len(returns[returns > 0]) / len(returns[returns != 0])
        }
        
        return metrics 