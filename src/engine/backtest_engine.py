from typing import Dict, Any, List
import pandas as pd
import numpy as np
from src.strategy.base_strategy import BaseStrategy
from src.data.data_fetcher import FutuDataFetcher
from src.utils.logger import setup_logger
from .backtest_report import BacktestReport
from .portfolio import Portfolio

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
        
    def run(self, data: pd.DataFrame) -> BacktestReport:
        """
        Run backtest on historical data
        
        Args:
            data: Historical market data
            
        Returns:
            BacktestReport object containing detailed results
        """
        signals = self.strategy.generate_signals(data)
        self.portfolio = self._calculate_portfolio(data, signals)
        
        # Generate trades list
        trades = self._generate_trades_list(data, signals)
        
        # Create and return backtest report
        report = BacktestReport.from_backtest_results(
            portfolio=self.portfolio,
            trades=trades,
            initial_capital=self.initial_capital
        )
        
        return report
        
    def _calculate_portfolio(
        self,
        data: pd.DataFrame,
        signals: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate portfolio value over time
        
        Args:
            data: Historical market data
            signals: Trading signals series
            
        Returns:
            DataFrame with portfolio values and returns
        """
        # Create portfolio DataFrame with datetime index
        portfolio = pd.DataFrame(index=data.index)
        
        # Ensure the index is datetime
        if not isinstance(portfolio.index, pd.DatetimeIndex):
            portfolio.index = pd.to_datetime(portfolio.index)
        
        # Calculate portfolio values
        portfolio['holdings'] = signals.shift(1) * data['close']
        portfolio['cash'] = self.initial_capital - (
            signals.diff() * data['close'] * (1 + self.commission)
        ).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change(fill_method=None)
        
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
        
    def _generate_trades_list(self, data: pd.DataFrame, signals: pd.Series) -> List[Dict]:
        """
        Generate list of trades from signals
        
        Args:
            data: Historical market data
            signals: Trading signals series
            
        Returns:
            List of trade dictionaries
        """
        trades = []
        position = 0
        
        # Iterate through signals to identify trades
        for i in range(1, len(signals)):
            signal_change = signals[i] - signals[i-1]
            
            if signal_change != 0:  # Trade occurred
                price = data['close'].iloc[i]
                timestamp = data.index[i]
                
                # Calculate trade details
                quantity = signal_change
                cost = abs(quantity * price)
                commission_cost = cost * self.commission
                
                trades.append({
                    'timestamp': timestamp,
                    'type': 'buy' if quantity > 0 else 'sell',
                    'price': price,
                    'quantity': abs(quantity),
                    'cost': cost,
                    'commission': commission_cost,
                    'pnl': 0  # Will be calculated later
                })
                
                position += quantity
        
        # Calculate PnL for each trade
        running_position_cost = 0
        for trade in trades:
            if trade['type'] == 'buy':
                running_position_cost += trade['cost']
            else:  # sell
                trade['pnl'] = (trade['cost'] - running_position_cost) - trade['commission']
                running_position_cost = 0
                
        return trades