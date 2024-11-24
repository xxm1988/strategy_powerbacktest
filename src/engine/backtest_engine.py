from typing import Dict, Any, List
import pandas as pd
import numpy as np
from src.strategy.base_strategy import BaseStrategy
from src.data.data_fetcher import FutuDataFetcher
from src.utils.logger import setup_logger
from .backtest_report import BacktestReport
from .portfolio import Portfolio

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
        self.logger = setup_logger(__name__)
        
    def run(self, data: pd.DataFrame) -> BacktestReport:
        """Run backtest on historical data"""
        # Log backtest initialization
        start_date = pd.to_datetime(data['time_key'].iloc[0]).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(data['time_key'].iloc[-1]).strftime('%Y-%m-%d')
        
        self.logger.info(f"Starting backtest for {data['name'].iloc[0]}) from {start_date} to {end_date}")
        self.logger.info(f"Strategy: {self.strategy.__class__.__name__}")
        self.logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        self.logger.info(f"Commission rate: {self.commission*100:.2f}%")
        
        signals = self.strategy.generate_signals(data)
        self.logger.info("Generated trading signals")
        
        self.portfolio = self._calculate_portfolio(data, signals)
        self.logger.info("Calculated portfolio values")
        
        trades = self._generate_trades_list(data, signals)
        self.logger.info(f"Generated {len(trades)} trades")
        
        # Log final results
        final_value = self.portfolio['total'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        self.logger.info(f"Backtest completed. Final portfolio value: ${final_value:,.2f}")
        self.logger.info(f"Total return: {total_return:.2f}%")
        
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
        """Calculate portfolio value over time"""
        portfolio = pd.DataFrame(index=data.index)
        
        # Ensure the index is datetime
        if not isinstance(portfolio.index, pd.DatetimeIndex):
            portfolio.index = pd.to_datetime(portfolio.index)
        
        # Track positions and cash
        portfolio['position'] = 0  # Number of units held
        portfolio['close'] = data['close']
        
        # Initialize cash with initial capital
        portfolio['cash'] = self.initial_capital
        
        # Process signals and update portfolio
        current_position = 0
        for i in range(1, len(signals)):
            if signals[i] == 1 and current_position == 0:  # Buy signal
                # Buy 1 unit
                cost = data['close'].iloc[i] * (1 + self.commission)
                if portfolio['cash'].iloc[i-1] >= cost:
                    current_position = 1
                    portfolio.loc[portfolio.index[i:], 'position'] = 1
                    portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - cost
            elif signals[i] == -1 and current_position > 0:  # Sell signal
                # Sell all units
                proceeds = current_position * data['close'].iloc[i] * (1 - self.commission)
                current_position = 0
                portfolio.loc[portfolio.index[i:], 'position'] = 0
                portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] + proceeds
            else:
                portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
        
        # Forward fill cash values
        portfolio['cash'] = portfolio['cash'].fillna(method='ffill')
        
        # Calculate holdings value and total portfolio value
        portfolio['holdings'] = portfolio['position'] * portfolio['close']
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
        """Generate list of trades from signals"""
        trades = []
        current_position = 0
        last_buy_price = 0
        realized_pnl = 0
        
        for i in range(1, len(signals)):
            if signals[i] == 1 and current_position == 0:  # Buy signal
                price = data['close'].iloc[i]
                timestamp = data['time_key'].iloc[i]
                cost = price * (1 + self.commission)
                
                trade = {
                    'timestamp': timestamp,
                    'type': 'buy',
                    'price': price,
                    'quantity': 1,
                    'cost': cost,
                    'commission': cost - price
                }
                trades.append(trade)
                current_position = 1
                last_buy_price = price
                
                self.logger.info(
                    f"Trade executed at {timestamp}: BUY "
                    f"1 unit at ${price:.2f} "
                    f"(Cost: ${cost:,.2f}, Commission: ${cost - price:.2f})"
                )
                
            elif signals[i] == -1 and current_position > 0:  # Sell signal
                price = data['close'].iloc[i]
                timestamp = data['time_key'].iloc[i]
                proceeds = price * (1 - self.commission)
                trade_pnl = proceeds - last_buy_price * (1 + self.commission)
                realized_pnl += trade_pnl
                
                trade = {
                    'timestamp': timestamp,
                    'type': 'sell',
                    'price': price,
                    'quantity': current_position,
                    'proceeds': proceeds,
                    'commission': price - proceeds,
                    'pnl': trade_pnl
                }
                trades.append(trade)
                current_position = 0
                
                self.logger.info(
                    f"Trade executed at {timestamp}: SELL "
                    f"1 unit at ${price:.2f} "
                    f"(Proceeds: ${proceeds:,.2f}, Commission: ${price - proceeds:.2f})"
                )
                self.logger.info(f"Trade PnL: ${trade_pnl:,.2f} (Running PnL: ${realized_pnl:,.2f})")
        
        # Calculate floating PnL at the end of backtest period
        if current_position > 0:
            final_price = data['close'].iloc[-1]
            floating_pnl = (final_price - last_buy_price) * current_position
            self.logger.info(
                f"\nEnd of Backtest Summary:"
                f"\nRealized PnL: ${realized_pnl:,.2f}"
                f"\nFloating PnL: ${floating_pnl:,.2f} (from {current_position} open position)"
                f"\nTotal PnL: ${(realized_pnl + floating_pnl):,.2f}"
            )
        else:
            self.logger.info(
                f"\nEnd of Backtest Summary:"
                f"\nRealized PnL: ${realized_pnl:,.2f}"
                f"\nNo open positions"
            )
        
        return trades