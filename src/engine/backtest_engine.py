from typing import Dict, Any, List
import pandas as pd
import numpy as np
from src.strategy.base_strategy import BaseStrategy
from src.data.data_fetcher import FutuDataFetcher
from src.utils.logger import setup_logger
from .backtest_report import BacktestReport
from .portfolio import Portfolio
from futu import OpenQuoteContext

class BacktestEngine:
    """
    BacktestEngine handles the execution of trading strategies in a simulated environment.
    
    This class manages portfolio state, executes trades based on strategy signals,
    and calculates performance metrics while respecting lot size constraints for HK stocks.
    
    Attributes:
        strategy (BaseStrategy): Trading strategy to backtest
        initial_capital (float): Starting capital for the backtest
        commission (float): Commission rate for trades
        lot_size (int): Trading lot size for the instrument
        logger: Configured logging instance
    """
    
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
        self.lot_size = 1  # Will be updated in run()
        
    def run(self, data: pd.DataFrame, symbol: str) -> BacktestReport:
        """
        Execute backtest for the given data and symbol.
        
        Args:
            data (pd.DataFrame): Historical price data
            symbol (str): Trading symbol (e.g., 'HK.00700')
            
        Returns:
            BacktestReport: Comprehensive backtest results and metrics
        """
        self.lot_size = self.fetch_lot_size(symbol)
        self._log_backtest_start(data, symbol)
        
        signals = self.strategy.generate_signals(data)
        self.portfolio = self._calculate_portfolio(data, signals)
        trades = self._generate_trades_list(data, signals)
        self.trades = trades
        
        metrics = self._calculate_metrics()
        self._log_backtest_summary(metrics)
        
        return BacktestReport.from_backtest_results(
            portfolio=self.portfolio,
            trades=trades,
            initial_capital=self.initial_capital,
            metrics=metrics
        )
        
    def _log_backtest_start(self, data: pd.DataFrame, symbol: str) -> None:
        """Log backtest initialization details"""
        start_date = pd.to_datetime(data['time_key'].iloc[0]).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(data['time_key'].iloc[-1]).strftime('%Y-%m-%d')
        
        self.logger.info("=" * 50)
        self.logger.info("Backtest Configuration:")
        self.logger.info(f"Symbol: {symbol} (Lot Size: {self.lot_size})")
        self.logger.info(f"Period: {start_date} to {end_date}")
        self.logger.info(f"Strategy: {self.strategy.__class__.__name__}")
        self.logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        self.logger.info(f"Commission Rate: {self.commission*100:.2f}%")
        self.logger.info("=" * 50)
        
    def _log_trade_execution(self, trade: Dict[str, Any]) -> None:
        """Log trade execution details"""
        trade_type = trade['type'].upper()
        quantity = trade['quantity']
        lots = quantity // self.lot_size
        price = trade['price']
        
        if trade_type == 'BUY':
            cost = trade['cost']
            commission = trade['commission']
            self.logger.info(
                f"TRADE: {trade['timestamp']} | {trade_type} | "
                f"{quantity:,d} units ({lots:,d} lots) @ ${price:.3f} | "
                f"Cost: ${cost:,.2f} | Commission: ${commission:.2f}"
            )
        else:  # SELL
            proceeds = trade['proceeds']
            pnl = trade['pnl']
            self.logger.info(
                f"TRADE: {trade['timestamp']} | {trade_type} | "
                f"{quantity:,d} units ({lots:,d} lots) @ ${price:.3f} | "
                f"Proceeds: ${proceeds:,.2f} | PnL: ${pnl:,.2f}"
            )
        
    def _log_backtest_summary(self, metrics: Dict[str, Any]) -> None:
        """Log final backtest results"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Backtest Summary:")
        self.logger.info(f"Total Return: {metrics['total_return']*100:.2f}%")
        self.logger.info(f"Annual Return: {metrics['annual_return']*100:.2f}%")
        self.logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        self.logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        self.logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        self.logger.info(f"Total Trades: {metrics['total_trades']}")
        self.logger.info(f"Total PnL: ${metrics['total_pnl']:,.2f}")
        self.logger.info("=" * 50)
        
    def _calculate_portfolio(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """Calculate portfolio value over time with lot size handling"""
        # Initialize portfolio DataFrame properly
        portfolio = pd.DataFrame({
            'time_key': data['time_key'],
            'position': 0,
            'close': pd.to_numeric(data['close'].values),
            'cash': self.initial_capital
        }).set_index('time_key')
        portfolio.index = pd.to_datetime(portfolio.index)
        
        current_position = 0
        for i in range(1, len(signals)):
            price = portfolio['close'].iloc[i]
            
            if signals[i] == 1 and current_position == 0:  # Buy signal
                max_shares = int(portfolio['cash'].iloc[i-1] / (price * (1 + self.commission)))
                lots_to_buy = (max_shares // self.lot_size) * self.lot_size
                
                if lots_to_buy >= self.lot_size:
                    cost = lots_to_buy * price * (1 + self.commission)
                    current_position = lots_to_buy
                    portfolio.loc[portfolio.index[i:], 'position'] = lots_to_buy
                    portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - cost
                    
            elif signals[i] == -1 and current_position > 0:  # Sell signal
                proceeds = current_position * price * (1 - self.commission)
                current_position = 0
                portfolio.loc[portfolio.index[i:], 'position'] = 0
                portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] + proceeds
            else:
                portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
        
        # Use proper forward fill
        portfolio['cash'] = portfolio['cash'].ffill()
        
        # Calculate portfolio values
        portfolio['holdings'] = portfolio['position'] * portfolio['close']
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        
        # Calculate returns properly
        portfolio.loc[:, 'returns'] = portfolio['total'].pct_change()
        portfolio.loc[portfolio.index[0], 'returns'] = 0
        
        return portfolio
        
    def _calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics including PnL and risk metrics
        
        Returns:
            Dict containing performance metrics:
            - total_return: Overall return percentage
            - annual_return: Annualized return
            - sharpe_ratio: Risk-adjusted return metric
            - max_drawdown: Maximum peak to trough decline
            - win_rate: Percentage of winning trades
            - realized_pnl: Total realized profit/loss
            - floating_pnl: Unrealized profit/loss from open positions
            - total_pnl: Combined realized and floating PnL
        """
        # Calculate basic return metrics
        returns = self.portfolio['returns'].dropna()
        total_value = self.portfolio['total'].iloc[-1]
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        # Calculate PnL metrics from trades
        realized_pnl = sum(trade['pnl'] for trade in self.trades if trade['type'] == 'sell')
        
        # Calculate floating PnL from current positions
        current_position = self.portfolio['position'].iloc[-1]
        if current_position > 0:
            last_buy_trade = next(trade for trade in reversed(self.trades) 
                                if trade['type'] == 'buy')
            last_buy_price = last_buy_trade['price']
            final_price = self.portfolio['close'].iloc[-1]
            floating_pnl = (final_price - last_buy_price) * current_position
        else:
            floating_pnl = 0
        
        total_pnl = realized_pnl + floating_pnl
        
        # Calculate risk metrics
        if len(returns) > 0:
            annual_return = returns.mean() * 252
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            max_drawdown = (self.portfolio['total'] / self.portfolio['total'].cummax() - 1).min()
            
            # Calculate win rate from completed trades only
            completed_trades = [t for t in self.trades if t['type'] == 'sell']
            winning_trades = len([t for t in completed_trades if t['pnl'] > 0])
            win_rate = winning_trades / len(completed_trades) if completed_trades else 0
        else:
            annual_return = sharpe_ratio = max_drawdown = win_rate = 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'realized_pnl': realized_pnl,
            'floating_pnl': floating_pnl,
            'total_pnl': total_pnl,
            'total_trades': len(self.trades),
            'open_positions': current_position
        }
        
        return metrics
        
    def _generate_trades_list(self, data: pd.DataFrame, signals: pd.Series) -> List[Dict]:
        """Generate list of trades from signals with lot size handling"""
        trades = []
        current_position = 0
        last_buy_price = 0
        realized_pnl = 0
        
        for i in range(1, len(signals)):
            if signals[i] == 1 and current_position == 0:  # Buy signal
                price = data['close'].iloc[i]
                timestamp = data['time_key'].iloc[i]
                
                # Calculate lots to buy based on available cash
                max_shares = int(self.portfolio['cash'].iloc[i-1] / (price * (1 + self.commission)))
                lots_to_buy = (max_shares // self.lot_size) * self.lot_size
                
                if lots_to_buy >= self.lot_size:
                    cost = lots_to_buy * price * (1 + self.commission)
                    trade = {
                        'timestamp': timestamp,
                        'type': 'buy',
                        'price': price,
                        'quantity': lots_to_buy,
                        'cost': cost,
                        'commission': cost - (lots_to_buy * price)
                    }
                    trades.append(trade)
                    current_position = lots_to_buy
                    last_buy_price = price
                    
                    self.logger.info(
                        f"Trade executed at {timestamp}: BUY "
                        f"{lots_to_buy} units ({lots_to_buy//self.lot_size} lots) at ${price:.2f} "
                        f"(Cost: ${cost:,.2f}, Commission: ${cost - (lots_to_buy * price):.2f})"
                    )
                
            elif signals[i] == -1 and current_position > 0:  # Sell signal
                price = data['close'].iloc[i]
                timestamp = data['time_key'].iloc[i]
                proceeds = current_position * price * (1 - self.commission)
                trade_pnl = proceeds - (current_position * last_buy_price * (1 + self.commission))
                realized_pnl += trade_pnl
                
                trade = {
                    'timestamp': timestamp,
                    'type': 'sell',
                    'price': price,
                    'quantity': current_position,
                    'proceeds': proceeds,
                    'commission': (current_position * price) - proceeds,
                    'pnl': trade_pnl
                }
                trades.append(trade)
                
                self.logger.info(
                    f"Trade executed at {timestamp}: SELL "
                    f"{current_position} units ({current_position//self.lot_size} lots) at ${price:.2f} "
                    f"(Proceeds: ${proceeds:,.2f}, Commission: ${(current_position * price) - proceeds:.2f})"
                )
                self.logger.info(f"Trade PnL: ${trade_pnl:,.2f} (Running PnL: ${realized_pnl:,.2f})")
                
                current_position = 0
        
        # Calculate floating PnL at the end of backtest period
        if current_position > 0:
            final_price = data['close'].iloc[-1]
            floating_pnl = (final_price - last_buy_price) * current_position
            self.logger.info(
                f"\nEnd of Backtest Summary:"
                f"\nRealized PnL: ${realized_pnl:,.2f}"
                f"\nFloating PnL: ${floating_pnl:,.2f} (from {current_position} units, {current_position//self.lot_size} lots)"
                f"\nTotal PnL: ${(realized_pnl + floating_pnl):,.2f}"
            )
        else:
            self.logger.info(
                f"\nEnd of Backtest Summary:"
                f"\nRealized PnL: ${realized_pnl:,.2f}"
                f"\nNo open positions"
            )
        
        return trades
        
    def fetch_lot_size(self, symbol: str) -> int:
        """Fetch lot size from Futu OpenD"""
        quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
        ret, data = quote_ctx.get_market_snapshot([symbol])
        quote_ctx.close()
        
        if ret == 0:
            return data['lot_size'][0]
        return 1  # Default to 1 if unable to fetch