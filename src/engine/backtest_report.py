from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from jinja2 import Template
from ..engine.fundamental_data import FundamentalData
from .metrics.risk_metrics import RiskMetrics


@dataclass
class BacktestReport:
    """
    Comprehensive backtest report with analytics and visualization capabilities.
    
    This class handles the generation of performance metrics, charts, and HTML reports
    for trading strategy backtests.
    
    Attributes:
        Basic Info:
            strategy_name (str): Name of the trading strategy
            symbol (str): Trading instrument symbol
            start_date (datetime): Backtest start date
            end_date (datetime): Backtest end date
            lot_size (int): Trading lot size
            commission_rate (float): Trading commission rate
            
        Performance Metrics:
            initial_capital (float): Starting capital
            final_portfolio_value (float): Ending portfolio value
            total_return (float): Total return percentage
            annual_return (float): Annualized return percentage
            
        Risk Metrics:
            sharpe_ratio (float): Risk-adjusted return metric
            sortino_ratio (float): Downside risk-adjusted return metric
            max_drawdown (float): Maximum peak to trough decline
            max_drawdown_duration (int): Longest drawdown period in days
            volatility (float): Standard deviation of returns
            value_at_risk (float): 95% Value at Risk
            beta (float): Market correlation coefficient
            
        Trading Statistics:
            total_trades (int): Total number of trades
            winning_trades (int): Number of profitable trades
            losing_trades (int): Number of unprofitable trades
            win_rate (float): Percentage of winning trades
            profit_factor (float): Ratio of gross profits to gross losses
            
        Time Series Data:
            portfolio (pd.DataFrame): Portfolio value and returns history
            trades (List[Dict[str, Any]]): Detailed trade history
            monthly_returns (pd.DataFrame): Monthly return statistics
        Fundamental Data:
            fundamental_data (FundamentalData): Company fundamental data
    """
    # Basic Info
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    lot_size: int
    commission_rate: float
    
    # Capital and Returns
    initial_capital: float
    final_portfolio_value: float
    total_return: float
    annual_return: float
    
    # Risk Metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    value_at_risk: float
    beta: float
    
    # Trading Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Position Info
    avg_position_size: float
    max_position_size: float
    avg_position_duration: float
    
    # Time Series Data
    portfolio: pd.DataFrame
    trades: List[Dict[str, Any]]
    monthly_returns: pd.DataFrame
    fundamental_data: FundamentalData

    @classmethod
    def from_backtest_results(
        cls,
        portfolio: pd.DataFrame,
        trades: List[Dict[str, Any]],
        initial_capital: float,
        metrics: Dict[str, Any]
    ) -> 'BacktestReport':
        """Create a backtest report from raw backtest results"""
        # Count all trades, not just those with PNL
        total_trades = len(trades)
        
        # Filter trades with PNL for calculations that require it
        trades_with_pnl = [t for t in trades if t.get('pnl') is not None]
        
        # Handle empty trades list
        if not trades:
            return cls(
                strategy_name=metrics['strategy_name'],
                symbol=metrics['symbol'],
                start_date=portfolio.index[0],
                end_date=portfolio.index[-1],
                lot_size=metrics['lot_size'],
                commission_rate=metrics['commission_rate'],
                initial_capital=initial_capital,
                final_portfolio_value=portfolio['total'].iloc[-1],
                total_return=metrics['total_return'],
                annual_return=metrics['annual_return'],
                sharpe_ratio=metrics['sharpe_ratio'],
                sortino_ratio=RiskMetrics.calculate_sortino_ratio(portfolio),
                max_drawdown=metrics['max_drawdown'],
                max_drawdown_duration=RiskMetrics.calculate_drawdown_duration(portfolio),
                volatility=RiskMetrics.calculate_volatility(portfolio),
                value_at_risk=RiskMetrics.calculate_value_at_risk(portfolio),
                beta=RiskMetrics.calculate_beta(portfolio),
                
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_trade_return=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                
                avg_position_size=portfolio['position'].mean(),
                max_position_size=portfolio['position'].max(),
                avg_position_duration=0.0,
                
                portfolio=portfolio,
                trades=trades,
                monthly_returns=metrics['monthly_returns'],
                fundamental_data=metrics['fundamental_data']
            )

        return cls(
            strategy_name=metrics['strategy_name'],
            symbol=metrics['symbol'],
            start_date=portfolio.index[0],
            end_date=portfolio.index[-1],
            lot_size=metrics['lot_size'],
            commission_rate=metrics['commission_rate'],
            initial_capital=initial_capital,
            final_portfolio_value=portfolio['total'].iloc[-1],
            total_return=metrics['total_return'],
            annual_return=metrics['annual_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=RiskMetrics.calculate_sortino_ratio(portfolio),
            max_drawdown=metrics['max_drawdown'],
            max_drawdown_duration=RiskMetrics.calculate_drawdown_duration(portfolio),
            volatility=RiskMetrics.calculate_volatility(portfolio),
            value_at_risk=RiskMetrics.calculate_value_at_risk(portfolio),
            beta=RiskMetrics.calculate_beta(portfolio),
            
            total_trades=total_trades,
            winning_trades=len([t for t in trades_with_pnl if t['pnl'] > 0]),
            losing_trades=len([t for t in trades_with_pnl if t['pnl'] < 0]),
            win_rate=metrics['win_rate'],
            profit_factor=cls._calculate_profit_factor(trades_with_pnl),
            avg_trade_return=np.mean([t['pnl'] for t in trades_with_pnl]),
            avg_win=np.mean([t['pnl'] for t in trades_with_pnl if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in trades_with_pnl) else 0.0,
            avg_loss=np.mean([t['pnl'] for t in trades_with_pnl if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades_with_pnl) else 0.0,
            largest_win=max((t['pnl'] for t in trades_with_pnl), default=0.0),
            largest_loss=min((t['pnl'] for t in trades_with_pnl), default=0.0),
            max_consecutive_wins=cls._calculate_consecutive_stats(trades_with_pnl, 'wins'),
            max_consecutive_losses=cls._calculate_consecutive_stats(trades_with_pnl, 'losses'),
            
            avg_position_size=portfolio['position'].mean(),
            max_position_size=portfolio['position'].max(),
            avg_position_duration=cls._calculate_avg_position_duration(trades_with_pnl),
            
            portfolio=portfolio,
            trades=trades,
            monthly_returns=metrics['monthly_returns'],
            fundamental_data=metrics['fundamental_data']
        )

    def generate_report(self, output_dir: str) -> None:
        """
        Generate comprehensive HTML report with visualizations.
        
        Args:
            output_dir (str): Directory path for saving report files
            
        Raises:
            OSError: If unable to create output directory or save files
            ValueError: If required data is missing or invalid
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate all charts
            self._generate_equity_curve(output_dir)
            self._generate_drawdown_chart(output_dir)
            self._generate_monthly_returns_heatmap(output_dir)
            self._generate_trade_distribution(output_dir)
            
            # Generate HTML report
            self._generate_html_report(output_dir)
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate backtest report: {str(e)}")
  
    @staticmethod
    def _calculate_monthly_returns(portfolio: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns from portfolio data"""
        # Keep original monthly returns calculation for the chart
        monthly_returns = portfolio['returns'].resample('ME').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Create DataFrame with returns column
        return pd.DataFrame({'returns': monthly_returns})

    @staticmethod
    def _calculate_profit_factor(trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        profits = sum(t['pnl'] for t in trades if t.get('pnl', 0) > 0)
        losses = abs(sum(t['pnl'] for t in trades if t.get('pnl', 0) < 0))
        return profits / losses if losses != 0 else float('inf')

    @staticmethod
    def _calculate_consecutive_stats(trades: List[Dict[str, Any]], stat_type: str) -> int:
        """Calculate maximum consecutive wins or losses"""
        if not trades:
            return 0
            
        current_streak = max_streak = 0
        for trade in trades:
            pnl = trade.get('pnl', 0)
            if (stat_type == 'wins' and pnl > 0) or (stat_type == 'losses' and pnl < 0):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    @staticmethod
    def _calculate_avg_position_duration(trades: List[Dict[str, Any]]) -> float:
        """Calculate average position duration in days"""
        if not trades:
            return 0.0
        
        durations = []
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades):
                entry = pd.Timestamp(trades[i]['timestamp'])
                exit = pd.Timestamp(trades[i + 1]['timestamp'])
                duration = (exit - entry).total_seconds() / (24 * 3600)  # Convert to days
                durations.append(duration)
        return float(np.mean(durations) if durations else 0.0)

    def _generate_equity_curve(self, output_dir: str) -> None:
        """Generate equity curve plot"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio.index, self.portfolio['total'], label='Portfolio Value')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
        plt.close()

    def _generate_drawdown_chart(self, output_dir: str) -> None:
        """Generate drawdown chart"""
        drawdowns = RiskMetrics.calculate_drawdowns(self.portfolio)
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio.index, drawdowns * 100)
        plt.title('Drawdown Chart')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'drawdown.png'))
        plt.close()

    def _generate_monthly_returns_heatmap(self, output_dir: str) -> None:
        """Generate monthly returns heatmap"""
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(
            self.monthly_returns,
            annot=True,
            fmt='.2%',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Monthly Return'}
        )
        
        # Customize appearance
        plt.title('Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        # Use month names for x-axis
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(np.arange(12) + 0.5, month_labels, rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'monthly_returns.png'))
        plt.close()

    def _generate_trade_distribution(self, output_dir: str) -> None:
        """Generate trade P&L distribution plot"""
        pnls = [trade.get('pnl', 0) for trade in self.trades]
        plt.figure(figsize=(10, 6))
        plt.hist(pnls, bins=50, edgecolor='black')
        plt.title('Trade P&L Distribution')
        plt.xlabel('P&L ($)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'trade_distribution.png'))
        plt.close()

    def _generate_html_report(self, output_dir: str) -> None:
        """Generate HTML report with all metrics and charts"""
        template_path = os.path.join(os.path.dirname(__file__), '../templates/backtest_report.html')
        with open(template_path, 'r') as f:
            template = Template(f.read())
        
        # Reference to grouped_metrics preparation
        grouped_metrics = self._get_metrics_dict()
        
        # Prepare chart data
        equity_curve_data = [[int(t.timestamp() * 1000), v] for t, v in 
                            zip(self.portfolio.index, self.portfolio['total'])]
        
        # Prepare price chart data
        price_data = [[int(t.timestamp() * 1000), float(p)] for t, p in 
                      zip(self.portfolio.index, self.portfolio['close'])]
        
        # Prepare trade annotations
        trade_annotations = []
        for trade in self.trades:
            timestamp = pd.Timestamp(trade['timestamp']).timestamp() * 1000
            price = float(trade['price'])
            trade_type = trade['type'].upper()
            
            annotation = {
                'x': int(timestamp),
                'y': price,
                'title': '↑' if trade_type == 'BUY' else '↓',
                'text': f"{trade_type} @ ${price:,.2f}\nQty: {trade['quantity']}",
                'className': f"trade-{trade_type.lower()}"
            }
            trade_annotations.append(annotation)
        
        # Prepare drawdown series for plotting
        drawdown_data = [[int(t.timestamp() * 1000), d * 100] for t, d in 
                         zip(self.portfolio.index, self._calculate_drawdowns())]
        
        # Format monthly returns data for Highcharts
        monthly_returns_data = [[int(t.timestamp() * 1000), float(r * 100)] 
                               for t, r in self.monthly_returns['returns'].items()]
        
        # Prepare trades data
        trades_data = [{
            'timestamp': pd.Timestamp(trade['timestamp']).strftime('%Y-%m-%d %H:%M') if isinstance(trade['timestamp'], str) 
                    else trade['timestamp'].strftime('%Y-%m-%d %H:%M'),
            'type': trade['type'].upper(),
            'price': float(trade['price']),
            'quantity': int(trade['quantity']),
            'cost': float(abs(trade.get('cost', 0))),
            'commission': float(trade.get('commission', 0)),
            'pnl': float(trade.get('pnl', 0))
        } for trade in self.trades]
        
        # Generate HTML content with all required data
        html_content = template.render(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            grouped_metrics=grouped_metrics,
            equity_curve_data=equity_curve_data,
            price_data=price_data,
            trade_annotations=trade_annotations,
            drawdown_data=drawdown_data,
            monthly_returns_data=monthly_returns_data,
            trades_data=trades_data,
            fundamental_data=self.fundamental_data
        )
        
        # Save HTML report
        report_path = os.path.join(output_dir, 'backtest_report.html')
        with open(report_path, 'w') as f:
            f.write(html_content)

    def _calculate_drawdowns(self) -> pd.Series:
        """Calculate drawdown series for plotting"""
        cumulative_returns = (1 + self.portfolio['returns']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return drawdowns

    def _get_metrics_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Organize metrics into groups for HTML report"""
        metrics = {
            'Strategy Info': [
                {'title': 'Strategy', 'value': self.strategy_name},
                {'title': 'Symbol', 'value': self.symbol},
                {'title': 'Period', 'value': f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}"},
                {'title': 'Initial Capital', 'value': f"${self.initial_capital:,.2f}"},
                {'title': 'Commission Rate', 'value': f"{self.commission_rate*100:.2f}%"}
            ],
            'Performance Metrics': [
                {'title': 'Final Portfolio Value', 'value': f"${self.final_portfolio_value:,.2f}"},
                {'title': 'Total Return', 'value': f"{self.total_return*100:.2f}%", 
                 'color': 'positive' if self.total_return > 0 else 'negative'},
                {'title': 'Annual Return', 'value': f"{self.annual_return*100:.2f}%",
                 'color': 'positive' if self.annual_return > 0 else 'negative'},
                {'title': 'Sharpe Ratio', 'value': f"{self.sharpe_ratio:.2f}"},
                {'title': 'Sortino Ratio', 'value': f"{self.sortino_ratio:.2f}"}
            ],
            'Risk Metrics': [
                {'title': 'Max Drawdown', 'value': f"{self.max_drawdown*100:.2f}%", 'color': 'negative'},
                {'title': 'Max Drawdown Duration', 'value': f"{self.max_drawdown_duration} days"},
                {'title': 'Volatility', 'value': f"{self.volatility*100:.2f}%"},
                {'title': 'Value at Risk (95%)', 'value': f"{self.value_at_risk*100:.2f}%"},
                {'title': 'Beta', 'value': f"{self.beta:.2f}"}
            ],
            'Trading Statistics': [
                {'title': 'Total Trades', 'value': str(self.total_trades)},
                {'title': 'Win Rate', 'value': f"{self.win_rate*100:.2f}%"},
                {'title': 'Profit Factor', 'value': f"{self.profit_factor:.2f}"},
                {'title': 'Average Trade Return', 'value': f"${self.avg_trade_return:,.2f}",
                 'color': 'positive' if self.avg_trade_return > 0 else 'negative'},
                {'title': 'Average Win', 'value': f"${self.avg_win:,.2f}", 'color': 'positive'},
                {'title': 'Average Loss', 'value': f"${self.avg_loss:,.2f}", 'color': 'negative'},
                {'title': 'Largest Win', 'value': f"${self.largest_win:,.2f}", 'color': 'positive'},
                {'title': 'Largest Loss', 'value': f"${self.largest_loss:,.2f}", 'color': 'negative'},
                {'title': 'Max Consecutive Wins', 'value': str(self.max_consecutive_wins)},
                {'title': 'Max Consecutive Losses', 'value': str(self.max_consecutive_losses)}
            ],
            'Position Info': [
                {'title': 'Average Position Size', 'value': f"{self.avg_position_size:,.0f} units"},
                {'title': 'Max Position Size', 'value': f"{self.max_position_size:,.0f} units"},
                {'title': 'Average Position Duration', 'value': f"{self.avg_position_duration:.1f} days"}
            ]
        }
        return metrics
  