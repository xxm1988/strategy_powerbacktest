from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

@dataclass
class BacktestReport:
    """Enhanced backtest report with comprehensive analytics"""
    initial_capital: float
    final_portfolio_value: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades: pd.DataFrame
    equity_curve: pd.Series
    monthly_returns: pd.Series
    realized_pnl: float
    floating_pnl: float
    total_pnl: float
    total_trades: int
    open_positions: int
    
    @classmethod
    def from_backtest_results(cls, portfolio: pd.DataFrame, trades: List[Dict], initial_capital: float, metrics: Dict[str, Any]):
        """Create a backtest report from raw backtest results"""
        equity_curve = portfolio['total']
        returns = portfolio['returns']
        
        # Ensure we have a datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("Portfolio returns must have a DatetimeIndex")
        
        # Monthly returns using 'ME' (month end) frequency
        monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(trades)
        
        return cls(
            initial_capital=initial_capital,
            final_portfolio_value=portfolio['total'].iloc[-1],
            total_return=metrics['total_return'] * 100,  # Convert to percentage
            annual_return=metrics['annual_return'] * 100,  # Convert to percentage
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'] * 100,  # Convert to percentage
            win_rate=metrics['win_rate'] * 100,  # Convert to percentage
            profit_factor=metrics['total_pnl'] / abs(metrics['realized_pnl']) if metrics['realized_pnl'] < 0 else float('inf'),
            trades=trades_df,
            equity_curve=equity_curve,
            monthly_returns=monthly_returns,
            realized_pnl=metrics['realized_pnl'],
            floating_pnl=metrics['floating_pnl'],
            total_pnl=metrics['total_pnl'],
            total_trades=metrics['total_trades'],
            open_positions=metrics['open_positions']
        )
    
    def _group_metrics(self) -> Dict[str, List[Dict]]:
        """Group metrics into logical categories"""
        return {
            "Portfolio Statistics": [
                {"title": "Initial Capital", "value": f"${self.initial_capital:,.2f}"},
                {"title": "Final Value", "value": f"${self.final_portfolio_value:,.2f}"},
                {"title": "Total Return", "value": f"{self.total_return:.2f}%", 
                 "color": "positive" if self.total_return > 0 else "negative"},
                {"title": "Annual Return", "value": f"{self.annual_return:.2f}%"}
            ],
            "Risk Metrics": [
                {"title": "Sharpe Ratio", "value": f"{self.sharpe_ratio:.2f}"},
                {"title": "Maximum Drawdown", "value": f"{self.max_drawdown:.2f}%"},
                {"title": "Win Rate", "value": f"{self.win_rate:.2f}%"},
                {"title": "Profit Factor", "value": f"{self.profit_factor:.2f}"}
            ],
            "Trading Statistics": [
                {"title": "Total Trades", "value": str(self.total_trades)},
                {"title": "Realized P&L", "value": f"${self.realized_pnl:,.2f}",
                 "color": "positive" if self.realized_pnl > 0 else "negative"},
                {"title": "Floating P&L", "value": f"${self.floating_pnl:,.2f}",
                 "color": "positive" if self.floating_pnl > 0 else "negative"},
                {"title": "Open Positions", "value": str(self.open_positions)}
            ]
        }

    def _prepare_trades_data(self) -> List[Dict]:
        """Prepare trades data for DataTables"""
        return [
            {
                "timestamp": trade["timestamp"] if isinstance(trade["timestamp"], str) 
                            else trade["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "type": trade["type"].upper(),
                "price": trade["price"],
                "quantity": trade["quantity"],
                "cost": trade["cost"],
                "commission": trade["commission"],
                "pnl": trade["pnl"]
            }
            for trade in self.trades.to_dict('records')
        ]

    def _prepare_highcharts_data(self, series: pd.Series) -> List[List]:
        """Convert pandas Series to Highcharts-compatible format"""
        return [
            [int(timestamp.timestamp() * 1000), float(value)]
            for timestamp, value in series.items()
        ]

    def _calculate_drawdown(self) -> pd.Series:
        """Calculate drawdown series"""
        return (self.equity_curve / self.equity_curve.expanding().max() - 1) * 100

    def generate_report(self, output_dir: str = "reports") -> str:
        """Generate an enhanced HTML report with interactive charts and tables"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare all data for the template
        template_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "grouped_metrics": self._group_metrics(),
            "equity_curve_data": self._prepare_highcharts_data(self.equity_curve),
            "drawdown_data": self._prepare_highcharts_data(self._calculate_drawdown()),
            "monthly_returns_data": self._prepare_highcharts_data(self.monthly_returns * 100),
            "trades_data": self._prepare_trades_data()
        }
        
        # Generate and save report
        template = self._get_template()
        report_html = template.render(**template_data)
        report_path = os.path.join(output_dir, 
                                 f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        with open(report_path, 'w') as f:
            f.write(report_html)
            
        return report_path
    
    def _get_metrics_dict(self) -> Dict[str, Any]:
        """Get formatted metrics dictionary"""
        return {
            "Initial Capital": f"${self.initial_capital:,.2f}",
            "Final Portfolio Value": f"${self.final_portfolio_value:,.2f}",
            "Total Return": f"{self.total_return:.2f}%",
            "Annual Return": f"{self.annual_return:.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Maximum Drawdown": f"{self.max_drawdown:.2f}%",
            "Win Rate": f"{self.win_rate:.2f}%",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Number of Trades": f"{self.total_trades}",
            "Realized PnL": f"${self.realized_pnl:,.2f}",
            "Floating PnL": f"${self.floating_pnl:,.2f}",
            "Total PnL": f"${self.total_pnl:,.2f}",
            "Open Positions": f"{self.open_positions}"
        }
    
    def _generate_plots(self, output_dir: str):
        """Generate and save analysis plots"""
        # Equity curve
        plt.figure(figsize=(12, 6))
        self.equity_curve.plot()
        plt.title('Portfolio Equity Curve')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
        plt.close()
        
        # Monthly returns heatmap
        monthly_returns_table = self.monthly_returns.round(4) * 100
        # Convert to a proper month/year format for the heatmap
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns_table.index.year,
            'Month': monthly_returns_table.index.month,
            'Returns': monthly_returns_table.values
        })
        
        # Create pivot table for heatmap
        pivot_table = monthly_returns_df.pivot(
            index='Year',
            columns='Month',
            values='Returns'
        )
        
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Monthly Returns (%)'}
        )
        plt.title('Monthly Returns (%)')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.savefig(os.path.join(output_dir, 'monthly_returns.png'))
        plt.close()
    
    def _get_template(self):
        """Get Jinja2 template for report generation"""
        import jinja2
        
        template_loader = jinja2.FileSystemLoader(searchpath="src/templates")
        template_env = jinja2.Environment(loader=template_loader)
        return template_env.get_template("backtest_report.html")
  