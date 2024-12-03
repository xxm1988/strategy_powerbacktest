from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import os
from jinja2 import Template
from .symbol_results import SymbolResults
import logging
import json

@dataclass
class MultiSymbolBacktestReport:
    """Multi-symbol backtest report with aggregated metrics and per-symbol analysis"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float
    symbol_results: Dict[str, SymbolResults]

    def _calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate aggregated portfolio metrics"""
        total_returns = [float(results.metrics['Performance Metrics'][0]['value'].strip('$').replace(',', '')) 
                         for results in self.symbol_results.values()]
                         
        sharpe_ratios = [float(results.metrics['Risk Metrics'][0]['value'].strip('%'))
                         for results in self.symbol_results.values()]
                         
        drawdowns = [int(results.metrics['Risk Metrics'][1]['value'].split()[0])
                     for results in self.symbol_results.values()]
        
        return {
            'total_return': np.mean(total_returns),
            'sharpe_ratio': np.mean(sharpe_ratios),
            'max_drawdown': min(drawdowns),
            'correlation_matrix': self._calculate_correlation_matrix()
        }

    def _calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix between symbols"""
        # Create DataFrame with equity curves for all symbols
        equity_curves = pd.DataFrame({
            symbol: results.equity_curve
            for symbol, results in self.symbol_results.items()
        })
        
        # Calculate returns correlation
        returns = equity_curves.pct_change().dropna()
        return returns.corr()

    def _prepare_correlation_data(self) -> List[List]:
        """Prepare correlation data for heatmap visualization"""
        corr_matrix = self._calculate_correlation_matrix()
        correlation_data = []
        
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                correlation_data.append([
                    i,  # x
                    j,  # y
                    round(corr_matrix.iloc[i, j], 2)  # correlation value
                ])
        
        return correlation_data

    def _prepare_portfolio_equity_data(self) -> List[List]:
        """Prepare combined portfolio equity curve data"""
        # Combine equity curves with equal weighting
        combined_equity = pd.DataFrame({
            symbol: results.equity_curve
            for symbol, results in self.symbol_results.items()
        }).mean(axis=1)
        
        return [[int(t.timestamp() * 1000), float(v)] 
                for t, v in combined_equity.items()]

    def generate_report(self, output_dir: str) -> None:
        """Generate HTML report for multi-symbol backtest"""
        template_path = os.path.join(os.path.dirname(__file__), 
                                   '../templates/multi_symbol_backtest_report.html')
        with open(template_path, 'r') as f:
            template = Template(f.read())

        # Calculate portfolio-level metrics
        portfolio_metrics = self._calculate_portfolio_metrics()

        # Prepare data for template
        correlation_symbols = list(self.symbol_results.keys())
        correlation_data = self._prepare_correlation_data()
        portfolio_equity_data = self._prepare_portfolio_equity_data()

        # Format metrics for display with proper extraction
        overall_metrics = [
            {'title': 'Total Return', 
             'value': f"{float(portfolio_metrics['total_return']):.2f}%",
             'color': 'positive' if portfolio_metrics['total_return'] > 0 else 'negative'},
            {'title': 'Sharpe Ratio', 
             'value': f"{float(portfolio_metrics['sharpe_ratio']):.2f}"},
            {'title': 'Max Drawdown', 
             'value': f"{float(portfolio_metrics['max_drawdown']):.2f}%",
             'color': 'negative'}
        ]

        # Convert numpy types to native Python types in symbol results
        symbol_results_data = {}
        for symbol, results in self.symbol_results.items():
            # Extract metrics properly
            perf_metrics = {m['title']: m['value'] for m in results.metrics['Performance Metrics']}
            risk_metrics = {m['title']: m['value'] for m in results.metrics['Risk Metrics']}
            trade_metrics = {m['title']: m['value'] for m in results.metrics['Trading Statistics']}

            symbol_results_data[symbol] = {
                'trades': [{
                    'timestamp': pd.Timestamp(trade['timestamp']).strftime('%Y-%m-%d %H:%M') if isinstance(trade['timestamp'], str) 
                            else trade['timestamp'].strftime('%Y-%m-%d %H:%M'),
                    'type': trade['type'].upper(),
                    'price': float(trade['price']),
                    'quantity': int(trade['quantity']),
                    'cost': float(abs(trade.get('cost', 0))),
                    'commission': float(trade.get('commission', 0)),
                    'pnl': float(trade.get('pnl', 0))
                } for trade in results.trades],
                'metrics': {
                    'Performance Metrics': [
                        {'title': 'Total Return', 'value': perf_metrics['Total Return'],
                         'color': 'positive' if float(perf_metrics['Total Return'].strip('%')) > 0 else 'negative'},
                        {'title': 'Annual Return', 'value': perf_metrics['Annual Return'],
                         'color': 'positive' if float(perf_metrics['Annual Return'].strip('%')) > 0 else 'negative'},
                        {'title': 'Sharpe Ratio', 'value': perf_metrics['Sharpe Ratio']},
                        {'title': 'Sortino Ratio', 'value': perf_metrics['Sortino Ratio']}
                    ],
                    'Risk Metrics': [
                        {'title': 'Max Drawdown', 'value': risk_metrics['Max Drawdown'], 'color': 'negative'},
                        {'title': 'Volatility', 'value': risk_metrics['Volatility']},
                        {'title': 'Value at Risk', 'value': risk_metrics['Value at Risk (95%)']}
                    ],
                    'Trading Statistics': [
                        {'title': 'Total Trades', 'value': trade_metrics['Total Trades']},
                        {'title': 'Win Rate', 'value': trade_metrics['Win Rate']},
                        {'title': 'Profit Factor', 'value': trade_metrics['Profit Factor']},
                        {'title': 'Average Trade', 'value': trade_metrics['Average Trade Return']}
                    ]
                },
                'total_return': float(perf_metrics['Total Return'].strip('%')),
                'equity_curve': [[int(pd.Timestamp(t).timestamp() * 1000), float(v)] 
                               for t, v in results.equity_curve.items()],
                'drawdown': [[int(pd.Timestamp(t).timestamp() * 1000), float(v)] 
                            for t, v in results.drawdown.items()],
                'monthly_returns': [[int(pd.Timestamp(t).timestamp() * 1000), float(v)] 
                                  for t, v in results.monthly_returns['returns'].items()],
                'trade_annotations': [{
                    'x': int(pd.Timestamp(ann['x']).timestamp() * 1000),
                    'y': float(ann['y']),
                    'type': str(ann.get('type', '')),
                    'quantity': int(ann.get('quantity', 0)),
                    'text': str(ann.get('text', ''))
                } for ann in results.trade_annotations]
            }

        # Generate HTML content
        html_content = template.render(
            strategy_name=self.strategy_name,
            start_date=self.start_date.strftime('%Y-%m-%d'),
            end_date=self.end_date.strftime('%Y-%m-%d'),
            overall_metrics=overall_metrics,
            correlation_symbols=correlation_symbols,
            correlation_data=correlation_data,
            portfolio_equity_data=portfolio_equity_data,
            symbol_results=symbol_results_data
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write HTML file
        output_path = os.path.join(output_dir, 'multi_symbol_backtest_report.html')
        with open(output_path, 'w') as f:
            f.write(html_content)