from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd
from .backtest_report import BacktestReport

@dataclass
class SymbolResults:
    """Results for individual symbol in multi-symbol backtest"""
    symbol: str
    trades: List[Dict[str, Any]]
    metrics: Dict[str, List[Dict[str, Any]]]
    equity_curve: pd.Series
    drawdown: pd.Series
    monthly_returns: pd.DataFrame
    trade_annotations: List[Dict[str, Any]]
    portfolio: pd.DataFrame
    benchmark_data: pd.DataFrame

    @classmethod
    def from_backtest_report(cls, report: BacktestReport) -> 'SymbolResults':
        """Create SymbolResults from a BacktestReport instance"""
        return cls(
            symbol=report.symbol,
            trades=report.trades,
            metrics=report._get_metrics_dict(),
            equity_curve=report.portfolio['total'],
            drawdown=report._calculate_drawdowns(),
            monthly_returns=report.monthly_returns,
            trade_annotations=report._prepare_trade_annotations(),
            portfolio=report.portfolio,
            benchmark_data=report.benchmark_data
        ) 