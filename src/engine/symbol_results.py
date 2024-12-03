from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
from .backtest_report import BacktestReport

@dataclass
class SymbolResults:
    """Results for individual symbol"""
    symbol: str
    trades: List[Dict]
    metrics: Dict[str, float]
    equity_curve: pd.Series
    drawdown: pd.Series
    monthly_returns: pd.DataFrame
    price_data: pd.Series
    trade_annotations: List[Dict]

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
            price_data=report.portfolio['close'],
            trade_annotations=report._prepare_trade_annotations()
        ) 