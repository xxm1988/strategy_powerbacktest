from typing import Optional
import pandas as pd
import numpy as np


class RiskMetrics:
    """
    Handles all risk-related metric calculations for the backtest report.
    """

    @staticmethod
    def calculate_sortino_ratio(
        portfolio: pd.DataFrame, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio using negative returns only"""
        returns = portfolio["returns"]
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return np.inf

        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_std = np.sqrt(252) * np.sqrt(np.mean(negative_returns**2))
        return excess_returns / downside_std if downside_std != 0 else 0

    @staticmethod
    def calculate_drawdown_duration(portfolio: pd.DataFrame) -> int:
        """Calculate maximum drawdown duration in days"""
        cumulative_returns = (1 + portfolio["returns"]).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1

        is_drawdown = drawdowns < 0
        drawdown_periods = pd.Series(range(len(drawdowns)), index=drawdowns.index)
        drawdown_periods.loc[~is_drawdown] = np.nan
        drawdown_periods = drawdown_periods.ffill()

        if drawdown_periods.empty:
            return 0

        drawdown_groups = (drawdown_periods.diff() != 0).cumsum()
        durations = drawdown_groups.value_counts()
        return int(durations.max()) if not durations.empty else 0

    @staticmethod
    def calculate_value_at_risk(
        portfolio: pd.DataFrame, confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk"""
        returns = portfolio["returns"]
        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def calculate_beta(
        portfolio: pd.DataFrame, market_returns: Optional[pd.Series] = None
    ) -> float:
        """Calculate portfolio beta against market returns"""
        if market_returns is None:
            return 1.0

        returns = portfolio["returns"]
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        return covariance / market_variance if market_variance != 0 else 1.0

    @staticmethod
    def calculate_drawdowns(portfolio: pd.DataFrame) -> pd.Series:
        """Calculate drawdown series for plotting"""
        cumulative_returns = (1 + portfolio["returns"]).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        return cumulative_returns / rolling_max - 1

    @staticmethod
    def calculate_volatility(portfolio: pd.DataFrame) -> float:
        """Calculate annualized volatility"""
        return portfolio["returns"].std() * np.sqrt(252)

    @staticmethod
    def calculate_max_drawdown(portfolio: pd.DataFrame) -> float:
        """Calculate maximum drawdown percentage"""
        cumulative_returns = (1 + portfolio["returns"]).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return drawdowns.min()
