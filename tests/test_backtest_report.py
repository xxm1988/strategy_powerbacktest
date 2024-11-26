import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.engine.backtest_report import BacktestReport
from src.engine.fundamental_data import FundamentalData

@pytest.fixture
def sample_dates():
    start = datetime(2023, 1, 1)
    end = datetime(2023, 12, 31)
    return pd.date_range(start=start, end=end, freq='D')

@pytest.fixture
def sample_portfolio(sample_dates):
    """Create sample portfolio data"""
    data = {
        'total': [100000 * (1 + 0.001 * i) for i in range(len(sample_dates))],
        'position': [100] * len(sample_dates),
        'close': [150 + i * 0.1 for i in range(len(sample_dates))],
        'returns': [0.001] * len(sample_dates)
    }
    return pd.DataFrame(data, index=sample_dates)

@pytest.fixture
def sample_trades():
    """Create sample trades data"""
    return [
        {
            'timestamp': datetime(2023, 1, 15),
            'type': 'BUY',
            'price': 150.5,
            'quantity': 100,
            'commission': 15.05,
            'cost': 15065.05,
            'pnl': None
        },
        {
            'timestamp': datetime(2023, 6, 30),
            'type': 'SELL',
            'price': 168.75,
            'quantity': 100,
            'commission': 16.88,
            'pnl': 1826.83
        }
    ]

@pytest.fixture
def sample_fundamental_data():
    """Create sample fundamental data"""
    return FundamentalData(
        market_cap=1000000000.0,
        pe_ratio=15.5,
        pb_ratio=2.3,
        dividend_yield=2.5,
        industry='Technology',
        lot_size=100,
        stock_name='Test Stock',
        stock_type='Common Stock',
        listing_date='2020-01-01',
        total_shares=1000000,
        float_shares=800000,
        current_price=100.0,
        fifty_two_week_high=120.0,
        fifty_two_week_low=80.0,
        avg_volume_3m=50000
    )

@pytest.fixture
def sample_metrics(sample_fundamental_data):
    """Create sample metrics dictionary"""
    return {
        'strategy_name': 'Test Strategy',
        'symbol': 'TEST',
        'lot_size': 100,
        'commission_rate': 0.001,
        'total_return': 0.25,
        'annual_return': 0.15,
        'sharpe_ratio': 1.8,
        'max_drawdown': -0.12,
        'win_rate': 0.65,
        'monthly_returns': pd.DataFrame({'returns': [0.02] * 12}),
        'fundamental_data': sample_fundamental_data
    }

class TestBacktestReport:
    def test_report_creation(self, sample_portfolio, sample_trades, sample_metrics):
        """Test basic report creation"""
        report = BacktestReport.from_backtest_results(
            portfolio=sample_portfolio,
            trades=sample_trades,
            initial_capital=100000.0,
            metrics=sample_metrics
        )
        
        assert report.strategy_name == 'Test Strategy'
        assert report.symbol == 'TEST'
        assert report.initial_capital == 100000.0
        assert report.total_trades == len(sample_trades)
        assert isinstance(report.portfolio, pd.DataFrame)

    def test_metrics_calculation(self, sample_portfolio, sample_trades, sample_metrics):
        """Test metrics calculations"""
        report = BacktestReport.from_backtest_results(
            portfolio=sample_portfolio,
            trades=sample_trades,
            initial_capital=100000.0,
            metrics=sample_metrics
        )
        
        assert report.total_return == 0.25
        assert report.annual_return == 0.15
        assert report.sharpe_ratio == 1.8
        assert report.win_rate == 0.65
        assert report.max_drawdown == -0.12

    def test_position_calculations(self, sample_portfolio, sample_trades, sample_metrics):
        """Test position-related calculations"""
        report = BacktestReport.from_backtest_results(
            portfolio=sample_portfolio,
            trades=sample_trades,
            initial_capital=100000.0,
            metrics=sample_metrics
        )
        
        assert report.avg_position_size == 100
        assert report.max_position_size == 100
        assert isinstance(report.avg_position_duration, float)

    def test_trade_statistics(self, sample_portfolio, sample_trades, sample_metrics):
        """Test trade statistics calculations"""
        report = BacktestReport.from_backtest_results(
            portfolio=sample_portfolio,
            trades=sample_trades,
            initial_capital=100000.0,
            metrics=sample_metrics
        )
        
        assert report.total_trades == 2
        assert report.winning_trades == 1
        assert report.losing_trades == 0
        assert isinstance(report.profit_factor, float)

    def test_edge_cases(self, sample_portfolio, sample_metrics):
        """Test edge cases with empty trades"""
        report = BacktestReport.from_backtest_results(
            portfolio=sample_portfolio,
            trades=[],
            initial_capital=100000.0,
            metrics=sample_metrics
        )
        
        assert report.total_trades == 0
        assert report.winning_trades == 0
        assert report.losing_trades == 0
        assert report.avg_position_duration == 0