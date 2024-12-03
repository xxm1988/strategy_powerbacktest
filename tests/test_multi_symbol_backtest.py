import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.engine.multi_symbol_backtest_report import MultiSymbolBacktestReport, SymbolResults
from src.engine.backtest_report import BacktestReport
from src.engine.fundamental_data import FundamentalData

@pytest.fixture
def sample_dates():
    """Create sample date range"""
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    return start_date, end_date, dates

@pytest.fixture
def sample_portfolio(sample_dates):
    """Create sample portfolio DataFrame"""
    _, _, dates = sample_dates
    return pd.DataFrame({
        'close': np.random.uniform(90, 110, len(dates)),
        'total': np.linspace(100000, 125000, len(dates))
    }, index=dates)

@pytest.fixture
def sample_trades():
    """Create sample trades list"""
    return [
        {
            'timestamp': datetime(2023, 2, 1),
            'type': 'BUY',
            'price': 95.0,
            'quantity': 100,
            'cost': 9500.0,
            'commission': 9.5,
            'pnl': 0.0
        },
        {
            'timestamp': datetime(2023, 3, 1),
            'type': 'SELL',
            'price': 105.0,
            'quantity': 100,
            'cost': 10500.0,
            'commission': 10.5,
            'pnl': 985.0
        }
    ]

@pytest.fixture
def sample_backtest_report(sample_portfolio, sample_trades, sample_dates):
    """Create sample BacktestReport instance"""
    start_date, end_date, _ = sample_dates
    metrics = {
        'strategy_name': 'Test Strategy',
        'symbol': 'AAPL',
        'lot_size': 100,
        'commission_rate': 0.001,
        'total_return': 0.25,
        'annual_return': 0.15,
        'sharpe_ratio': 1.8,
        'sortino_ratio': 2.1,
        'max_drawdown': -0.12,
        'max_drawdown_duration': 30,
        'volatility': 0.2,
        'value_at_risk': 0.15,
        'beta': 1.1,
        'total_trades': len(sample_trades),
        'winning_trades': 1,
        'losing_trades': 0,
        'win_rate': 1.0,
        'profit_factor': 2.0,
        'avg_trade_return': 985.0,
        'avg_win': 985.0,
        'avg_loss': 0.0,
        'largest_win': 985.0,
        'largest_loss': 0.0,
        'max_consecutive_wins': 1,
        'max_consecutive_losses': 0
    }
    
    return BacktestReport.from_backtest_results(
        portfolio=sample_portfolio,
        trades=sample_trades,
        initial_capital=100000.0,
        metrics=metrics
    )

class TestSymbolResults:
    def test_from_backtest_report(self, sample_backtest_report):
        """Test creation of SymbolResults from BacktestReport"""
        symbol_results = SymbolResults.from_backtest_report(sample_backtest_report)
        
        assert symbol_results.symbol == 'AAPL'
        assert len(symbol_results.trades) == 2
        assert isinstance(symbol_results.equity_curve, pd.Series)
        assert isinstance(symbol_results.drawdown, pd.Series)
        assert isinstance(symbol_results.monthly_returns, pd.DataFrame)
        assert isinstance(symbol_results.metrics, dict)
        assert len(symbol_results.trade_annotations) > 0

class TestMultiSymbolBacktestReport:
    def test_report_creation(self, sample_backtest_report):
        """Test creation of multi-symbol report"""
        symbol_results = {
            'AAPL': SymbolResults.from_backtest_report(sample_backtest_report),
            'GOOGL': SymbolResults.from_backtest_report(sample_backtest_report)
        }
        
        report = MultiSymbolBacktestReport(
            strategy_name='Test Multi-Symbol Strategy',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=200000.0,
            commission_rate=0.001,
            symbol_results=symbol_results
        )
        
        assert report.strategy_name == 'Test Multi-Symbol Strategy'
        assert len(report.symbol_results) == 2
        assert isinstance(report._calculate_correlation_matrix(), pd.DataFrame)

    def test_portfolio_metrics(self, sample_backtest_report):
        """Test portfolio-level metrics calculation"""
        symbol_results = {
            'AAPL': SymbolResults.from_backtest_report(sample_backtest_report),
            'GOOGL': SymbolResults.from_backtest_report(sample_backtest_report)
        }
        
        report = MultiSymbolBacktestReport(
            strategy_name='Test Multi-Symbol Strategy',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=200000.0,
            commission_rate=0.001,
            symbol_results=symbol_results
        )
        
        metrics = report._calculate_portfolio_metrics()
        assert isinstance(metrics['total_return'], float)
        assert isinstance(metrics['sharpe_ratio'], float)
        assert isinstance(metrics['max_drawdown'], float)
        assert isinstance(metrics['correlation_matrix'], pd.DataFrame)

    def test_report_generation(self, sample_backtest_report, tmp_path):
        """Test HTML report generation"""
        symbol_results = {
            'AAPL': SymbolResults.from_backtest_report(sample_backtest_report),
            'GOOGL': SymbolResults.from_backtest_report(sample_backtest_report)
        }
        
        report = MultiSymbolBacktestReport(
            strategy_name='Test Multi-Symbol Strategy',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=200000.0,
            commission_rate=0.001,
            symbol_results=symbol_results
        )
        
        output_dir = tmp_path / "reports"
        report.generate_report(str(output_dir))
        
        assert (output_dir / "multi_symbol_backtest_report.html").exists()
        
    def test_correlation_data_preparation(self, sample_backtest_report):
        """Test correlation data preparation for heatmap"""
        symbol_results = {
            'AAPL': SymbolResults.from_backtest_report(sample_backtest_report),
            'GOOGL': SymbolResults.from_backtest_report(sample_backtest_report)
        }
        
        report = MultiSymbolBacktestReport(
            strategy_name='Test Multi-Symbol Strategy',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=200000.0,
            commission_rate=0.001,
            symbol_results=symbol_results
        )
        
        correlation_matrix = report._calculate_correlation_matrix()
        assert isinstance(correlation_matrix, pd.DataFrame)
        assert correlation_matrix.shape == (2, 2) 