import pytest
import numpy as np
import pandas as pd
from src.strategy.moving_average_strategy import MovingAverageCrossStrategy

class TestMovingAverageCrossStrategy:
    @pytest.fixture
    def strategy(self):
        return MovingAverageCrossStrategy({
            'short_window': 20,
            'long_window': 50
        })

    @pytest.fixture
    def sample_data(self):
        """Create sample price data with known patterns"""
        np.random.seed(42)
        periods = 100
        
        # Generate trending price data
        base = np.linspace(100, 200, periods)
        noise = np.random.normal(0, 2, periods)
        prices = base + noise
        
        df = pd.DataFrame({
            'close': prices,
            'timestamp': pd.date_range(start='2024-01-01', periods=periods, freq='h')
        })
        return df

    def test_calculate_indicators(self, strategy, sample_data):
        """Test that moving averages are calculated correctly"""
        data = strategy.calculate_indicators(sample_data)
        
        # Check that both MAs exist
        assert 'SMA_short' in data.columns
        assert 'SMA_long' in data.columns

        # Verify MA calculations for a few known points
        # First 19 points of short MA should be calculated (min_periods=1)
        assert not pd.isna(data['SMA_short'].iloc[0])
        
        # First 49 points of long MA should be calculated (min_periods=1)
        assert not pd.isna(data['SMA_long'].iloc[0])
        
        # Verify actual MA calculation for a specific point
        test_idx = 60
        expected_short_ma = sample_data['close'].iloc[test_idx-20+1:test_idx+1].mean()
        expected_long_ma = sample_data['close'].iloc[test_idx-50+1:test_idx+1].mean()
        
        assert abs(data['SMA_short'].iloc[test_idx] - expected_short_ma) < 0.0001
        assert abs(data['SMA_long'].iloc[test_idx] - expected_long_ma) < 0.0001

    def test_generate_signals(self, strategy, sample_data):
        """Test signal generation logic"""
        signals = strategy.generate_signals(sample_data)
        
        # Verify signals are valid
        assert isinstance(signals, pd.Series)
        assert signals.isin([-1, 0, 1]).all()
        assert len(signals) == len(sample_data)

        # Get the indicator data
        data = strategy.calculate_indicators(sample_data)
        
        # Verify signal logic
        for i in range(len(signals)):
            if data['SMA_short'].iloc[i] > data['SMA_long'].iloc[i]:
                assert signals.iloc[i] == 1
            elif data['SMA_short'].iloc[i] < data['SMA_long'].iloc[i]:
                assert signals.iloc[i] == -1
            else:
                assert signals.iloc[i] == 0

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test valid parameters
        strategy = MovingAverageCrossStrategy({
            'short_window': 20,
            'long_window': 50
        })
        assert strategy.validate_parameters() == True

        # Test invalid parameters
        with pytest.raises(ValueError):
            MovingAverageCrossStrategy({
                'short_window': 50,
                'long_window': 20
            })

        with pytest.raises(ValueError):
            MovingAverageCrossStrategy({
                'short_window': 0,
                'long_window': 50
            })

    def test_warmup_period(self, strategy):
        """Test warmup period calculation"""
        assert strategy.calculate_warmup_period() == 50  # Should be long_window

    def test_default_parameters(self):
        """Test default parameter initialization"""
        strategy = MovingAverageCrossStrategy({})
        assert strategy.short_window == 20
        assert strategy.long_window == 50

    def test_edge_cases(self, strategy):
        """Test edge cases with minimal data"""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        with pytest.raises(KeyError):  # Should raise error due to missing 'close' column
            strategy.calculate_indicators(empty_df)

        # Test with single row
        single_row = pd.DataFrame({'close': [100]})
        result = strategy.calculate_indicators(single_row)
        assert result['SMA_short'].iloc[0] == 100  # Single value MA equals the value
        assert result['SMA_long'].iloc[0] == 100

    def test_crossover_signals(self):
        """Test specific crossover scenarios"""
        # Create a scenario where short MA crosses above long MA
        strategy = MovingAverageCrossStrategy({'short_window': 2, 'long_window': 3})
        df = pd.DataFrame({
            'close': [10, 11, 12, 13, 14]  # Rising prices
        })
        
        signals = strategy.generate_signals(df)
        
        # In a rising market, we should see a buy signal
        assert 1 in signals.values
        
        # Create a scenario where short MA crosses below long MA
        df = pd.DataFrame({
            'close': [14, 13, 12, 11, 10]  # Falling prices
        })
        
        signals = strategy.generate_signals(df)
        
        # In a falling market, we should see a sell signal
        assert -1 in signals.values