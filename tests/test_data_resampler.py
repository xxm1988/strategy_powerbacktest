import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.data_resampler import resample_ohlcv


@pytest.fixture
def sample_minute_data():
    """Create sample 1-minute OHLCV data"""
    start_date = datetime(2023, 1, 1, 9, 30)  # Market open
    periods = 390  # Full trading day in minutes

    dates = [start_date + timedelta(minutes=i) for i in range(periods)]
    data = {
        "time_key": dates,
        "open": np.random.uniform(100, 101, periods),
        "high": np.random.uniform(101, 102, periods),
        "low": np.random.uniform(99, 100, periods),
        "close": np.random.uniform(100, 101, periods),
        "volume": np.random.randint(1000, 5000, periods),
    }

    # Ensure high is highest and low is lowest
    for i in range(periods):
        data["high"][i] = max(data["open"][i], data["close"][i], data["high"][i])
        data["low"][i] = min(data["open"][i], data["close"][i], data["low"][i])

    return pd.DataFrame(data)


def test_resample_to_4hour(sample_minute_data):
    """Test resampling from 1-minute to 4-hour data"""
    resampled = resample_ohlcv(sample_minute_data, "4H")

    # For a full trading day, expect 2 4-hour candles
    assert len(resampled) == 2
    assert resampled["open"].iloc[0] == sample_minute_data["open"].iloc[0]
    assert resampled["close"].iloc[-1] == sample_minute_data["close"].iloc[-1]
    assert all(resampled["volume"] > 0)


def test_ohlc_integrity(sample_minute_data):
    """Test that OHLC relationships are maintained after resampling"""
    resampled = resample_ohlcv(sample_minute_data, "2H")

    # Check OHLC relationships
    assert all(resampled["high"] >= resampled["open"])
    assert all(resampled["high"] >= resampled["close"])
    assert all(resampled["low"] <= resampled["open"])
    assert all(resampled["low"] <= resampled["close"])


def test_empty_dataframe():
    """Test handling of empty DataFrame"""
    empty_df = pd.DataFrame(
        columns=["time_key", "open", "high", "low", "close", "volume"]
    )
    resampled = resample_ohlcv(empty_df, "1H")
    assert len(resampled) == 0
    assert all(
        col in resampled.columns for col in ["open", "high", "low", "close", "volume"]
    )


def test_invalid_timeframe():
    """Test handling of invalid timeframe"""
    df = pd.DataFrame(
        {
            "time_key": pd.date_range("2023-01-01", periods=5, freq="1min"),
            "open": [1, 2, 3, 4, 5],
            "high": [1.5, 2.5, 3.5, 4.5, 5.5],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.2, 2.2, 3.2, 4.2, 5.2],
            "volume": [100, 200, 300, 400, 500],
        }
    )

    with pytest.raises(ValueError):
        resample_ohlcv(df, "invalid")
