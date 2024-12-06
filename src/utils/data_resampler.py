from typing import Optional
import pandas as pd

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe, respecting HKEX trading hours
    
    Args:
        df: DataFrame with columns ['time_key', 'open', 'high', 'low', 'close', 'volume']
        timeframe: Target timeframe (e.g., '4H', '1H', etc.)
        
    Returns:
        Resampled DataFrame
    """
    # Add input validation at the start of the function
    valid_timeframes = ['1min', '5min', '15min', '30min', '1H', '2H', '4H', '1D']
    if timeframe not in valid_timeframes:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")
    
    # Convert time_key to datetime index if it's not already
    if 'time_key' in df.columns:
        df = df.set_index(pd.to_datetime(df['time_key']))
    
    # Define HKEX trading sessions
    def custom_trading_session(timestamp, timeframe: str):
        """Return trading session label for timestamp based on timeframe"""
        if timeframe == '4H':
            # Morning session (09:30-12:00) and Afternoon session (13:00-16:00)
            if timestamp.hour <= 12:
                return pd.Timestamp(timestamp.date()) + pd.Timedelta(hours=12)
            else:
                return pd.Timestamp(timestamp.date()) + pd.Timedelta(hours=16)
        elif timeframe == '2H':
            # Four sessions: 11:30, 12:00, 15:00, 16:00
            if timestamp.hour < 11 or (timestamp.hour == 11 and timestamp.minute < 30):
                return pd.Timestamp(timestamp.date()) + pd.Timedelta(hours=11, minutes=30)
            elif timestamp.hour < 13:
                return pd.Timestamp(timestamp.date()) + pd.Timedelta(hours=12)
            elif timestamp.hour <= 15:
                return pd.Timestamp(timestamp.date()) + pd.Timedelta(hours=15)
            else:
                return pd.Timestamp(timestamp.date()) + pd.Timedelta(hours=16)
        else:
            raise ValueError(f"Unsupported timeframe for custom sessions: {timeframe}")
    
    # Group data by trading session
    df['session'] = df.index.map(lambda x: custom_trading_session(x, timeframe))
    
    # Group by session and calculate OHLCV
    resampled = df.groupby('session').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Reset index and rename time column
    resampled = resampled.reset_index()
    resampled = resampled.rename(columns={'session': 'time_key'})
    
    return resampled 