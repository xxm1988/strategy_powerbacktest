from typing import Optional
import pandas as pd

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe
    
    Args:
        df: DataFrame with columns ['time_key', 'open', 'high', 'low', 'close', 'volume']
        timeframe: Target timeframe (e.g., '4H', '1H', etc.)
        
    Returns:
        Resampled DataFrame
    """
    # Convert time_key to datetime index if it's not already
    if 'time_key' in df.columns:
        df = df.set_index(pd.to_datetime(df['time_key']))
    
    # Define resampling rules
    ohlcv_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Resample data
    resampled = df.resample(timeframe).agg(ohlcv_dict)
    
    # Reset index and rename time column
    resampled = resampled.reset_index()
    resampled = resampled.rename(columns={'index': 'time_key'})
    
    # Drop rows with NaN values (incomplete periods)
    resampled = resampled.dropna()
    
    return resampled 