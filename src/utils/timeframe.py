from futu import KLType


def get_kl_type(timeframe: str) -> KLType:
    """Convert timeframe string to KLType enum"""
    timeframe_map = {
        "1M": KLType.K_1M,
        "3M": KLType.K_3M,
        "5M": KLType.K_5M,
        "15M": KLType.K_15M,
        "30M": KLType.K_30M,
        "60M": KLType.K_60M,
        "DAY": KLType.K_DAY,
        "WEEK": KLType.K_WEEK,
        "MON": KLType.K_MON,
    }

    # Handle custom timeframes
    custom_timeframes = {"4H": "60M", "2H": "60M", "3H": "60M"}

    timeframe = timeframe.upper()
    if timeframe in custom_timeframes:
        timeframe = custom_timeframes[timeframe]

    if timeframe not in timeframe_map:
        raise ValueError(
            f"Invalid timeframe: {timeframe}. Must be one of {list(timeframe_map.keys())}"
        )

    return timeframe_map[timeframe]
