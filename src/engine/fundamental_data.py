from dataclasses import dataclass
from typing import Optional


@dataclass
class FundamentalData:
    """Company fundamental data from Futu API"""

    market_cap: float
    pe_ratio: float
    pb_ratio: float
    dividend_yield: float
    industry: str
    lot_size: int
    stock_name: str
    stock_type: str
    listing_date: str
    total_shares: float
    float_shares: float
    current_price: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    avg_volume_3m: float
    eps: Optional[float] = None
    net_profit_ttm: Optional[float] = None
    net_profit_growth_ttm: Optional[float] = None
