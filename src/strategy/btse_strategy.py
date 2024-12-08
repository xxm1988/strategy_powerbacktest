from typing import Dict
import numpy as np
import pandas as pd
import talib
from .base_strategy import BaseStrategy


class BTSEStrategy(BaseStrategy):
    def __init__(self, parameters: Dict = None):
        super().__init__(parameters)
        self.lookback_period = parameters.get("lookback_period", 25)
        self.zig_threshold = parameters.get("zig_threshold", 10)  # 50/5 = 10%
        self.ma_period = parameters.get("ma_period", 2)

    def calculate_zig(self, data: pd.DataFrame) -> pd.Series:
        """
        Implement ZIG function that tracks price reversals
        ZIG(3, N) where 3 means tracking close price and N is percentage threshold
        """
        close = data["close"].values
        threshold = self.zig_threshold / 100  # Convert percentage to decimal
        zig = np.zeros_like(close)
        last_zig = close[0]
        trend = 0  # 0: init, 1: up, -1: down

        for i in range(1, len(close)):
            price = close[i]
            price_change = (price - last_zig) / last_zig

            if trend == 0:
                if abs(price_change) >= threshold:
                    trend = 1 if price_change > 0 else -1
                    last_zig = price
            elif trend == 1:  # Up trend
                if price > last_zig:
                    last_zig = price
                elif (last_zig - price) / last_zig >= threshold:
                    trend = -1
                    last_zig = price
            else:  # Down trend
                if price < last_zig:
                    last_zig = price
                elif (price - last_zig) / last_zig >= threshold:
                    trend = 1
                    last_zig = price

            zig[i] = last_zig

        return pd.Series(zig * 100, index=data.index)  # Multiply by 100 as per original

    def calculate_strength_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate STR1-STR9 exactly as in original script"""
        high, low, close = data["high"], data["low"], data["close"]
        prev_close = close.shift(1)

        # STR1:=SUM(MAX(MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1))),ABS(LOW-REF(CLOSE,1))),25)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        str1 = pd.DataFrame([tr1, tr2, tr3]).max().rolling(self.lookback_period).sum()

        # STR2:=HIGH-REF(HIGH,1)
        str2 = high - high.shift(1)

        # STR3:=REF(LOW,1)-LOW
        str3 = low.shift(1) - low

        # STR4:=SUM(IF(STR2>0 AND STR2>STR3,STR2,0),25)
        str4 = (
            pd.Series(np.where((str2 > 0) & (str2 > str3), str2, 0))
            .rolling(self.lookback_period)
            .sum()
        )

        # STR5:=SUM(IF(STR3>0 AND STR3>STR2,STR3,0),25)
        str5 = (
            pd.Series(np.where((str3 > 0) & (str3 > str2), str3, 0))
            .rolling(self.lookback_period)
            .sum()
        )

        # STR6 and STR7
        str6 = str4 * 100 / str1
        str7 = str5 * 100 / str1

        # STR8:=MA(ABS(STR7-STR6)/(STR7+STR6)*100,15)
        str8 = (abs(str7 - str6) / (str7 + str6) * 100).rolling(15).mean()

        # STR9:=(STR8+REF(STR8,15))/2
        str9 = (str8 + str8.shift(15)) / 2

        return pd.DataFrame({"str6": str6, "str7": str7, "str8": str8, "str9": str9})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on original script conditions"""
        strength = self.calculate_strength_indicators(data)
        zig = self.calculate_zig(data)

        zig_ma = zig.rolling(self.ma_period).mean()

        # A:=(STR7>STR6 AND STR7>25 AND STR6<25)
        condition_a = (
            (strength["str7"] > strength["str6"])
            & (strength["str7"] > 25)
            & (strength["str6"] < 25)
        )

        # D:=CROSS(B,G) - ZIG crosses above MA
        condition_d = (zig > zig_ma) & (zig.shift(1) <= zig_ma.shift(1))

        # W:=CROSS(G,B) - MA crosses above ZIG
        condition_w = (zig_ma > zig) & (zig_ma.shift(1) <= zig.shift(1))

        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[condition_d & condition_a] = 1  # Buy signal
        signals[condition_w] = -1  # Sell signal

        return signals
