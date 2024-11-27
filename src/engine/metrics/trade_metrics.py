from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

class TradeMetrics:
    """
    Handles all trade-related metric calculations for the backtest report.
    """
    
    @staticmethod
    def calculate_profit_factor(trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        profits = sum(t['pnl'] for t in trades if t.get('pnl', 0) > 0)
        losses = abs(sum(t['pnl'] for t in trades if t.get('pnl', 0) < 0))
        return profits / losses if losses != 0 else float('inf')
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trades"""
        if not trades:
            return 0.0
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        return winning_trades / len(trades)
    
    @staticmethod
    def calculate_trade_stats(trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate comprehensive trade statistics"""
        if not trades:
            return {
                'avg_trade_return': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        pnls = [t.get('pnl', 0) for t in trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        return {
            'avg_trade_return': np.mean(pnls) if pnls else 0.0,
            'avg_win': np.mean(winning_trades) if winning_trades else 0.0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0.0,
            'largest_win': max(pnls) if pnls else 0.0,
            'largest_loss': min(pnls) if pnls else 0.0
        }
    
    @staticmethod
    def calculate_consecutive_stats(trades: List[Dict[str, Any]], stat_type: str) -> int:
        """Calculate maximum consecutive wins or losses"""
        if not trades:
            return 0
            
        current_streak = max_streak = 0
        for trade in trades:
            pnl = trade.get('pnl', 0)
            if (stat_type == 'wins' and pnl > 0) or (stat_type == 'losses' and pnl < 0):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak
    
    @staticmethod
    def calculate_position_duration(trades: List[Dict[str, Any]]) -> float:
        """Calculate average position duration in days"""
        if not trades:
            return 0.0
        
        durations = []
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades):
                entry = pd.Timestamp(trades[i]['timestamp'])
                exit = pd.Timestamp(trades[i + 1]['timestamp'])
                duration = (exit - entry).total_seconds() / (24 * 3600)  # Convert to days
                durations.append(duration)
        return float(np.mean(durations) if durations else 0.0)
 