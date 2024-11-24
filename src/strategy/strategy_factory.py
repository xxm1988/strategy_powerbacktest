from typing import Dict, Type
from .base_strategy import BaseStrategy
from .macd_strategy import MACDStrategy
from .moving_average_cross import MovingAverageCrossStrategy

class StrategyFactory:
    """
    Factory class for creating trading strategies
    
    Attributes:
        _strategies (Dict[str, Type[BaseStrategy]]): Registry of available strategies
    """
    _strategies = {
        'macd': MACDStrategy,
        'ma_cross': MovingAverageCrossStrategy
    }
    
    @classmethod
    def get_available_strategies(cls) -> Dict[str, Dict]:
        """
        Get list of available strategies with their descriptions and parameters
        
        Returns:
            Dict[str, Dict]: Dictionary of strategy information
        """
        return {
            'macd': {
                'name': 'MACD Strategy',
                'description': 'Moving Average Convergence Divergence strategy',
                'parameters': {
                    'fast_period': {'type': 'int', 'default': 12, 'min': 2, 'max': 50},
                    'slow_period': {'type': 'int', 'default': 26, 'min': 5, 'max': 100},
                    'signal_period': {'type': 'int', 'default': 9, 'min': 2, 'max': 50}
                }
            },
            'ma_cross': {
                'name': 'Moving Average Crossover',
                'description': 'Simple moving average crossover strategy',
                'parameters': {
                    'short_window': {'type': 'int', 'default': 20, 'min': 1, 'max': 100},
                    'long_window': {'type': 'int', 'default': 50, 'min': 10, 'max': 200}
                }
            }
        }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, parameters: Dict) -> BaseStrategy:
        """
        Create a strategy instance with given parameters
        
        Args:
            strategy_name (str): Name of the strategy to create
            parameters (Dict): Strategy parameters
            
        Returns:
            BaseStrategy: Initialized strategy instance
            
        Raises:
            ValueError: If strategy_name is not found in registry
        """
        if strategy_name not in cls._strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found. Available strategies: {list(cls._strategies.keys())}")
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class(parameters) 