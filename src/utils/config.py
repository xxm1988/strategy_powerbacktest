import yaml
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    @property
    def futu_config(self) -> Dict[str, Any]:
        return self.config.get('futu', {})
        
    @property
    def backtest_config(self) -> Dict[str, Any]:
        return self.config.get('backtest', {})
        
    @property
    def strategy(self) -> Dict[str, Any]:
        return self.config.get('strategy', {})
        
    @property
    def data_config(self) -> Dict[str, Any]:
        return self.config.get('data', {})
        
    @property
    def logging_config(self) -> Dict[str, Any]:
        return self.config.get('logging', {})
        
    @property
    def strategy(self) -> Dict[str, Any]:
        return self.config.get('strategy', {})
        
    @property
    def data_config(self) -> Dict[str, Any]:
        return self.config.get('data', {})
        
    @property
    def logging_config(self) -> Dict[str, Any]:
        return self.config.get('logging', {})
        
    def get_strategy_params(self, strategy_name):
        """
        Get strategy-specific parameters from config file
        """
        return self.strategy.get(strategy_name, {})