from datetime import datetime
from src.data.data_fetcher import FutuDataFetcher
from src.strategy.base_strategy import MovingAverageCrossStrategy
from src.engine.backtest_engine import BacktestEngine
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    # Load configuration
    config = Config('config.yaml')
    
    # Initialize data fetcher
    fetcher = FutuDataFetcher(
        host=config.futu_config['host'],
        port=config.futu_config['port']
    )
    
    # Fetch historical data
    data = fetcher.fetch_historical_data(
        symbol='HK.00700',
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31)
    )
    
    # Initialize strategy
    strategy = MovingAverageCrossStrategy({
        'short_window': config.strategy['moving_average_cross']['short_window'],
        'long_window': config.strategy['moving_average_cross']['long_window']
    })
    
    # Run backtest
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=config.backtest_config['initial_capital'],
        commission=config.backtest_config['commission']
    )
    
    results = engine.run(data)
    logger.info(f"Backtest Results: {results}")

if __name__ == '__main__':
    main()