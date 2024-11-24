from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime
from typing import Dict, Any
from .config import Config

def parse_date(date_str: str) -> datetime:
    """Convert date string to datetime object"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format")

def create_cli_parser() -> ArgumentParser:
    """Create command line argument parser with all available options"""
    parser = ArgumentParser(
        description='Strategy Power Backtester - A professional-grade algorithmic trading backtesting framework',
        formatter_class=RawDescriptionHelpFormatter
    )
    
    # Strategy selection
    parser.add_argument(
        '--strategy',
        type=str,
        help='Strategy to use for backtesting (e.g., macd, ma_cross)'
    )
    
    # Symbol selection
    parser.add_argument(
        '--symbol',
        type=str,
        help='Trading symbol (e.g., HK.00700)'
    )
    
    # Date range
    parser.add_argument(
        '--start-date',
        type=parse_date,
        help='Start date for backtesting (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=parse_date,
        help='End date for backtesting (YYYY-MM-DD)'
    )
    
    # Capital and commission
    parser.add_argument(
        '--initial-capital',
        type=float,
        help='Initial capital for backtesting'
    )
    
    parser.add_argument(
        '--commission',
        type=float,
        help='Commission rate (e.g., 0.001 for 0.1%%)'
    )
    
    return parser

def get_backtest_config(args: Any, config: Config) -> Dict[str, Any]:
    """
    Merge command line arguments with config file values
    
    Args:
        args: Parsed command line arguments
        config: Configuration object from config.yaml
        
    Returns:
        Dict containing final configuration values
    """
    return {
        'strategy_name': args.strategy or config.strategy.get('default', 'macd'),
        'strategy_params': config.strategy[args.strategy] if args.strategy else config.strategy['macd'],
        'symbol': args.symbol or config.backtest_config.get('default_symbol', 'HK.00700'),
        'start_date': args.start_date or datetime(2023, 1, 1),
        'end_date': args.end_date or datetime(2023, 12, 31),
        'initial_capital': args.initial_capital or config.backtest_config['initial_capital'],
        'commission': args.commission or config.backtest_config['commission']
    } 