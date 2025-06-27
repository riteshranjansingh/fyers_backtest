"""
Backtesting Package - Complete backtesting system for strategy evaluation
"""

from .engine import BacktestEngine, BacktestConfig, BacktestResults, quick_backtest
from .trade_executor import TradeExecutor, ExecutionConfig, create_realistic_executor
from .cost_model import TransactionCostModel, CostConfig, create_discount_broker_config
from .trade_logger import TradeLogger, create_backtest_logger

# Cost configuration functions (import on demand to avoid circular imports)
def get_cost_profiles():
    """Get available cost profiles"""
    from ..config.cost_profiles import get_cost_profiles_for_ui
    return get_cost_profiles_for_ui()

def create_cost_model_with_profile(profile_name: str = None):
    """Create cost model with specified profile"""
    from ..config.cost_profiles import create_cost_model_with_profile as _create
    return _create(profile_name)

__all__ = [
    'BacktestEngine',
    'BacktestConfig', 
    'BacktestResults',
    'quick_backtest',
    'TradeExecutor',
    'ExecutionConfig',
    'create_realistic_executor',
    'TransactionCostModel',
    'CostConfig',
    'create_discount_broker_config',
    'TradeLogger',
    'create_backtest_logger',
    'get_cost_profiles',
    'create_cost_model_with_profile'
]