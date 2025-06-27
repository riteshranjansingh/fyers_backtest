"""
Risk Management Module
Advanced position sizing, stop losses, and trailing stop management
"""

# Version info
__version__ = "1.0.0"
__author__ = "Fyers Backtesting System"

# Import main classes
from .position_sizer import PositionSizer, RiskConfig
from .stop_loss import StopLossManager, StopLossConfig  
from .advanced_trailing import AdvancedTrailingStop, TrailingConfig
from .gap_handler import GapHandler, GapConfig
from .risk_integration import RiskManager

# Export main classes
__all__ = [
    'PositionSizer',
    'RiskConfig', 
    'StopLossManager',
    'StopLossConfig',
    'AdvancedTrailingStop', 
    'TrailingConfig',
    'GapHandler',
    'GapConfig',
    'RiskManager'
]

# Risk management constants
DEFAULT_RISK_PERCENTAGE = 2.0  # 2% default risk per trade
DEFAULT_MAX_POSITION_SIZE = 0.1  # Max 10% of account per position
DEFAULT_STOP_LOSS_PERCENTAGE = 1.5  # 1.5% stop loss
DEFAULT_TRAILING_TRIGGER = 1.0  # Start trailing at 1:1 R:R

# Common risk profiles
RISK_PROFILES = {
    'conservative': {
        'risk_percentage': 1.0,
        'max_position_size': 0.05,
        'stop_loss_percentage': 1.0,
        'trailing_trigger': 1.5
    },
    'moderate': {
        'risk_percentage': 2.0,
        'max_position_size': 0.1,
        'stop_loss_percentage': 1.5,
        'trailing_trigger': 1.0
    },
    'aggressive': {
        'risk_percentage': 3.0,
        'max_position_size': 0.15,
        'stop_loss_percentage': 2.0,
        'trailing_trigger': 0.8
    }
}

def get_risk_profile(profile_name: str) -> dict:
    """Get predefined risk profile"""
    return RISK_PROFILES.get(profile_name, RISK_PROFILES['moderate'])