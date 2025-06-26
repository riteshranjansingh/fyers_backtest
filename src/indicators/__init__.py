"""
Technical Indicators Module
Foundation for all technical analysis indicators used in strategies
"""

# Version info
__version__ = "1.0.0"
__author__ = "Fyers Backtesting System"

# Import base classes and common indicators
from .base_indicator import BaseIndicator, IndicatorError
from .moving_averages import EMA, SMA, WMA

# Export main classes for easy import
__all__ = [
    'BaseIndicator',
    'IndicatorError', 
    'EMA',
    'SMA', 
    'WMA'
]

# Indicator registry for dynamic loading
INDICATOR_REGISTRY = {
    'ema': EMA,
    'sma': SMA,
    'wma': WMA
}

def get_indicator(name: str):
    """
    Get an indicator class by name
    
    Args:
        name: Indicator name (case insensitive)
        
    Returns:
        Indicator class
        
    Example:
        ema_class = get_indicator('ema')
        ema = ema_class(data, period=21)
    """
    name_lower = name.lower()
    if name_lower in INDICATOR_REGISTRY:
        return INDICATOR_REGISTRY[name_lower]
    
    available = ', '.join(INDICATOR_REGISTRY.keys())
    raise IndicatorError(f"Indicator '{name}' not found. Available: {available}")

def list_indicators():
    """List all available indicators"""
    return list(INDICATOR_REGISTRY.keys())