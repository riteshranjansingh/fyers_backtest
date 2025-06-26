"""
Moving Averages Implementation
Core moving average indicators: SMA, EMA, WMA with optimized calculations
"""
import pandas as pd
import numpy as np
from typing import Union, Optional
import logging

from .base_indicator import MovingAverageBase, IndicatorError, CrossoverMixin

logger = logging.getLogger(__name__)


class SMA(MovingAverageBase, CrossoverMixin):
    """
    Simple Moving Average (SMA)
    
    The arithmetic mean of prices over N periods.
    Formula: SMA = (P1 + P2 + ... + Pn) / n
    
    Usage:
        sma20 = SMA(period=20)
        values = sma20(data)
        signals = sma20.get_signals()
    """
    
    def __init__(self, period: int, source: str = 'close', **kwargs):
        """
        Initialize Simple Moving Average
        
        Args:
            period: Number of periods for calculation
            source: Column to use ('close', 'open', 'high', 'low')
            **kwargs: Additional parameters
        """
        super().__init__(name=f"SMA({period})", period=period, source=source, **kwargs)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            data: OHLCV DataFrame
            **kwargs: Override parameters (period, source)
            
        Returns:
            Series with SMA values
        """
        # Get parameters (allow runtime override)
        period = kwargs.get('period', self.period)
        source = kwargs.get('source', self.source)
        
        # Validate
        period = self.validate_period(period)
        source = self.validate_source_column(data, source)
        self.ensure_sufficient_data(data, period, f"SMA({period})")
        
        # Calculate SMA using pandas rolling window
        sma_values = data[source].rolling(window=period, min_periods=period).mean()
        
        # Set name for the series
        sma_values.name = f"SMA_{period}"
        
        logger.debug(f"SMA({period}) calculated: {(~sma_values.isna()).sum()} valid values")
        
        return sma_values


class EMA(MovingAverageBase, CrossoverMixin):
    """
    Exponential Moving Average (EMA)
    
    Gives more weight to recent prices. More responsive than SMA.
    Formula: EMA = (Price * α) + (Previous_EMA * (1-α))
    Where α = 2 / (period + 1)
    
    Usage:
        ema21 = EMA(period=21)
        values = ema21(data)
        signals = ema21.get_signals()
        crossovers = ema9.crossover(ema21)
    """
    
    def __init__(self, period: int, source: str = 'close', **kwargs):
        """
        Initialize Exponential Moving Average
        
        Args:
            period: Number of periods for calculation
            source: Column to use ('close', 'open', 'high', 'low')
            **kwargs: Additional parameters
        """
        super().__init__(name=f"EMA({period})", period=period, source=source, **kwargs)
        
        # Calculate smoothing factor (alpha)
        self.alpha = 2.0 / (period + 1)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: OHLCV DataFrame
            **kwargs: Override parameters (period, source)
            
        Returns:
            Series with EMA values
        """
        # Get parameters
        period = kwargs.get('period', self.period)
        source = kwargs.get('source', self.source)
        
        # Validate
        period = self.validate_period(period)
        source = self.validate_source_column(data, source)
        self.ensure_sufficient_data(data, period, f"EMA({period})")
        
        # Calculate alpha for this period
        alpha = kwargs.get('alpha', 2.0 / (period + 1))
        
        # Method 1: Using pandas ewm (fastest)
        ema_values = data[source].ewm(alpha=alpha, adjust=False).mean()
        
        # Set first 'period-1' values to NaN (industry standard)
        ema_values.iloc[:period-1] = np.nan
        
        # Set name
        ema_values.name = f"EMA_{period}"
        
        logger.debug(f"EMA({period}) calculated with α={alpha:.4f}: {(~ema_values.isna()).sum()} valid values")
        
        return ema_values
    
    def calculate_manual(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Manual EMA calculation (for educational purposes or verification)
        Slower but more transparent than pandas ewm
        """
        period = kwargs.get('period', self.period)
        source = kwargs.get('source', self.source)
        alpha = 2.0 / (period + 1)
        
        prices = data[source].values
        ema_values = np.full(len(prices), np.nan)
        
        # First EMA value is SMA of first 'period' values
        if len(prices) >= period:
            ema_values[period-1] = np.mean(prices[:period])
            
            # Calculate subsequent EMA values
            for i in range(period, len(prices)):
                ema_values[i] = (prices[i] * alpha) + (ema_values[i-1] * (1 - alpha))
        
        return pd.Series(ema_values, index=data.index, name=f"EMA_{period}_manual")


class WMA(MovingAverageBase, CrossoverMixin):
    """
    Weighted Moving Average (WMA)
    
    Applies linearly decreasing weights to older prices.
    Most recent price gets weight=n, previous gets n-1, etc.
    
    Formula: WMA = (P1*n + P2*(n-1) + ... + Pn*1) / (n + (n-1) + ... + 1)
    
    Usage:
        wma10 = WMA(period=10)
        values = wma10(data)
    """
    
    def __init__(self, period: int, source: str = 'close', **kwargs):
        """
        Initialize Weighted Moving Average
        
        Args:
            period: Number of periods for calculation
            source: Column to use ('close', 'open', 'high', 'low')
            **kwargs: Additional parameters
        """
        super().__init__(name=f"WMA({period})", period=period, source=source, **kwargs)
        
        # Pre-calculate weights for efficiency
        self.weights = np.arange(1, period + 1)
        self.weight_sum = np.sum(self.weights)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate Weighted Moving Average
        
        Args:
            data: OHLCV DataFrame
            **kwargs: Override parameters (period, source)
            
        Returns:
            Series with WMA values
        """
        # Get parameters
        period = kwargs.get('period', self.period)
        source = kwargs.get('source', self.source)
        
        # Validate
        period = self.validate_period(period)
        source = self.validate_source_column(data, source)
        self.ensure_sufficient_data(data, period, f"WMA({period})")
        
        # Get weights for this calculation
        weights = np.arange(1, period + 1)
        weight_sum = np.sum(weights)
        
        # Calculate WMA using rolling apply
        def wma_calc(x):
            return np.dot(x, weights) / weight_sum
        
        wma_values = data[source].rolling(window=period, min_periods=period).apply(wma_calc, raw=True)
        
        # Set name
        wma_values.name = f"WMA_{period}"
        
        logger.debug(f"WMA({period}) calculated: {(~wma_values.isna()).sum()} valid values")
        
        return wma_values


class VWMA(MovingAverageBase, CrossoverMixin):
    """
    Volume Weighted Moving Average (VWMA)
    
    Weights prices by their volume. Higher volume periods get more weight.
    
    Formula: VWMA = Σ(Price * Volume) / Σ(Volume)
    
    Usage:
        vwma20 = VWMA(period=20)
        values = vwma20(data)
    """
    
    def __init__(self, period: int, source: str = 'close', **kwargs):
        """
        Initialize Volume Weighted Moving Average
        
        Args:
            period: Number of periods for calculation
            source: Column to use ('close', 'open', 'high', 'low')
            **kwargs: Additional parameters
        """
        super().__init__(name=f"VWMA({period})", period=period, source=source, **kwargs)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate Volume Weighted Moving Average
        
        Args:
            data: OHLCV DataFrame
            **kwargs: Override parameters (period, source)
            
        Returns:
            Series with VWMA values
        """
        # Get parameters
        period = kwargs.get('period', self.period)
        source = kwargs.get('source', self.source)
        
        # Validate
        period = self.validate_period(period)
        source = self.validate_source_column(data, source)
        self.ensure_sufficient_data(data, period, f"VWMA({period})")
        
        # Ensure volume column exists
        if 'volume' not in data.columns:
            raise IndicatorError("VWMA requires 'volume' column in data")
        
        # Calculate price * volume
        pv = data[source] * data['volume']
        
        # Rolling sums
        pv_sum = pv.rolling(window=period, min_periods=period).sum()
        volume_sum = data['volume'].rolling(window=period, min_periods=period).sum()
        
        # VWMA = sum(price * volume) / sum(volume)
        vwma_values = pv_sum / volume_sum
        
        # Handle division by zero (when volume sum is 0)
        vwma_values = vwma_values.replace([np.inf, -np.inf], np.nan)
        
        # Set name
        vwma_values.name = f"VWMA_{period}"
        
        logger.debug(f"VWMA({period}) calculated: {(~vwma_values.isna()).sum()} valid values")
        
        return vwma_values


# Utility Functions for Moving Averages

def compare_moving_averages(data: pd.DataFrame, periods: list, ma_type: str = 'EMA', source: str = 'close') -> pd.DataFrame:
    """
    Calculate multiple moving averages for comparison
    
    Args:
        data: OHLCV DataFrame
        periods: List of periods to calculate [9, 21, 50, 200]
        ma_type: Type of MA ('SMA', 'EMA', 'WMA', 'VWMA')
        source: Source column
        
    Returns:
        DataFrame with all MAs
        
    Example:
        mas = compare_moving_averages(data, [9, 21, 50], 'EMA')
        print(mas.head())
    """
    # Get MA class
    ma_classes = {'SMA': SMA, 'EMA': EMA, 'WMA': WMA, 'VWMA': VWMA}
    
    if ma_type not in ma_classes:
        raise IndicatorError(f"Unknown MA type: {ma_type}. Available: {list(ma_classes.keys())}")
    
    ma_class = ma_classes[ma_type]
    
    # Calculate all MAs
    result = pd.DataFrame(index=data.index)
    
    for period in periods:
        ma = ma_class(period=period, source=source)
        ma_values = ma(data)
        result[f"{ma_type}_{period}"] = ma_values
    
    return result


def get_ma_signals(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.DataFrame:
    """
    Get crossover signals between two moving averages
    
    Args:
        fast_ma: Faster moving average (smaller period)
        slow_ma: Slower moving average (larger period)
        
    Returns:
        DataFrame with signals and crossovers
    """
    signals = pd.DataFrame(index=fast_ma.index)
    
    # Position signals
    signals['position'] = 0
    signals.loc[fast_ma > slow_ma, 'position'] = 1  # Bullish
    signals.loc[fast_ma < slow_ma, 'position'] = -1  # Bearish
    
    # Crossover detection
    signals['crossover'] = 0
    
    # Bullish crossover: fast was below slow, now above
    bullish_cross = (fast_ma.shift(1) <= slow_ma.shift(1)) & (fast_ma > slow_ma)
    signals.loc[bullish_cross, 'crossover'] = 1
    
    # Bearish crossover: fast was above slow, now below  
    bearish_cross = (fast_ma.shift(1) >= slow_ma.shift(1)) & (fast_ma < slow_ma)
    signals.loc[bearish_cross, 'crossover'] = -1
    
    # Signal strength (distance between MAs as percentage)
    signals['signal_strength'] = abs(fast_ma - slow_ma) / slow_ma * 100
    
    return signals


# Quick factory functions
def ema(data: pd.DataFrame, period: int, source: str = 'close') -> pd.Series:
    """Quick EMA calculation"""
    return EMA(period=period, source=source)(data)

def sma(data: pd.DataFrame, period: int, source: str = 'close') -> pd.Series:
    """Quick SMA calculation"""
    return SMA(period=period, source=source)(data)

def wma(data: pd.DataFrame, period: int, source: str = 'close') -> pd.Series:
    """Quick WMA calculation"""
    return WMA(period=period, source=source)(data)

def vwma(data: pd.DataFrame, period: int, source: str = 'close') -> pd.Series:
    """Quick VWMA calculation"""
    return VWMA(period=period, source=source)(data)