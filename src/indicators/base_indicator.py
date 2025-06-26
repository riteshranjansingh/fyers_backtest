"""
Base Indicator Class
Foundation for all technical indicators with consistent interface and validation
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class IndicatorError(Exception):
    """Custom exception for indicator-related errors"""
    pass


class BaseIndicator(ABC):
    """
    Abstract base class for all technical indicators
    
    Provides consistent interface, validation, and common functionality
    for all indicators used in the backtesting system.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize base indicator
        
        Args:
            name: Human-readable name of the indicator
            **kwargs: Indicator-specific parameters
        """
        self.name = name
        self.parameters = kwargs
        self.data = None
        self.result = None
        self._is_calculated = False
        
        # Validation
        self._validate_parameters()
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate the indicator values
        
        Args:
            data: OHLCV DataFrame with DatetimeIndex
            **kwargs: Additional parameters for calculation
            
        Returns:
            Calculated indicator values as Series or DataFrame
        """
        pass
    
    def __call__(self, data: pd.DataFrame, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Make indicator callable: indicator(data)
        
        Args:
            data: OHLCV DataFrame
            **kwargs: Override parameters for this calculation
            
        Returns:
            Calculated indicator values
        """
        # Merge runtime parameters with instance parameters
        calc_params = {**self.parameters, **kwargs}
        
        # Validate data
        self._validate_data(data)
        
        # Calculate with merged parameters
        self.data = data
        self.result = self.calculate(data, **calc_params)
        self._is_calculated = True
        
        return self.result
    
    def _validate_data(self, data: pd.DataFrame):
        """Validate input data format"""
        if data is None or data.empty:
            raise IndicatorError(f"{self.name}: Data cannot be None or empty")
        
        if not isinstance(data, pd.DataFrame):
            raise IndicatorError(f"{self.name}: Data must be a pandas DataFrame")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise IndicatorError(f"{self.name}: Data must have DatetimeIndex")
        
        # Check for required OHLCV columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"{self.name}: Missing columns {missing_columns}. Some indicators may not work.")
    
    def _validate_parameters(self):
        """Validate indicator parameters (override in subclasses)"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get indicator information and current state"""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'calculated': self._is_calculated,
            'data_shape': self.data.shape if self.data is not None else None,
            'result_shape': self.result.shape if self.result is not None else None,
            'result_type': type(self.result).__name__ if self.result is not None else None
        }
    
    def get_signals(self, **signal_params) -> pd.Series:
        """
        Generate trading signals based on indicator values
        Override in subclasses for indicator-specific signals
        
        Returns:
            Series with signal values (1=buy, -1=sell, 0=hold)
        """
        if not self._is_calculated:
            raise IndicatorError(f"{self.name}: Calculate indicator first before generating signals")
        
        # Default: no signals (override in subclasses)
        return pd.Series(0, index=self.data.index, name=f"{self.name}_signal")
    
    def plot_data(self) -> Dict[str, Any]:
        """
        Return data formatted for plotting
        Override in subclasses for custom plotting data
        """
        if not self._is_calculated:
            raise IndicatorError(f"{self.name}: Calculate indicator first before plotting")
        
        plot_data = {
            'name': self.name,
            'data': self.result,
            'parameters': self.parameters
        }
        
        return plot_data
    
    @staticmethod
    def validate_period(period: int, min_period: int = 1, max_period: int = 1000) -> int:
        """Validate period parameter"""
        if not isinstance(period, int):
            raise IndicatorError(f"Period must be an integer, got {type(period)}")
        
        if period < min_period:
            raise IndicatorError(f"Period must be >= {min_period}, got {period}")
        
        if period > max_period:
            raise IndicatorError(f"Period must be <= {max_period}, got {period}")
        
        return period
    
    @staticmethod
    def validate_source_column(data: pd.DataFrame, column: str = 'close') -> str:
        """Validate that source column exists in data"""
        if column not in data.columns:
            available = ', '.join(data.columns)
            raise IndicatorError(f"Column '{column}' not found. Available: {available}")
        
        return column
    
    @staticmethod
    def ensure_sufficient_data(data: pd.DataFrame, required_periods: int, indicator_name: str = "Indicator"):
        """Ensure data has sufficient periods for calculation"""
        if len(data) < required_periods:
            raise IndicatorError(
                f"{indicator_name}: Insufficient data. Need {required_periods} periods, "
                f"got {len(data)}"
            )
    
    def __str__(self) -> str:
        """String representation of indicator"""
        params_str = ', '.join([f"{k}={v}" for k, v in self.parameters.items()])
        return f"{self.name}({params_str})"
    
    def __repr__(self) -> str:
        """Detailed representation of indicator"""
        return f"<{self.__class__.__name__}: {self.__str__()}>"


class MovingAverageBase(BaseIndicator):
    """
    Base class for all moving average indicators
    Provides common functionality for SMA, EMA, WMA, etc.
    """
    
    def __init__(self, name: str, period: int, source: str = 'close', **kwargs):
        """
        Initialize moving average base
        
        Args:
            name: Indicator name
            period: Number of periods for calculation
            source: Column to use for calculation (default: 'close')
            **kwargs: Additional parameters
        """
        super().__init__(name, period=period, source=source, **kwargs)
        
        # Store for easy access
        self.period = period
        self.source = source
    
    def _validate_parameters(self):
        """Validate moving average parameters"""
        self.period = self.validate_period(self.parameters['period'])
        
        # Validate source column will be done when data is provided
        if 'source' not in self.parameters:
            self.parameters['source'] = 'close'
    
    def get_signals(self, price_column: str = None, **signal_params) -> pd.Series:
        """
        Generate signals based on price vs moving average
        
        Args:
            price_column: Column to compare with MA (default: same as source)
            **signal_params: Additional signal parameters
            
        Returns:
            Series with signals: 1 (price > MA), -1 (price < MA), 0 (equal/NaN)
        """
        if not self._is_calculated:
            raise IndicatorError(f"{self.name}: Calculate indicator first")
        
        price_col = price_column or self.source
        
        if price_col not in self.data.columns:
            raise IndicatorError(f"Price column '{price_col}' not found in data")
        
        price = self.data[price_col]
        ma_values = self.result
        
        # Generate signals
        signals = pd.Series(0, index=self.data.index, name=f"{self.name}_signal")
        
        # Price above MA = bullish (1)
        signals[price > ma_values] = 1
        
        # Price below MA = bearish (-1)  
        signals[price < ma_values] = -1
        
        return signals


class CrossoverMixin:
    """
    Mixin class for indicators that support crossover analysis
    """
    
    def crossover(self, other_indicator, **kwargs) -> pd.Series:
        """
        Detect crossovers between this indicator and another
        
        Args:
            other_indicator: Another indicator or Series to compare with
            **kwargs: Additional parameters
            
        Returns:
            Series with crossover signals (1=upward cross, -1=downward cross, 0=no cross)
        """
        if not self._is_calculated:
            raise IndicatorError(f"{self.name}: Calculate indicator first")
        
        # Get values to compare
        if isinstance(other_indicator, BaseIndicator):
            if not other_indicator._is_calculated:
                raise IndicatorError(f"Other indicator must be calculated first")
            other_values = other_indicator.result
        elif isinstance(other_indicator, (pd.Series, pd.DataFrame)):
            other_values = other_indicator
        else:
            raise IndicatorError("Other indicator must be BaseIndicator, Series, or DataFrame")
        
        # Ensure same index
        common_index = self.result.index.intersection(other_values.index)
        
        if len(common_index) < 2:
            raise IndicatorError("Insufficient overlapping data for crossover analysis")
        
        # Calculate crossovers
        val1 = self.result.reindex(common_index)
        val2 = other_values.reindex(common_index)
        
        # Previous values
        val1_prev = val1.shift(1)
        val2_prev = val2.shift(1)
        
        crossovers = pd.Series(0, index=common_index, name='crossover')
        
        # Upward crossover: val1 was below val2, now above
        upward_cross = (val1_prev <= val2_prev) & (val1 > val2)
        crossovers[upward_cross] = 1
        
        # Downward crossover: val1 was above val2, now below
        downward_cross = (val1_prev >= val2_prev) & (val1 < val2)
        crossovers[downward_cross] = -1
        
        return crossovers