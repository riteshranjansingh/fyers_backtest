"""
Unified Moving Average Crossover Strategy System
Complete MA crossover implementation with flexible configuration and optimization
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
import logging
import json

from .base_strategy import BaseStrategy, CrossoverStrategy, StrategyResult, Signal, StrategyError
from ..indicators.moving_averages import EMA, SMA, WMA, VWMA, MovingAverageBase

logger = logging.getLogger(__name__)


class MAType(Enum):
    """Available Moving Average Types"""
    EMA = "EMA"
    SMA = "SMA" 
    WMA = "WMA"
    VWMA = "VWMA"


@dataclass
class MAConfig:
    """
    Complete configuration for MA crossover strategy
    This is what UI will create and pass to strategy
    """
    # Core MA settings
    fast_ma_type: MAType = MAType.EMA
    fast_period: int = 9
    slow_ma_type: MAType = MAType.EMA
    slow_period: int = 21
    
    # Signal filtering
    min_separation: float = 0.1
    signal_strength_threshold: float = 0.3
    enable_filtering: bool = True
    
    # Data settings
    price_column: str = 'close'
    
    # Advanced filtering
    volume_confirmation: bool = True
    anti_whipsaw: bool = True
    min_signal_distance: int = 5
    
    # Custom metadata
    name: str = field(default="")
    description: str = field(default="")
    
    def __post_init__(self):
        """Validate configuration and set defaults"""
        if self.fast_period >= self.slow_period:
            raise ValueError(f"Fast period ({self.fast_period}) must be less than slow period ({self.slow_period})")
        
        if not (0 <= self.min_separation <= 10):
            raise ValueError("min_separation must be between 0 and 10 percent")
        
        if not (0 <= self.signal_strength_threshold <= 1):
            raise ValueError("signal_strength_threshold must be between 0 and 1")
        
        # Auto-generate name if not provided
        if not self.name:
            self.name = f"{self.fast_ma_type.value}({self.fast_period})_{self.slow_ma_type.value}({self.slow_period})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'fast_ma_type': self.fast_ma_type.value,
            'fast_period': self.fast_period,
            'slow_ma_type': self.slow_ma_type.value,
            'slow_period': self.slow_period,
            'min_separation': self.min_separation,
            'signal_strength_threshold': self.signal_strength_threshold,
            'enable_filtering': self.enable_filtering,
            'price_column': self.price_column,
            'volume_confirmation': self.volume_confirmation,
            'anti_whipsaw': self.anti_whipsaw,
            'min_signal_distance': self.min_signal_distance,
            'name': self.name,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MAConfig':
        """Create from dictionary"""
        config_dict = config_dict.copy()
        config_dict['fast_ma_type'] = MAType(config_dict['fast_ma_type'])
        config_dict['slow_ma_type'] = MAType(config_dict['slow_ma_type'])
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'MAConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class MACrossoverStrategy(CrossoverStrategy):
    """
    Unified Moving Average Crossover Strategy
    
    Supports any combination of MA types with comprehensive filtering and optimization.
    This is the main strategy class that everything else builds on.
    
    Features:
    - Any MA combination (EMA/SMA/WMA/VWMA)
    - Configurable filtering and validation
    - Rich signal metadata
    - UI-ready interface
    - Optimization support
    
    Usage:
        config = MAConfig(fast_ma_type=MAType.EMA, fast_period=9, 
                         slow_ma_type=MAType.SMA, slow_period=21)
        strategy = MACrossoverStrategy(config)
        result = strategy.generate_signals(data)
    """
    
    # Available MA classes mapping
    MA_CLASSES = {
        MAType.EMA: EMA,
        MAType.SMA: SMA,
        MAType.WMA: WMA,
        MAType.VWMA: VWMA
    }
    
    def __init__(self, config: MAConfig):
        """
        Initialize strategy with configuration
        
        Args:
            config: MAConfig object with all parameters
        """
        self.config = config
        
        # Get parameters without 'name' to avoid duplicate
        parameters = config.to_dict()
        parameters.pop('name', None)  # Remove name from parameters
        parameters.pop('description', None)  # Remove description too
        
        # Initialize base strategy
        super().__init__(name=config.name, **parameters)
        
        # Create MA indicators based on configuration
        self.fast_ma = self._create_ma_indicator(
            config.fast_ma_type, config.fast_period, config.price_column
        )
        self.slow_ma = self._create_ma_indicator(
            config.slow_ma_type, config.slow_period, config.price_column
        )
        
        # Cache for calculated values
        self.fast_ma_values = None
        self.slow_ma_values = None
        self.trend_direction = None
        self.last_result = None
        
        logger.info(f"Created MA strategy: {config.name}")
    
    def _create_ma_indicator(self, ma_type: MAType, period: int, source: str) -> MovingAverageBase:
        """Create MA indicator based on type"""
        if ma_type not in self.MA_CLASSES:
            raise StrategyError(f"Unsupported MA type: {ma_type}")
        
        ma_class = self.MA_CLASSES[ma_type]
        return ma_class(period=period, source=source)
    
    def get_minimum_periods(self) -> int:
        """Return minimum periods needed"""
        # For testing: use adaptive minimum based on slow MA + small buffer
        # For production: this can be higher for more stable signals
        return max(self.config.slow_period + 5, 30)  # Reduced from 50
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> StrategyResult:
        """
        Generate MA crossover signals
        
        Args:
            data: OHLCV DataFrame (from Phase 2 cached data)
            **kwargs: Runtime parameter overrides (from UI)
            
        Returns:
            StrategyResult with comprehensive signal information
        """
        # Validate input data
        self._validate_data(data)
        self.data = data
        
        # Apply runtime overrides if any
        runtime_config = self._apply_runtime_overrides(**kwargs)
        
        logger.info(f"{self.name}: Analyzing {len(data)} periods")
        
        # Calculate both MAs using our modular indicators
        self.fast_ma_values = self.fast_ma(data)
        self.slow_ma_values = self.slow_ma(data)
        
        # Detect crossover points
        try:
            bullish_crossovers, bearish_crossovers = self._detect_crossovers(
                self.fast_ma_values,
                self.slow_ma_values,
                min_separation=runtime_config.min_separation
            )
            
            # Validate crossover results
            if not isinstance(bullish_crossovers, pd.Series):
                bullish_crossovers = pd.Series(False, index=data.index)
            if not isinstance(bearish_crossovers, pd.Series):
                bearish_crossovers = pd.Series(False, index=data.index)
            
            # Debug logging
            logger.debug(f"Crossover detection: {bullish_crossovers.sum()} bullish, {bearish_crossovers.sum()} bearish")
            
        except Exception as e:
            logger.warning(f"Error in crossover detection: {e}")
            # Create empty crossover series as fallback
            bullish_crossovers = pd.Series(False, index=data.index, name='bullish_crossovers')
            bearish_crossovers = pd.Series(False, index=data.index, name='bearish_crossovers')
        
        # Calculate signal strength for each point
        try:
            signal_strength = self._calculate_signal_strength(
                self.fast_ma_values,
                self.slow_ma_values,
                normalize=True
            )
        except Exception as e:
            logger.warning(f"Error calculating signal strength: {e}")
            # Create default signal strength
            signal_strength = pd.Series(0.5, index=data.index, name='signal_strength')
        
        # Determine overall trend direction
        self.trend_direction = self._calculate_trend_direction()
        
        # Create initial signal series
        signal_series = pd.Series(0, index=data.index, name=f"{self.name}_signals")
        signal_series[bullish_crossovers] = 1   # Buy signals
        signal_series[bearish_crossovers] = -1  # Sell signals
        
        # Apply advanced filtering if enabled
        if runtime_config.enable_filtering:
            signal_series = self._apply_comprehensive_filtering(
                signal_series, signal_strength, runtime_config, data
            )
        
        # Create detailed Signal objects with rich metadata
        signals = self._create_detailed_signals(data, signal_series, signal_strength, runtime_config)
        
        # Generate comprehensive metadata
        metadata = self._create_comprehensive_metadata(
            runtime_config, bullish_crossovers, bearish_crossovers, signals, data
        )
        
        # Store result and mark as fitted
        self.last_result = StrategyResult(
            strategy_name=self.name,
            signals=signals,
            signal_series=signal_series,
            metadata=metadata
        )
        self._is_fitted = True
        
        logger.info(
            f"{self.name}: Generated {len(signals)} signals "
            f"({metadata['bullish_crossovers']} bullish, {metadata['bearish_crossovers']} bearish)"
        )
        
        return self.last_result
    
    def _apply_runtime_overrides(self, **kwargs) -> MAConfig:
        """Apply runtime parameter overrides (typically from UI)"""
        if not kwargs:
            return self.config
        
        config_dict = self.config.to_dict()
        
        # Apply overrides
        for key, value in kwargs.items():
            if key in config_dict:
                config_dict[key] = value
                logger.debug(f"Runtime override: {key} = {value}")
        
        return MAConfig.from_dict(config_dict)
    
    def _calculate_trend_direction(self) -> pd.Series:
        """Calculate trend direction based on MA relationship"""
        trend = pd.Series(0, index=self.data.index, name='trend')
        trend[self.fast_ma_values > self.slow_ma_values] = 1   # Bullish trend
        trend[self.fast_ma_values < self.slow_ma_values] = -1  # Bearish trend
        return trend
    
    def _apply_comprehensive_filtering(
        self,
        signal_series: pd.Series,
        signal_strength: pd.Series,
        config: MAConfig,
        data: pd.DataFrame
    ) -> pd.Series:
        """Apply all configured filters to signals"""
        filtered_series = signal_series.copy()
        original_count = (signal_series != 0).sum()
        
        # Filter 1: Signal strength threshold
        weak_signals = signal_strength < config.signal_strength_threshold
        filtered_series[weak_signals & (signal_series != 0)] = 0
        
        # Filter 2: Anti-whipsaw (remove signals too close together)
        if config.anti_whipsaw:
            min_distance = max(config.min_signal_distance, config.slow_period // 2)
            filtered_series = self._remove_close_signals(filtered_series, min_distance)
        
        # Filter 3: Volume confirmation
        if config.volume_confirmation and 'volume' in data.columns:
            filtered_series = self._apply_volume_filter(filtered_series, data)
        
        # Filter 4: Trend alignment (optional enhancement)
        filtered_series = self._apply_trend_alignment_filter(filtered_series)
        
        filtered_count = (filtered_series != 0).sum()
        logger.debug(f"Filtering: {original_count} → {filtered_count} signals ({original_count - filtered_count} removed)")
        
        return filtered_series
    
    def _remove_close_signals(self, signal_series: pd.Series, min_distance: int) -> pd.Series:
        """Remove signals that are too close to each other"""
        filtered_series = signal_series.copy()
        signal_indices = signal_series[signal_series != 0].index
        
        last_signal_idx = None
        for idx in signal_indices:
            if last_signal_idx is not None:
                # Calculate distance in periods
                current_pos = signal_series.index.get_loc(idx)
                last_pos = signal_series.index.get_loc(last_signal_idx)
                distance = current_pos - last_pos
                
                if distance < min_distance:
                    # Keep the stronger signal
                    current_strength = abs(signal_series.loc[idx])
                    last_strength = abs(signal_series.loc[last_signal_idx])
                    
                    if current_strength <= last_strength:
                        filtered_series.loc[idx] = 0  # Remove current
                        continue
                    else:
                        filtered_series.loc[last_signal_idx] = 0  # Remove previous
            
            last_signal_idx = idx
        
        return filtered_series
    
    def _apply_volume_filter(self, signal_series: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Filter signals based on volume confirmation"""
        filtered_series = signal_series.copy()
        
        # Calculate volume moving average
        volume_window = min(20, len(data) // 4)
        volume_ma = data['volume'].rolling(window=volume_window, min_periods=1).mean()
        
        # Only keep signals with above-average volume
        low_volume_mask = data['volume'] < (volume_ma * 0.8)  # 80% of average
        filtered_series[low_volume_mask & (signal_series != 0)] = 0
        
        return filtered_series
    
    def _apply_trend_alignment_filter(self, signal_series: pd.Series) -> pd.Series:
        """Optional: Filter signals that go against overall trend"""
        # This is a placeholder for future enhancement
        # Could add longer-term MA for trend filter
        return signal_series
    
    def _create_detailed_signals(
        self,
        data: pd.DataFrame,
        signal_series: pd.Series,
        signal_strength: pd.Series,
        config: MAConfig
    ) -> List[Signal]:
        """Create detailed Signal objects with comprehensive metadata"""
        signals = []
        
        for timestamp, signal_value in signal_series.items():
            if signal_value != 0:
                signal_type = 'BUY' if signal_value > 0 else 'SELL'
                price = data.loc[timestamp, config.price_column]
                confidence = signal_strength.loc[timestamp]
                
                # Get MA values at signal time
                fast_ma_val = self.fast_ma_values.loc[timestamp]
                slow_ma_val = self.slow_ma_values.loc[timestamp]
                separation_pct = abs(fast_ma_val - slow_ma_val) / slow_ma_val * 100
                
                # Rich metadata for analysis and UI display
                metadata = {
                    'strategy_name': self.name,
                    'config': config.to_dict(),
                    'price_column': config.price_column,
                    'fast_ma_type': config.fast_ma_type.value,
                    'fast_ma_period': config.fast_period,
                    'fast_ma_value': fast_ma_val,
                    'slow_ma_type': config.slow_ma_type.value,
                    'slow_ma_period': config.slow_period,
                    'slow_ma_value': slow_ma_val,
                    'ma_separation_pct': separation_pct,
                    'trend_direction': self.trend_direction.loc[timestamp],
                    'signal_strength': confidence,
                    'volume': data.loc[timestamp, 'volume'] if 'volume' in data.columns else None,
                    'high': data.loc[timestamp, 'high'],
                    'low': data.loc[timestamp, 'low'],
                    'open': data.loc[timestamp, 'open']
                }
                
                signal = Signal(
                    timestamp=timestamp,
                    signal_type=signal_type,
                    price=price,
                    confidence=confidence,
                    metadata=metadata
                )
                signals.append(signal)
        
        return signals
    
    def _create_comprehensive_metadata(
        self,
        config: MAConfig,
        bullish_crossovers: pd.Series,
        bearish_crossovers: pd.Series,
        signals: List[Signal],
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create comprehensive strategy metadata"""
        # Safe calculation of crossovers (handle empty series)
        try:
            total_crossovers = (bullish_crossovers | bearish_crossovers).sum()
            bullish_count = int(bullish_crossovers.sum())
            bearish_count = int(bearish_crossovers.sum())
        except Exception as e:
            logger.warning(f"Error calculating crossovers: {e}")
            total_crossovers = 0
            bullish_count = 0
            bearish_count = 0
        
        # Calculate performance metrics
        signal_frequency = len(signals) / len(data) * 100 if len(data) > 0 else 0
        avg_signal_strength = np.mean([s.confidence for s in signals]) if signals else 0
        
        # MA statistics (with error handling)
        try:
            ma_stats = self._calculate_ma_statistics()
        except Exception as e:
            logger.warning(f"Error calculating MA statistics: {e}")
            ma_stats = {}
        
        return {
            'strategy_config': config.to_dict(),
            'data_period': {
                'start_date': data.index[0].strftime('%Y-%m-%d'),
                'end_date': data.index[-1].strftime('%Y-%m-%d'),
                'total_periods': len(data)
            },
            'crossover_analysis': {
                'total_crossovers': int(total_crossovers),
                'bullish_crossovers': bullish_count,
                'bearish_crossovers': bearish_count,
                'crossover_frequency_pct': total_crossovers / len(data) * 100 if len(data) > 0 else 0
            },
            'signal_analysis': {
                'total_signals': len(signals),
                'buy_signals': sum(1 for s in signals if s.signal_type == 'BUY'),
                'sell_signals': sum(1 for s in signals if s.signal_type == 'SELL'),
                'signal_frequency_pct': signal_frequency,
                'avg_signal_strength': avg_signal_strength,
                'filtering_enabled': config.enable_filtering
            },
            'ma_statistics': ma_stats,
            'performance_metrics': self._calculate_basic_performance_metrics(signals, data)
        }
    
    def _calculate_ma_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about the MA values"""
        if self.fast_ma_values is None or self.slow_ma_values is None:
            return {'status': 'no_ma_values'}
        
        try:
            # Remove NaN values for statistics
            fast_clean = self.fast_ma_values.dropna()
            slow_clean = self.slow_ma_values.dropna()
            
            if len(fast_clean) == 0 or len(slow_clean) == 0:
                return {'status': 'insufficient_clean_data'}
            
            # Calculate separation safely
            common_index = fast_clean.index.intersection(slow_clean.index)
            if len(common_index) == 0:
                return {'status': 'no_common_index'}
            
            fast_aligned = fast_clean.reindex(common_index)
            slow_aligned = slow_clean.reindex(common_index)
            
            separation = abs(fast_aligned - slow_aligned) / slow_aligned * 100
            separation = separation.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(separation) == 0:
                return {'status': 'no_valid_separation'}
            
            return {
                'fast_ma_stats': {
                    'mean': float(fast_clean.mean()),
                    'std': float(fast_clean.std()),
                    'min': float(fast_clean.min()),
                    'max': float(fast_clean.max())
                },
                'slow_ma_stats': {
                    'mean': float(slow_clean.mean()),
                    'std': float(slow_clean.std()),
                    'min': float(slow_clean.min()),
                    'max': float(slow_clean.max())
                },
                'separation_stats': {
                    'mean_separation_pct': float(separation.mean()),
                    'max_separation_pct': float(separation.max()),
                    'min_separation_pct': float(separation.min())
                },
                'status': 'success'
            }
        except Exception as e:
            logger.warning(f"Error in MA statistics calculation: {e}")
            return {'status': 'calculation_error', 'error': str(e)}
    
    def _calculate_basic_performance_metrics(self, signals: List[Signal], data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic performance metrics for the strategy"""
        try:
            if not signals:
                return {'status': 'no_signals'}
            
            # Simple metrics (more advanced ones will come in backtesting phase)
            signal_timestamps = [s.timestamp for s in signals]
            signal_prices = [s.price for s in signals]
            
            return {
                'first_signal_date': signal_timestamps[0].strftime('%Y-%m-%d') if signal_timestamps else None,
                'last_signal_date': signal_timestamps[-1].strftime('%Y-%m-%d') if signal_timestamps else None,
                'price_range_coverage': {
                    'signal_price_min': min(signal_prices) if signal_prices else None,
                    'signal_price_max': max(signal_prices) if signal_prices else None,
                    'data_price_min': float(data[self.config.price_column].min()),
                    'data_price_max': float(data[self.config.price_column].max())
                },
                'status': 'success'
            }
        except Exception as e:
            logger.warning(f"Error calculating performance metrics: {e}")
            return {'status': 'calculation_error', 'error': str(e)}
    
    def get_current_position(self, data: pd.DataFrame = None) -> Dict[str, Any]:
        """Get current market position based on latest MA values"""
        if data is None:
            data = self.data
        
        if (self.fast_ma_values is None or self.slow_ma_values is None or 
            data is None or len(data) == 0):
            return {'position': 'UNKNOWN', 'reason': 'Insufficient data'}
        
        # Get latest values
        latest_fast = self.fast_ma_values.iloc[-1]
        latest_slow = self.slow_ma_values.iloc[-1]
        latest_price = data[self.config.price_column].iloc[-1]
        
        if pd.isna(latest_fast) or pd.isna(latest_slow):
            return {'position': 'UNKNOWN', 'reason': 'MA values not available'}
        
        # Determine position
        if latest_fast > latest_slow:
            position = 'BULLISH'
            separation = (latest_fast - latest_slow) / latest_slow * 100
        else:
            position = 'BEARISH'
            separation = (latest_slow - latest_fast) / latest_slow * 100
        
        return {
            'position': position,
            'current_price': float(latest_price),
            'fast_ma': {
                'type': self.config.fast_ma_type.value,
                'period': self.config.fast_period,
                'value': float(latest_fast)
            },
            'slow_ma': {
                'type': self.config.slow_ma_type.value,
                'period': self.config.slow_period,
                'value': float(latest_slow)
            },
            'separation_pct': float(separation),
            'strength': float(min(separation / 2.0, 1.0)),  # Normalized 0-1
            'timestamp': data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def plot_data(self) -> Dict[str, Any]:
        """Return data formatted for plotting"""
        if not self._is_fitted:
            raise StrategyError("Strategy must be fitted before plotting")
        
        return {
            'strategy_name': self.name,
            'config': self.config.to_dict(),
            'price_data': self.data[self.config.price_column],
            'fast_ma': self.fast_ma_values,
            'slow_ma': self.slow_ma_values,
            'signals': self.last_result.signal_series if self.last_result else None,
            'trend_direction': self.trend_direction,
            'timestamps': self.data.index
        }


# === UI-READY FACTORY FUNCTIONS ===

def create_ma_strategy(
    fast_ma_type: str,
    fast_period: int,
    slow_ma_type: str,
    slow_period: int,
    **additional_params
) -> MACrossoverStrategy:
    """
    Create MA strategy from simple parameters (UI-friendly)
    
    Args:
        fast_ma_type: "EMA", "SMA", "WMA", "VWMA"
        fast_period: Fast MA period
        slow_ma_type: "EMA", "SMA", "WMA", "VWMA"
        slow_period: Slow MA period
        **additional_params: Other configuration options
        
    Returns:
        Configured MACrossoverStrategy
        
    Example:
        strategy = create_ma_strategy("EMA", 9, "SMA", 21, min_separation=0.2)
    """
    config = MAConfig(
        fast_ma_type=MAType(fast_ma_type),
        fast_period=fast_period,
        slow_ma_type=MAType(slow_ma_type),
        slow_period=slow_period,
        **additional_params
    )
    
    return MACrossoverStrategy(config)


def get_available_ma_types() -> List[str]:
    """Get list of available MA types for UI dropdown"""
    return [ma_type.value for ma_type in MAType]


def get_popular_configurations() -> List[Dict[str, Any]]:
    """Get popular MA configurations for UI presets"""
    return [
        {
            'name': 'EMA 9/21 (Short Term)',
            'config': {
                'fast_ma_type': 'EMA', 'fast_period': 9,
                'slow_ma_type': 'EMA', 'slow_period': 21,
                'description': 'Popular short-term trend following'
            }
        },
        {
            'name': 'SMA 10/20 (Classic)',
            'config': {
                'fast_ma_type': 'SMA', 'fast_period': 10,
                'slow_ma_type': 'SMA', 'slow_period': 20,
                'description': 'Classic swing trading setup'
            }
        },
        {
            'name': 'EMA/SMA 12/26 (Mixed)',
            'config': {
                'fast_ma_type': 'EMA', 'fast_period': 12,
                'slow_ma_type': 'SMA', 'slow_period': 26,
                'description': 'Mixed MA types for unique signals'
            }
        },
        {
            'name': 'EMA 20/50 (Medium Term)',
            'config': {
                'fast_ma_type': 'EMA', 'fast_period': 20,
                'slow_ma_type': 'EMA', 'slow_period': 50,
                'description': 'Medium-term trend identification'
            }
        },
        {
            'name': 'EMA 50/200 (Golden Cross)',
            'config': {
                'fast_ma_type': 'EMA', 'fast_period': 50,
                'slow_ma_type': 'EMA', 'slow_period': 200,
                'description': 'Long-term trend analysis'
            }
        }
    ]


# === PARAMETER OPTIMIZATION SYSTEM ===

class MAParameterOptimizer:
    """
    Comprehensive parameter optimization for MA crossover strategies
    """
    
    # Standard periods used in trading
    STANDARD_PERIODS = [5, 8, 9, 10, 12, 15, 20, 21, 26, 30, 50, 100, 200]
    
    def __init__(self):
        self.optimization_results = None
    
    def get_valid_combinations(
        self, 
        min_separation: int = 2,
        max_slow_period: int = 200
    ) -> List[Tuple[int, int]]:
        """Get all valid (fast, slow) period combinations"""
        combinations = []
        
        for fast in self.STANDARD_PERIODS:
            for slow in self.STANDARD_PERIODS:
                if (slow > fast and 
                    (slow - fast) >= min_separation and 
                    slow <= max_slow_period):
                    combinations.append((fast, slow))
        
        return combinations
    
    def get_combinations_by_style(self) -> Dict[str, List[Tuple[int, int]]]:
        """Categorize combinations by trading style"""
        all_combinations = self.get_valid_combinations()
        
        styles = {
            'scalping': [],        # Very short term
            'short_term': [],      # Day trading
            'swing_trading': [],   # Multi-day holds
            'position_trading': [], # Weeks to months
            'trend_following': []  # Long term
        }
        
        for fast, slow in all_combinations:
            if fast < 10 and slow < 25:
                styles['scalping'].append((fast, slow))
            elif fast < 15 and slow < 50:
                styles['short_term'].append((fast, slow))
            elif fast < 30 and slow < 100:
                styles['swing_trading'].append((fast, slow))
            elif fast < 100 and slow <= 200:
                styles['position_trading'].append((fast, slow))
            elif fast >= 50:
                styles['trend_following'].append((fast, slow))
        
        return styles
    
    def optimize_single_ma_type(
        self,
        data: pd.DataFrame,
        ma_type: str = "EMA",
        combinations: List[Tuple[int, int]] = None,
        optimization_metric: str = 'signal_frequency_pct',
        **common_params
    ) -> Dict[str, Any]:
        """
        Optimize parameters for a single MA type (e.g., all EMA combinations)
        
        Args:
            data: OHLCV DataFrame
            ma_type: "EMA", "SMA", "WMA", "VWMA"
            combinations: List of (fast, slow) periods to test
            optimization_metric: Metric to optimize for
            **common_params: Common strategy parameters
            
        Returns:
            Optimization results with best parameters
        """
        if combinations is None:
            combinations = self.get_valid_combinations(min_separation=3)
        
        logger.info(f"Optimizing {len(combinations)} {ma_type} combinations")
        
        results = []
        
        for fast, slow in combinations:
            try:
                strategy = create_ma_strategy(
                    fast_ma_type=ma_type,
                    fast_period=fast,
                    slow_ma_type=ma_type,
                    slow_period=slow,
                    **common_params
                )
                
                result = strategy.generate_signals(data)
                
                # Extract metrics
                signal_freq = result.metadata['signal_analysis']['signal_frequency_pct']
                avg_strength = result.metadata['signal_analysis']['avg_signal_strength']
                
                results.append({
                    'ma_type': ma_type,
                    'fast_period': fast,
                    'slow_period': slow,
                    'total_signals': result.total_signals,
                    'buy_signals': result.buy_signals,
                    'sell_signals': result.sell_signals,
                    'signal_frequency_pct': signal_freq,
                    'avg_signal_strength': avg_strength,
                    'crossovers': result.metadata.get('crossover_analysis', {}).get('total_crossovers', 0)
                })
                
            except Exception as e:
                logger.warning(f"Failed to test {ma_type}({fast}, {slow}): {str(e)}")
        
        if not results:
            raise ValueError(f"No valid results for {ma_type} optimization")
        
        # Convert to DataFrame and find best
        results_df = pd.DataFrame(results)
        best_idx = results_df[optimization_metric].idxmax()
        best_result = results_df.iloc[best_idx]
        
        return {
            'ma_type': ma_type,
            'best_parameters': {
                'fast_period': int(best_result['fast_period']),
                'slow_period': int(best_result['slow_period'])
            },
            'best_metrics': best_result.to_dict(),
            'optimization_metric': optimization_metric,
            'total_tested': len(results),
            'all_results': results_df,
            'top_5': results_df.nlargest(5, optimization_metric)[
                ['fast_period', 'slow_period', optimization_metric]
            ].to_dict('records')
        }
    
    def compare_ma_types(
        self,
        data: pd.DataFrame,
        ma_types: List[str] = None,
        period_pairs: List[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Compare different MA types with same period combinations
        
        Args:
            data: OHLCV DataFrame
            ma_types: List of MA types to compare
            period_pairs: List of (fast, slow) period combinations
            
        Returns:
            Comparison results across MA types
        """
        if ma_types is None:
            ma_types = ['EMA', 'SMA', 'WMA', 'VWMA']
        
        if period_pairs is None:
            period_pairs = [(9, 21), (10, 20), (12, 26), (20, 50)]
        
        logger.info(f"Comparing {len(ma_types)} MA types across {len(period_pairs)} period combinations")
        
        comparison_results = []
        
        for ma_type in ma_types:
            for fast, slow in period_pairs:
                try:
                    strategy = create_ma_strategy(ma_type, fast, ma_type, slow)
                    result = strategy.generate_signals(data)
                    
                    comparison_results.append({
                        'ma_type': ma_type,
                        'fast_period': fast,
                        'slow_period': slow,
                        'combination': f"{ma_type}({fast}/{slow})",
                        'total_signals': result.total_signals,
                        'signal_frequency_pct': result.metadata.get('signal_analysis', {}).get('signal_frequency_pct', 0),
                        'avg_signal_strength': result.metadata.get('signal_analysis', {}).get('avg_signal_strength', 0),
                        'crossovers': result.metadata.get('crossover_analysis', {}).get('total_crossovers', 0)
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to test {ma_type}({fast}, {slow}): {str(e)}")
        
        results_df = pd.DataFrame(comparison_results)
        
        # Summary by MA type
        ma_type_summary = results_df.groupby('ma_type').agg({
            'total_signals': 'mean',
            'signal_frequency_pct': 'mean',
            'avg_signal_strength': 'mean'
        }).round(2)
        
        return {
            'comparison_results': results_df,
            'ma_type_summary': ma_type_summary,
            'best_overall': results_df.loc[results_df['signal_frequency_pct'].idxmax()].to_dict(),
            'best_by_ma_type': {
                ma_type: group.loc[group['signal_frequency_pct'].idxmax()].to_dict()
                for ma_type, group in results_df.groupby('ma_type')
            }
        }
    
    def grid_search_optimization(
        self,
        data: pd.DataFrame,
        fast_ma_types: List[str] = None,
        slow_ma_types: List[str] = None,
        fast_periods: List[int] = None,
        slow_periods: List[int] = None,
        max_combinations: int = 100
    ) -> pd.DataFrame:
        """
        Comprehensive grid search across all parameter combinations
        
        Args:
            data: OHLCV DataFrame
            fast_ma_types: MA types for fast MA
            slow_ma_types: MA types for slow MA
            fast_periods: Periods for fast MA
            slow_periods: Periods for slow MA
            max_combinations: Maximum combinations to test
            
        Returns:
            DataFrame with all results
        """
        # Set defaults
        if fast_ma_types is None:
            fast_ma_types = ['EMA', 'SMA']
        if slow_ma_types is None:
            slow_ma_types = ['EMA', 'SMA']
        if fast_periods is None:
            fast_periods = [5, 9, 10, 12, 15, 20]
        if slow_periods is None:
            slow_periods = [15, 20, 21, 26, 30, 50]
        
        # Generate all combinations
        all_combinations = []
        for fast_ma in fast_ma_types:
            for slow_ma in slow_ma_types:
                for fast_period in fast_periods:
                    for slow_period in slow_periods:
                        if fast_period < slow_period:
                            all_combinations.append((fast_ma, fast_period, slow_ma, slow_period))
        
        # Limit combinations for performance
        if len(all_combinations) > max_combinations:
            logger.warning(f"Limiting grid search to {max_combinations} combinations")
            all_combinations = all_combinations[:max_combinations]
        
        logger.info(f"Grid search testing {len(all_combinations)} combinations")
        
        results = []
        for i, (fast_ma, fast_period, slow_ma, slow_period) in enumerate(all_combinations):
            try:
                strategy = create_ma_strategy(fast_ma, fast_period, slow_ma, slow_period)
                result = strategy.generate_signals(data)
                
                results.append({
                    'fast_ma_type': fast_ma,
                    'fast_period': fast_period,
                    'slow_ma_type': slow_ma,
                    'slow_period': slow_period,
                    'combination': f"{fast_ma}({fast_period})/{slow_ma}({slow_period})",
                    'total_signals': result.total_signals,
                    'signal_frequency_pct': result.metadata.get('signal_analysis', {}).get('signal_frequency_pct', 0),
                    'avg_signal_strength': result.metadata.get('signal_analysis', {}).get('avg_signal_strength', 0),
                    'crossovers': result.metadata.get('crossover_analysis', {}).get('total_crossovers', 0)
                })
                
                if (i + 1) % 20 == 0:
                    logger.info(f"Completed {i + 1}/{len(all_combinations)} combinations")
                    
            except Exception as e:
                logger.warning(f"Failed combination {fast_ma}({fast_period})/{slow_ma}({slow_period}): {str(e)}")
        
        return pd.DataFrame(results)


# === QUICK TESTING FUNCTIONS ===

def test_ma_combination(
    data: pd.DataFrame,
    fast_ma_type: str,
    fast_period: int,
    slow_ma_type: str,
    slow_period: int,
    **kwargs
) -> StrategyResult:
    """
    Quick test of a single MA combination
    
    Example:
        result = test_ma_combination(data, "EMA", 9, "SMA", 21)
    """
    strategy = create_ma_strategy(fast_ma_type, fast_period, slow_ma_type, slow_period, **kwargs)
    return strategy.generate_signals(data)


def quick_style_scan(
    data: pd.DataFrame,
    trading_style: str = 'swing_trading',
    ma_type: str = 'EMA'
) -> pd.DataFrame:
    """
    Quick scan of combinations for a specific trading style
    
    Args:
        data: OHLCV DataFrame
        trading_style: 'scalping', 'short_term', 'swing_trading', 'position_trading', 'trend_following'
        ma_type: MA type to use for both fast and slow
        
    Returns:
        DataFrame with results sorted by signal frequency
    """
    optimizer = MAParameterOptimizer()
    style_combinations = optimizer.get_combinations_by_style()
    
    if trading_style not in style_combinations:
        raise ValueError(f"Invalid style. Choose from: {list(style_combinations.keys())}")
    
    combinations = style_combinations[trading_style]
    if not combinations:
        raise ValueError(f"No combinations available for {trading_style}")
    
    results = []
    for fast, slow in combinations:
        try:
            result = test_ma_combination(data, ma_type, fast, ma_type, slow)
            results.append({
                'fast_period': fast,
                'slow_period': slow,
                'combination': f"{ma_type}({fast}/{slow})",
                'total_signals': result.total_signals,
                'signal_frequency_pct': result.metadata.get('signal_analysis', {}).get('signal_frequency_pct', 0),
                'avg_confidence': result.avg_confidence
            })
        except Exception as e:
            logger.warning(f"Failed {ma_type}({fast}/{slow}): {str(e)}")
    
    df = pd.DataFrame(results)
    return df.sort_values('signal_frequency_pct', ascending=False)


def find_best_parameters(
    data: pd.DataFrame,
    ma_type: str = 'EMA',
    metric: str = 'signal_frequency_pct'
) -> Dict[str, Any]:
    """
    Find the best parameters for a given MA type and metric
    
    Example:
        best = find_best_parameters(data, ma_type='EMA', metric='signal_frequency_pct')
        print(f"Best EMA combination: {best['fast_period']}/{best['slow_period']}")
    """
    optimizer = MAParameterOptimizer()
    return optimizer.optimize_single_ma_type(data, ma_type, optimization_metric=metric)


# === CONFIGURATION MANAGEMENT ===

def save_strategy_config(config: MAConfig, filepath: str):
    """Save strategy configuration to file"""
    config.save_to_file(filepath)
    logger.info(f"Strategy configuration saved to {filepath}")


def load_strategy_config(filepath: str) -> MAConfig:
    """Load strategy configuration from file"""
    config = MAConfig.load_from_file(filepath)
    logger.info(f"Strategy configuration loaded from {filepath}")
    return config


def create_preset_configs() -> Dict[str, MAConfig]:
    """Create preset configurations for common strategies"""
    presets = {}
    
    popular_configs = get_popular_configurations()
    for preset in popular_configs:
        name = preset['name']
        config_dict = preset['config']
        presets[name] = MAConfig.from_dict(config_dict)
    
    return presets


# === EXAMPLE USAGE FUNCTIONS ===

def example_basic_usage(data: pd.DataFrame):
    """Example showing basic usage"""
    print("=== Basic MA Crossover Strategy Usage ===")
    
    # Method 1: Simple creation
    strategy = create_ma_strategy("EMA", 9, "EMA", 21)
    result = strategy.generate_signals(data)
    print(f"EMA 9/21: {result.total_signals} signals")
    
    # Method 2: Using configuration
    config = MAConfig(
        fast_ma_type=MAType.EMA, fast_period=9,
        slow_ma_type=MAType.SMA, slow_period=21,
        min_separation=0.2
    )
    strategy = MACrossoverStrategy(config)
    result = strategy.generate_signals(data)
    print(f"EMA 9/SMA 21: {result.total_signals} signals")


def example_optimization(data: pd.DataFrame):
    """Example showing optimization usage"""
    print("=== Parameter Optimization Example ===")
    
    # Find best EMA combination
    best_ema = find_best_parameters(data, 'EMA')
    print(f"Best EMA: {best_ema['best_parameters']}")
    
    # Compare trading styles
    swing_results = quick_style_scan(data, 'swing_trading', 'EMA')
    print(f"Top swing trading combination: {swing_results.iloc[0]['combination']}")
    
    # Compare MA types
    optimizer = MAParameterOptimizer()
    comparison = optimizer.compare_ma_types(data)
    print(f"Best overall: {comparison['best_overall']['combination']}")


if __name__ == "__main__":
    # This file is ready to use!
    # Example workflow:
    
    # 1. Load your data (from Phase 2)
    # data = fetcher.get_historical_data("NSE:RELIANCE-EQ", "15m", years_back=2)
    
    # 2. Test specific combination
    # result = test_ma_combination(data, "EMA", 9, "SMA", 21)
    
    # 3. Find best parameters
    # best = find_best_parameters(data, ma_type='EMA')
    
    # 4. Run optimization
    # optimizer = MAParameterOptimizer()
    # results = optimizer.grid_search_optimization(data)
    
    print("✅ Unified MA Crossover Strategy System Ready!")
    print("📊 Supports: EMA, SMA, WMA, VWMA in any combination")
    print("🎯 Features: Flexible parameters, optimization, UI-ready interface")
    print("🚀 Usage: See example functions above")