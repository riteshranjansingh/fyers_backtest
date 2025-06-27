"""
Base Strategy Architecture with Signal Generation Framework
Foundation for all trading strategies with consistent interface
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StrategyError(Exception):
    """Custom exception for strategy-related errors"""
    pass


class Signal:
    """
    Represents a trading signal with all relevant information
    """
    
    def __init__(
        self,
        timestamp: datetime,
        signal_type: str,  # 'BUY', 'SELL', 'HOLD'
        price: float,
        confidence: float = 1.0,
        metadata: Dict = None
    ):
        self.timestamp = timestamp
        self.signal_type = signal_type.upper()
        self.price = price
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp between 0-1
        self.metadata = metadata or {}
        
        # Validate signal type
        if self.signal_type not in ['BUY', 'SELL', 'HOLD']:
            raise StrategyError(f"Invalid signal type: {signal_type}")
    
    def __repr__(self):
        return f"Signal({self.timestamp}, {self.signal_type}, {self.price:.2f}, conf={self.confidence:.2f})"
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary"""
        return {
            'timestamp': self.timestamp,
            'signal_type': self.signal_type,
            'price': self.price,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class StrategyResult:
    """
    Contains comprehensive results from strategy execution
    """
    
    def __init__(
        self,
        strategy_name: str,
        signals: List[Signal],
        signal_series: pd.Series,
        metadata: Dict = None
    ):
        self.strategy_name = strategy_name
        self.signals = signals
        self.signal_series = signal_series  # For easy plotting/analysis
        self.metadata = metadata or {}
        
        # Calculate summary statistics
        self.total_signals = len(signals)
        self.buy_signals = sum(1 for s in signals if s.signal_type == 'BUY')
        self.sell_signals = sum(1 for s in signals if s.signal_type == 'SELL')
        self.avg_confidence = np.mean([s.confidence for s in signals]) if signals else 0.0
    
    def get_signals_df(self) -> pd.DataFrame:
        """Convert signals to DataFrame for analysis"""
        if not self.signals:
            return pd.DataFrame()
        
        data = []
        for signal in self.signals:
            row = signal.to_dict()
            # Flatten metadata
            for key, value in signal.metadata.items():
                row[f"meta_{key}"] = value
            data.append(row)
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def summary(self) -> Dict:
        """Get strategy execution summary"""
        return {
            'strategy_name': self.strategy_name,
            'total_signals': self.total_signals,
            'buy_signals': self.buy_signals,
            'sell_signals': self.sell_signals,
            'avg_confidence': self.avg_confidence,
            'signal_frequency': f"{self.total_signals / len(self.signal_series) * 100:.1f}%" if len(self.signal_series) > 0 else "0%",
            'metadata': self.metadata
        }


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    
    Provides consistent interface for signal generation, parameter management,
    and integration with the backtesting system.
    """
    
    def __init__(self, name: str, **parameters):
        """
        Initialize base strategy
        
        Args:
            name: Strategy name
            **parameters: Strategy-specific parameters
        """
        self.name = name
        self.parameters = parameters
        self.data = None
        self.result = None
        self._is_fitted = False
        
        # Validate parameters
        self._validate_parameters()
        
        # Strategy metadata
        self.created_at = datetime.now()
        self.version = "1.0.0"
        
        logger.info(f"Initialized strategy: {self.name} with parameters: {parameters}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> StrategyResult:
        """
        Generate trading signals based on data
        
        Args:
            data: OHLCV DataFrame with DatetimeIndex
            **kwargs: Runtime parameter overrides
            
        Returns:
            StrategyResult with signals and metadata
        """
        pass
    
    def __call__(self, data: pd.DataFrame, **kwargs) -> StrategyResult:
        """
        Make strategy callable: strategy(data)
        
        Args:
            data: OHLCV DataFrame
            **kwargs: Runtime parameter overrides
            
        Returns:
            StrategyResult
        """
        return self.generate_signals(data, **kwargs)
    
    def _validate_data(self, data: pd.DataFrame):
        """Validate input data format"""
        if data is None or data.empty:
            raise StrategyError(f"{self.name}: Data cannot be None or empty")
        
        if not isinstance(data, pd.DataFrame):
            raise StrategyError(f"{self.name}: Data must be a pandas DataFrame")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise StrategyError(f"{self.name}: Data must have DatetimeIndex")
        
        # Check for required OHLCV columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise StrategyError(f"{self.name}: Missing required columns: {missing_columns}")
        
        # Check for minimum data points
        min_periods = self.get_minimum_periods()
        if len(data) < min_periods:
            raise StrategyError(
                f"{self.name}: Insufficient data. Need {min_periods} periods, got {len(data)}"
            )
    
    def _validate_parameters(self):
        """Validate strategy parameters (override in subclasses)"""
        pass
    
    def get_minimum_periods(self) -> int:
        """Get minimum number of periods required (override in subclasses)"""
        return 50  # Default minimum
    
    def get_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'fitted': self._is_fitted,
            'created_at': self.created_at,
            'version': self.version,
            'minimum_periods': self.get_minimum_periods(),
            'data_shape': self.data.shape if self.data is not None else None
        }
    
    def update_parameters(self, **new_parameters):
        """Update strategy parameters"""
        old_parameters = self.parameters.copy()
        self.parameters.update(new_parameters)
        
        try:
            self._validate_parameters()
            logger.info(f"{self.name}: Updated parameters from {old_parameters} to {self.parameters}")
        except Exception as e:
            # Rollback on validation error
            self.parameters = old_parameters
            raise StrategyError(f"Parameter update failed: {str(e)}")
    
    def _create_signal_series(self, data: pd.DataFrame, signals: List[Signal]) -> pd.Series:
        """
        Create a pandas Series with signal values for easy analysis
        
        Args:
            data: Original data DataFrame
            signals: List of Signal objects
            
        Returns:
            Series with signal values (1=BUY, -1=SELL, 0=HOLD)
        """
        # Initialize with HOLD (0)
        signal_series = pd.Series(0, index=data.index, name=f"{self.name}_signals")
        
        # Map signals to series
        for signal in signals:
            if signal.timestamp in signal_series.index:
                if signal.signal_type == 'BUY':
                    signal_series.loc[signal.timestamp] = 1
                elif signal.signal_type == 'SELL':
                    signal_series.loc[signal.timestamp] = -1
                # HOLD remains 0
        
        return signal_series
    
    def _create_signals_from_series(
        self, 
        data: pd.DataFrame, 
        signal_series: pd.Series,
        price_column: str = 'close'
    ) -> List[Signal]:
        """
        Create Signal objects from a pandas Series
        
        Args:
            data: OHLCV data
            signal_series: Series with signal values (1=BUY, -1=SELL, 0=HOLD)
            price_column: Column to use for signal price
            
        Returns:
            List of Signal objects
        """
        signals = []
        
        for timestamp, signal_value in signal_series.items():
            if signal_value != 0:  # Only non-hold signals
                signal_type = 'BUY' if signal_value > 0 else 'SELL'
                price = data.loc[timestamp, price_column]
                
                signal = Signal(
                    timestamp=timestamp,
                    signal_type=signal_type,
                    price=price,
                    confidence=abs(signal_value),  # Use absolute value as confidence
                    metadata={'strategy': self.name, 'price_column': price_column}
                )
                signals.append(signal)
        
        return signals
    
    def __str__(self) -> str:
        """String representation"""
        params_str = ', '.join([f"{k}={v}" for k, v in self.parameters.items()])
        return f"{self.name}({params_str})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"<{self.__class__.__name__}: {self.__str__()}>"


class CrossoverStrategy(BaseStrategy):
    """
    Base class for crossover-based strategies
    Provides common functionality for MA crossovers, price crossovers, etc.
    """
    
    def __init__(self, name: str, **parameters):
        super().__init__(name, **parameters)
    
    def _detect_crossovers(
        self,
        fast_line: pd.Series,
        slow_line: pd.Series,
        min_separation: float = 0.0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect crossovers between two lines
        
        Args:
            fast_line: Faster moving line (e.g., EMA9)
            slow_line: Slower moving line (e.g., EMA21)
            min_separation: Minimum % separation to consider valid crossover
            
        Returns:
            Tuple of (bullish_crossovers, bearish_crossovers) as boolean Series
        """
        # Ensure same index
        common_index = fast_line.index.intersection(slow_line.index)
        fast = fast_line.reindex(common_index)
        slow = slow_line.reindex(common_index)
        
        # Previous values
        fast_prev = fast.shift(1)
        slow_prev = slow.shift(1)
        
        # Calculate separation percentage
        separation = abs(fast - slow) / slow * 100
        valid_separation = separation >= min_separation
        
        # Bullish crossover: fast was below slow, now above (with sufficient separation)
        bullish_cross = (
            (fast_prev <= slow_prev) & 
            (fast > slow) & 
            valid_separation
        )
        
        # Bearish crossover: fast was above slow, now below (with sufficient separation)
        bearish_cross = (
            (fast_prev >= slow_prev) & 
            (fast < slow) & 
            valid_separation
        )
        
        return bullish_cross, bearish_cross
    
    def _calculate_signal_strength(
        self,
        fast_line: pd.Series,
        slow_line: pd.Series,
        normalize: bool = True
    ) -> pd.Series:
        """
        Calculate signal strength based on separation between lines
        
        Args:
            fast_line: Faster line
            slow_line: Slower line
            normalize: Normalize to 0-1 range
            
        Returns:
            Series with signal strength values
        """
        # Calculate percentage separation
        separation = (fast_line - slow_line) / slow_line * 100
        
        if normalize:
            # Improved normalization: use broader range and avoid over-penalizing small separations
            window = min(50, len(separation) // 3)  # Shorter window for more responsive strength
            separation_abs = abs(separation)
            
            # Use 75th percentile instead of 95th for more reasonable scaling
            rolling_reference = separation_abs.rolling(window=window, min_periods=1).quantile(0.75)
            
            # Add minimum threshold to prevent division by very small numbers
            rolling_reference = rolling_reference.clip(lower=0.1)  # Min 0.1% separation
            
            strength = separation_abs / rolling_reference
            # Use sigmoid-like scaling for better distribution
            strength = 1 / (1 + np.exp(-3 * (strength - 0.5)))  # Sigmoid centered at 0.5
            strength = strength.clip(0, 1)  # Ensure 0-1 range
        else:
            strength = abs(separation)
        
        return strength


# Utility Functions

def combine_strategies(*strategies: BaseStrategy, method: str = 'unanimous') -> 'CombinedStrategy':
    """
    Combine multiple strategies into one
    
    Args:
        *strategies: Strategy instances to combine
        method: Combination method ('unanimous', 'majority', 'weighted')
        
    Returns:
        CombinedStrategy instance
    """
    return CombinedStrategy(strategies, method=method)


def validate_strategy_result(result: StrategyResult) -> bool:
    """Validate a strategy result"""
    try:
        if not isinstance(result, StrategyResult):
            return False
        
        if not isinstance(result.signals, list):
            return False
        
        if not isinstance(result.signal_series, pd.Series):
            return False
        
        # Check signals are valid
        for signal in result.signals:
            if not isinstance(signal, Signal):
                return False
        
        return True
    except Exception:
        return False


class CombinedStrategy(BaseStrategy):
    """
    Combines multiple strategies for confluence-based signals
    """
    
    def __init__(self, strategies: List[BaseStrategy], method: str = 'unanimous', **kwargs):
        """
        Initialize combined strategy
        
        Args:
            strategies: List of strategy instances
            method: Combination method ('unanimous', 'majority', 'weighted')
            **kwargs: Additional parameters
        """
        self.strategies = strategies
        self.method = method
        
        strategy_names = [s.name for s in strategies]
        name = f"Combined({method})_" + "_".join(strategy_names)
        
        super().__init__(name, method=method, **kwargs)
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> StrategyResult:
        """
        Generate combined signals from multiple strategies
        """
        self._validate_data(data)
        
        # Run all individual strategies
        individual_results = []
        for strategy in self.strategies:
            try:
                result = strategy.generate_signals(data, **kwargs)
                individual_results.append(result)
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {str(e)}")
        
        if not individual_results:
            raise StrategyError("No strategies produced valid results")
        
        # Combine signals based on method
        combined_series = self._combine_signal_series(individual_results)
        combined_signals = self._create_signals_from_series(data, combined_series)
        
        # Create metadata
        metadata = {
            'method': self.method,
            'individual_strategies': [r.strategy_name for r in individual_results],
            'individual_signal_counts': [r.total_signals for r in individual_results]
        }
        
        return StrategyResult(
            strategy_name=self.name,
            signals=combined_signals,
            signal_series=combined_series,
            metadata=metadata
        )
    
    def _combine_signal_series(self, results: List[StrategyResult]) -> pd.Series:
        """Combine signal series based on method"""
        # Align all series to same index
        all_series = [r.signal_series for r in results]
        combined_index = all_series[0].index
        for series in all_series[1:]:
            combined_index = combined_index.intersection(series.index)
        
        # Align all series
        aligned_series = [s.reindex(combined_index) for s in all_series]
        signals_df = pd.DataFrame(aligned_series).T
        
        if self.method == 'unanimous':
            # All strategies must agree
            combined = pd.Series(0, index=combined_index)
            # Buy: All must be positive
            all_buy = (signals_df > 0).all(axis=1)
            combined[all_buy] = 1
            # Sell: All must be negative
            all_sell = (signals_df < 0).all(axis=1)
            combined[all_sell] = -1
            
        elif self.method == 'majority':
            # Majority vote
            buy_votes = (signals_df > 0).sum(axis=1)
            sell_votes = (signals_df < 0).sum(axis=1)
            majority_threshold = len(self.strategies) / 2
            
            combined = pd.Series(0, index=combined_index)
            combined[buy_votes > majority_threshold] = 1
            combined[sell_votes > majority_threshold] = -1
            
        else:  # weighted (average)
            combined = signals_df.mean(axis=1)
            # Convert to discrete signals
            combined[combined > 0.3] = 1
            combined[combined < -0.3] = -1
            combined[abs(combined) <= 0.3] = 0
        
        combined.name = f"{self.name}_signals"
        return combined
    
    def get_minimum_periods(self) -> int:
        """Return maximum minimum periods from all strategies"""
        return max(s.get_minimum_periods() for s in self.strategies)