"""
🎯 Advanced Stop Loss Configuration Engine

Comprehensive stop loss system supporting multiple methods:
- Fixed percentage stops
- ATR-based dynamic stops  
- Support/Resistance level stops
- Adaptive stops based on market conditions

Author: Fyers Backtesting System
Date: 2025-06-27
"""

from dataclasses import dataclass
from typing import Dict, Union, Optional, Any, List
import pandas as pd
import numpy as np
import logging

from .risk_profiles import RiskConfiguration, StopLossMethod

logger = logging.getLogger(__name__)


@dataclass
class StopLossResult:
    """Result of stop loss calculation"""
    stop_price: float
    stop_percentage: float
    risk_per_share: float
    method_used: str
    details: Dict[str, Any]
    valid: bool = True
    warnings: list = None
    confidence: float = 1.0  # Confidence in stop loss placement
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class AdvancedStopLossEngine:
    """
    Advanced stop loss calculation engine supporting multiple methods
    """
    
    def __init__(self, config: RiskConfiguration):
        """Initialize stop loss engine with configuration"""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Advanced stop loss engine initialized with {config.stop_loss_method.value} method")
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        signal_direction: str,
        historical_data: Optional[pd.DataFrame] = None,
        symbol: str = "UNKNOWN",
        custom_config: Optional[RiskConfiguration] = None
    ) -> StopLossResult:
        """
        Calculate stop loss using configured method
        
        Args:
            entry_price: Entry price per share
            signal_direction: 'BUY' or 'SELL'
            historical_data: Historical price data for ATR/Support-Resistance calculations
            symbol: Trading symbol
            custom_config: Optional custom configuration override
            
        Returns:
            StopLossResult with calculated stop loss details
        """
        config = custom_config or self.config
        
        try:
            # Validate inputs
            if entry_price <= 0:
                raise ValueError(f"Entry price must be positive, got {entry_price}")
            if signal_direction not in ['BUY', 'SELL']:
                raise ValueError(f"Signal direction must be 'BUY' or 'SELL', got {signal_direction}")
            
            # Route to appropriate stop loss method
            if config.stop_loss_method == StopLossMethod.PERCENTAGE:
                result = self._calculate_percentage_stop(entry_price, signal_direction, config)
            elif config.stop_loss_method == StopLossMethod.ATR:
                result = self._calculate_atr_stop(entry_price, signal_direction, historical_data, config)
            elif config.stop_loss_method == StopLossMethod.SUPPORT_RESISTANCE:
                result = self._calculate_support_resistance_stop(entry_price, signal_direction, historical_data, config)
            elif config.stop_loss_method == StopLossMethod.ADAPTIVE:
                result = self._calculate_adaptive_stop(entry_price, signal_direction, historical_data, config)
            else:
                raise ValueError(f"Unknown stop loss method: {config.stop_loss_method}")
            
            # Apply stop loss limits and validation
            result = self._apply_stop_loss_limits(result, entry_price, config)
            
            # Log the result
            self.logger.info(
                f"Stop loss calculated for {symbol}: ₹{result.stop_price:.2f} "
                f"({result.stop_percentage:.2f}%) using {result.method_used} method"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Stop loss calculation error for {symbol}: {str(e)}")
            
            # Fallback to percentage method
            fallback_result = self._calculate_percentage_stop(entry_price, signal_direction, config)
            fallback_result.warnings.append(f"Primary method failed, using percentage fallback: {str(e)}")
            fallback_result.method_used = f"{config.stop_loss_method.value}_fallback"
            
            return fallback_result
    
    def _calculate_percentage_stop(
        self,
        entry_price: float,
        signal_direction: str,
        config: RiskConfiguration
    ) -> StopLossResult:
        """Calculate fixed percentage stop loss"""
        
        stop_percentage = config.stop_loss_percent
        
        if signal_direction == 'BUY':
            stop_price = entry_price * (1 - stop_percentage / 100)
        else:  # SELL
            stop_price = entry_price * (1 + stop_percentage / 100)
        
        risk_per_share = abs(entry_price - stop_price)
        
        details = {
            "stop_loss_percent": stop_percentage,
            "signal_direction": signal_direction,
            "calculation": f"Entry ₹{entry_price:.2f} ± {stop_percentage}% = ₹{stop_price:.2f}"
        }
        
        return StopLossResult(
            stop_price=stop_price,
            stop_percentage=stop_percentage,
            risk_per_share=risk_per_share,
            method_used="percentage",
            details=details,
            confidence=1.0
        )
    
    def _calculate_atr_stop(
        self,
        entry_price: float,
        signal_direction: str,
        historical_data: Optional[pd.DataFrame],
        config: RiskConfiguration
    ) -> StopLossResult:
        """Calculate ATR-based stop loss"""
        
        if historical_data is None or len(historical_data) < config.atr_period:
            # Fallback to percentage method
            self.logger.warning("Insufficient data for ATR calculation, using percentage fallback")
            result = self._calculate_percentage_stop(entry_price, signal_direction, config)
            result.warnings.append("Insufficient data for ATR, used percentage method")
            result.method_used = "atr_fallback"
            return result
        
        # Calculate ATR
        atr_value = self._calculate_atr(historical_data, config.atr_period)
        
        if atr_value <= 0:
            # Fallback to percentage method
            result = self._calculate_percentage_stop(entry_price, signal_direction, config)
            result.warnings.append("Invalid ATR value, used percentage method")
            result.method_used = "atr_fallback"
            return result
        
        # Calculate stop distance
        stop_distance = atr_value * config.atr_multiplier
        
        if signal_direction == 'BUY':
            stop_price = entry_price - stop_distance
        else:  # SELL
            stop_price = entry_price + stop_distance
        
        # Ensure stop price is positive
        if stop_price <= 0:
            result = self._calculate_percentage_stop(entry_price, signal_direction, config)
            result.warnings.append("ATR stop resulted in negative price, used percentage method")
            result.method_used = "atr_fallback"
            return result
        
        risk_per_share = abs(entry_price - stop_price)
        stop_percentage = (risk_per_share / entry_price) * 100
        
        # Calculate confidence based on ATR consistency
        recent_atr = self._calculate_atr(historical_data.tail(config.atr_period // 2), config.atr_period // 2)
        atr_stability = 1.0 - abs(atr_value - recent_atr) / max(atr_value, recent_atr, 0.01)
        confidence = max(0.5, min(1.0, atr_stability))
        
        details = {
            "atr_value": atr_value,
            "atr_period": config.atr_period,
            "atr_multiplier": config.atr_multiplier,
            "stop_distance": stop_distance,
            "signal_direction": signal_direction,
            "atr_stability": atr_stability,
            "calculation": f"Entry ₹{entry_price:.2f} ± (ATR {atr_value:.2f} × {config.atr_multiplier}) = ₹{stop_price:.2f}"
        }
        
        warnings = []
        if confidence < 0.8:
            warnings.append(f"Low ATR stability ({atr_stability:.2f}), stop loss may be less reliable")
        
        return StopLossResult(
            stop_price=stop_price,
            stop_percentage=stop_percentage,
            risk_per_share=risk_per_share,
            method_used="atr",
            details=details,
            confidence=confidence,
            warnings=warnings
        )
    
    def _calculate_support_resistance_stop(
        self,
        entry_price: float,
        signal_direction: str,
        historical_data: Optional[pd.DataFrame],
        config: RiskConfiguration
    ) -> StopLossResult:
        """Calculate support/resistance-based stop loss"""
        
        if historical_data is None or len(historical_data) < 20:
            # Fallback to percentage method
            result = self._calculate_percentage_stop(entry_price, signal_direction, config)
            result.warnings.append("Insufficient data for support/resistance, used percentage method")
            result.method_used = "support_resistance_fallback"
            return result
        
        # Find support and resistance levels
        support_resistance = self._find_support_resistance_levels(historical_data)
        
        if signal_direction == 'BUY':
            # Find nearest support level below entry
            relevant_levels = [level for level in support_resistance['support'] if level < entry_price]
            if relevant_levels:
                nearest_support = max(relevant_levels)
                # Place stop slightly below support
                buffer = entry_price * 0.002  # 0.2% buffer
                stop_price = nearest_support - buffer
            else:
                # No support found, use percentage method
                result = self._calculate_percentage_stop(entry_price, signal_direction, config)
                result.warnings.append("No support level found, used percentage method")
                result.method_used = "support_resistance_fallback"
                return result
        else:  # SELL
            # Find nearest resistance level above entry
            relevant_levels = [level for level in support_resistance['resistance'] if level > entry_price]
            if relevant_levels:
                nearest_resistance = min(relevant_levels)
                # Place stop slightly above resistance
                buffer = entry_price * 0.002  # 0.2% buffer
                stop_price = nearest_resistance + buffer
            else:
                # No resistance found, use percentage method
                result = self._calculate_percentage_stop(entry_price, signal_direction, config)
                result.warnings.append("No resistance level found, used percentage method")
                result.method_used = "support_resistance_fallback"
                return result
        
        # Ensure stop price is positive and reasonable
        if stop_price <= 0 or abs(entry_price - stop_price) / entry_price > 0.1:  # More than 10% stop
            result = self._calculate_percentage_stop(entry_price, signal_direction, config)
            result.warnings.append("Support/resistance stop too wide, used percentage method")
            result.method_used = "support_resistance_fallback"
            return result
        
        risk_per_share = abs(entry_price - stop_price)
        stop_percentage = (risk_per_share / entry_price) * 100
        
        # Calculate confidence based on level strength
        level_strength = support_resistance['strength'].get(
            nearest_support if signal_direction == 'BUY' else nearest_resistance, 0.5
        )
        confidence = max(0.6, min(1.0, level_strength))
        
        details = {
            "signal_direction": signal_direction,
            "support_levels": support_resistance['support'][:3],  # Top 3
            "resistance_levels": support_resistance['resistance'][:3],  # Top 3
            "chosen_level": nearest_support if signal_direction == 'BUY' else nearest_resistance,
            "level_strength": level_strength,
            "buffer_applied": entry_price * 0.002,
            "calculation": f"Entry ₹{entry_price:.2f} → {'Support' if signal_direction == 'BUY' else 'Resistance'} ₹{stop_price:.2f}"
        }
        
        warnings = []
        if confidence < 0.8:
            warnings.append(f"Weak support/resistance level (strength: {level_strength:.2f})")
        
        return StopLossResult(
            stop_price=stop_price,
            stop_percentage=stop_percentage,
            risk_per_share=risk_per_share,
            method_used="support_resistance",
            details=details,
            confidence=confidence,
            warnings=warnings
        )
    
    def _calculate_adaptive_stop(
        self,
        entry_price: float,
        signal_direction: str,
        historical_data: Optional[pd.DataFrame],
        config: RiskConfiguration
    ) -> StopLossResult:
        """Calculate adaptive stop loss based on market conditions"""
        
        if historical_data is None or len(historical_data) < 20:
            # Fallback to percentage method
            result = self._calculate_percentage_stop(entry_price, signal_direction, config)
            result.warnings.append("Insufficient data for adaptive stop, used percentage method")
            result.method_used = "adaptive_fallback"
            return result
        
        # Calculate market volatility
        volatility = self._calculate_volatility(historical_data)
        
        # Calculate multiple stop methods
        atr_result = self._calculate_atr_stop(entry_price, signal_direction, historical_data, config)
        percentage_result = self._calculate_percentage_stop(entry_price, signal_direction, config)
        
        # Adaptive logic: choose based on market conditions
        if volatility > 0.03:  # High volatility (>3% daily)
            # Use wider stops in volatile markets
            if atr_result.valid and atr_result.stop_percentage > percentage_result.stop_percentage:
                chosen_result = atr_result
                adaptation_reason = "High volatility - using wider ATR stop"
            else:
                # Use percentage but increase it
                adaptive_percentage = config.stop_loss_percent * config.adaptive_volatility_factor
                adaptive_config = RiskConfiguration(**config.to_dict())
                adaptive_config.stop_loss_percent = adaptive_percentage
                chosen_result = self._calculate_percentage_stop(entry_price, signal_direction, adaptive_config)
                adaptation_reason = f"High volatility - increased stop to {adaptive_percentage:.1f}%"
        else:  # Normal/Low volatility
            # Use tighter stops in stable markets
            if atr_result.valid and atr_result.stop_percentage < percentage_result.stop_percentage:
                chosen_result = atr_result
                adaptation_reason = "Low volatility - using tighter ATR stop"
            else:
                chosen_result = percentage_result
                adaptation_reason = "Low volatility - using standard percentage stop"
        
        # Update result details
        chosen_result.method_used = "adaptive"
        chosen_result.details.update({
            "adaptation_logic": {
                "market_volatility": volatility,
                "volatility_category": "high" if volatility > 0.03 else "normal",
                "adaptation_reason": adaptation_reason,
                "atr_stop_percentage": atr_result.stop_percentage if atr_result.valid else None,
                "percentage_stop": percentage_result.stop_percentage,
                "chosen_method": chosen_result.method_used
            }
        })
        
        # Add adaptive-specific warnings
        if volatility > 0.05:
            chosen_result.warnings.append(f"Very high market volatility ({volatility*100:.1f}%)")
        
        return chosen_result
    
    def _apply_stop_loss_limits(
        self,
        result: StopLossResult,
        entry_price: float,
        config: RiskConfiguration
    ) -> StopLossResult:
        """Apply stop loss limits and validation"""
        
        # Check minimum stop distance (0.1%)
        min_stop_distance = entry_price * 0.001
        if result.risk_per_share < min_stop_distance:
            result.warnings.append("Stop loss too tight, may result in premature exits")
        
        # Check maximum stop distance (10%)
        max_stop_distance = entry_price * 0.10
        if result.risk_per_share > max_stop_distance:
            result.warnings.append("Stop loss very wide, high risk per trade")
            result.confidence *= 0.8  # Reduce confidence for wide stops
        
        # Ensure stop price is reasonable
        if result.stop_price <= 0:
            result.valid = False
            result.warnings.append("Invalid stop price calculated")
        
        return result
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> float:
        """Calculate Average True Range"""
        try:
            # Ensure we have required columns
            required_cols = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_cols):
                return 0.0
            
            high = data['high']
            low = data['low']
            close = data['close']
            prev_close = close.shift(1)
            
            # Calculate true range
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = true_range.rolling(window=period, min_periods=1).mean().iloc[-1]
            
            return max(0.0, atr) if not pd.isna(atr) else 0.0
            
        except Exception as e:
            self.logger.warning(f"ATR calculation error: {str(e)}")
            return 0.0
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate recent market volatility"""
        try:
            if 'close' not in data.columns or len(data) < 10:
                return 0.02  # Default 2% volatility
            
            # Calculate daily returns
            returns = data['close'].pct_change().dropna()
            
            # Calculate rolling volatility (standard deviation of returns)
            volatility = returns.rolling(window=min(20, len(returns))).std().iloc[-1]
            
            return max(0.005, volatility) if not pd.isna(volatility) else 0.02
            
        except Exception as e:
            self.logger.warning(f"Volatility calculation error: {str(e)}")
            return 0.02
    
    def _find_support_resistance_levels(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Find support and resistance levels using pivot points"""
        try:
            if len(data) < 10:
                return {'support': [], 'resistance': [], 'strength': {}}
            
            # Simple pivot point detection
            highs = data['high']
            lows = data['low']
            
            # Find local peaks (resistance) and valleys (support)
            resistance_levels = []
            support_levels = []
            strength_scores = {}
            
            window = 5  # Look for pivots in 5-period windows
            
            for i in range(window, len(data) - window):
                # Check for resistance (local high)
                if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
                    resistance_levels.append(highs.iloc[i])
                    # Strength based on how many times price touched this level
                    touches = sum(1 for j in range(max(0, i-20), min(len(data), i+20)) 
                                if abs(highs.iloc[j] - highs.iloc[i]) / highs.iloc[i] < 0.002)
                    strength_scores[highs.iloc[i]] = min(1.0, touches / 5.0)
                
                # Check for support (local low)
                if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
                    support_levels.append(lows.iloc[i])
                    # Strength based on how many times price touched this level
                    touches = sum(1 for j in range(max(0, i-20), min(len(data), i+20)) 
                                if abs(lows.iloc[j] - lows.iloc[i]) / lows.iloc[i] < 0.002)
                    strength_scores[lows.iloc[i]] = min(1.0, touches / 5.0)
            
            # Sort and return top levels
            resistance_levels = sorted(set(resistance_levels), reverse=True)[:5]
            support_levels = sorted(set(support_levels))[:5]
            
            return {
                'support': support_levels,
                'resistance': resistance_levels,
                'strength': strength_scores
            }
            
        except Exception as e:
            self.logger.warning(f"Support/resistance calculation error: {str(e)}")
            return {'support': [], 'resistance': [], 'strength': {}}
    
    def get_stop_loss_summary(self) -> Dict[str, Any]:
        """Get summary of current stop loss configuration"""
        return {
            "method": self.config.stop_loss_method.value,
            "stop_loss_percent": self.config.stop_loss_percent,
            "atr_multiplier": self.config.atr_multiplier,
            "atr_period": self.config.atr_period,
            "adaptive_volatility_factor": self.config.adaptive_volatility_factor,
            "enable_trailing_stops": self.config.enable_trailing_stops
        }


if __name__ == "__main__":
    # Example usage and testing
    from .risk_profiles import RiskProfileManager
    import pandas as pd
    
    print("🎯 Advanced Stop Loss Engine Test")
    print("=" * 50)
    
    # Create sample historical data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    price = 2500
    prices = [price]
    
    for _ in range(99):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        price = price * (1 + change)
        prices.append(price)
    
    sample_data = pd.DataFrame({
        'datetime': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [100000] * 100
    })
    
    # Test parameters
    entry_price = 2500.0
    signal_direction = 'BUY'
    
    # Test with different profiles
    profiles = RiskProfileManager.get_all_profiles()
    
    for profile_name, config in profiles.items():
        print(f"\n📊 Testing {profile_name.upper()} Profile:")
        print("-" * 30)
        
        engine = AdvancedStopLossEngine(config)
        result = engine.calculate_stop_loss(entry_price, signal_direction, sample_data)
        
        print(f"Method: {result.method_used}")
        print(f"Stop Price: ₹{result.stop_price:.2f}")
        print(f"Stop Percentage: {result.stop_percentage:.2f}%")
        print(f"Risk per Share: ₹{result.risk_per_share:.2f}")
        print(f"Confidence: {result.confidence:.2f}")
        
        if result.warnings:
            print(f"Warnings: {', '.join(result.warnings)}")
    
    print("\n✅ Advanced stop loss tests completed!")