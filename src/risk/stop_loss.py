"""
Stop Loss Manager
Handles initial stop loss calculation and basic stop loss management
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class StopLossConfig:
    """Configuration for stop loss management"""
    # Basic stop loss settings
    default_stop_pct: float = 1.5  # Default 1.5% stop loss
    min_stop_pct: float = 0.5  # Minimum 0.5% stop loss
    max_stop_pct: float = 5.0  # Maximum 5% stop loss
    
    # ATR-based settings
    atr_multiplier: float = 2.0  # Standard 2x ATR for stop loss
    atr_period: int = 14  # ATR calculation period
    
    # Volatility-based adjustments
    high_volatility_threshold: float = 3.0  # Above 3% daily volatility
    low_volatility_threshold: float = 1.0  # Below 1% daily volatility
    volatility_adjustment_factor: float = 0.5  # Adjustment multiplier
    
    # Time-based adjustments
    intraday_stop_multiplier: float = 0.8  # Tighter stops for intraday
    swing_stop_multiplier: float = 1.2  # Wider stops for swing trades
    
    # Gap protection
    max_gap_protection_pct: float = 2.0  # Maximum gap protection
    enable_gap_protection: bool = True


class StopLossManager:
    """
    Advanced stop loss management system
    
    Features:
    - Multiple stop loss calculation methods
    - Volatility-based adjustments
    - Time-based stop modifications
    - Gap protection
    """
    
    def __init__(self, config: StopLossConfig = None):
        """Initialize stop loss manager"""
        self.config = config or StopLossConfig()
        
        # Track active stop losses
        self.active_stops = {}
        
        logger.info(f"Stop loss manager initialized with {self.config.default_stop_pct}% default stop")
    
    def calculate_initial_stop_loss(
        self,
        entry_price: float,
        direction: str,
        method: str = "percentage",
        **kwargs
    ) -> Dict[str, Union[float, str]]:
        """
        Calculate initial stop loss using various methods
        
        Args:
            entry_price: Entry price for the trade
            direction: 'BUY' or 'SELL'
            method: 'percentage', 'atr', 'support_resistance', 'volatility'
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Dictionary with stop loss details
        """
        direction = direction.upper()
        if direction not in ['BUY', 'SELL']:
            raise ValueError("Direction must be 'BUY' or 'SELL'")
        
        result = {
            'entry_price': entry_price,
            'direction': direction,
            'method': method,
            'timestamp': datetime.now()
        }
        
        try:
            if method == "percentage":
                result.update(self._calculate_percentage_stop(entry_price, direction, **kwargs))
            elif method == "atr":
                result.update(self._calculate_atr_stop(entry_price, direction, **kwargs))
            elif method == "support_resistance":
                result.update(self._calculate_sr_stop(entry_price, direction, **kwargs))
            elif method == "volatility":
                result.update(self._calculate_volatility_stop(entry_price, direction, **kwargs))
            else:
                raise ValueError(f"Unknown stop loss method: {method}")
            
            # Apply time-based adjustments
            result = self._apply_time_adjustments(result, **kwargs)
            
            # Apply gap protection if enabled
            if self.config.enable_gap_protection:
                result = self._apply_gap_protection(result)
            
            result['valid'] = True
            
        except Exception as e:
            logger.error(f"Stop loss calculation error: {str(e)}")
            result['error'] = str(e)
            result['valid'] = False
        
        return result
    
    def _calculate_percentage_stop(
        self,
        entry_price: float,
        direction: str,
        stop_pct: float = None,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate percentage-based stop loss"""
        stop_percentage = stop_pct or self.config.default_stop_pct
        
        # Clamp to min/max limits
        stop_percentage = max(
            self.config.min_stop_pct,
            min(stop_percentage, self.config.max_stop_pct)
        )
        
        if direction == 'BUY':
            stop_price = entry_price * (1 - stop_percentage / 100)
        else:
            stop_price = entry_price * (1 + stop_percentage / 100)
        
        risk_amount = abs(entry_price - stop_price)
        
        return {
            'stop_price': stop_price,
            'stop_percentage': stop_percentage,
            'risk_per_share': risk_amount,
            'risk_percentage': (risk_amount / entry_price) * 100
        }
    
    def _calculate_atr_stop(
        self,
        entry_price: float,
        direction: str,
        atr_value: float = None,
        data: pd.DataFrame = None,
        atr_multiplier: float = None,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate ATR-based stop loss"""
        if atr_value is None and data is not None:
            atr_value = self._calculate_atr(data, self.config.atr_period)
        
        if atr_value is None:
            raise ValueError("ATR value or historical data required for ATR stop loss")
        
        multiplier = atr_multiplier or self.config.atr_multiplier
        
        if direction == 'BUY':
            stop_price = entry_price - (atr_value * multiplier)
        else:
            stop_price = entry_price + (atr_value * multiplier)
        
        risk_amount = abs(entry_price - stop_price)
        
        return {
            'stop_price': stop_price,
            'atr_value': atr_value,
            'atr_multiplier': multiplier,
            'risk_per_share': risk_amount,
            'risk_percentage': (risk_amount / entry_price) * 100
        }
    
    def _calculate_sr_stop(
        self,
        entry_price: float,
        direction: str,
        support_level: float = None,
        resistance_level: float = None,
        buffer_pct: float = 0.5,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate support/resistance-based stop loss"""
        if direction == 'BUY':
            if support_level is None:
                raise ValueError("Support level required for long position stop loss")
            
            # Place stop below support with buffer
            stop_price = support_level * (1 - buffer_pct / 100)
            reference_level = support_level
            
        else:  # SELL
            if resistance_level is None:
                raise ValueError("Resistance level required for short position stop loss")
            
            # Place stop above resistance with buffer
            stop_price = resistance_level * (1 + buffer_pct / 100)
            reference_level = resistance_level
        
        risk_amount = abs(entry_price - stop_price)
        
        return {
            'stop_price': stop_price,
            'reference_level': reference_level,
            'buffer_percentage': buffer_pct,
            'risk_per_share': risk_amount,
            'risk_percentage': (risk_amount / entry_price) * 100
        }
    
    def _calculate_volatility_stop(
        self,
        entry_price: float,
        direction: str,
        data: pd.DataFrame = None,
        volatility_period: int = 20,
        volatility_multiplier: float = 2.0,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate volatility-based stop loss"""
        if data is None:
            raise ValueError("Historical data required for volatility-based stop loss")
        
        # Calculate historical volatility
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=volatility_period).std().iloc[-1] * np.sqrt(252)
        
        # Adjust stop based on volatility
        base_stop_pct = self.config.default_stop_pct
        
        if volatility > self.config.high_volatility_threshold:
            # Increase stop for high volatility
            adjusted_stop_pct = base_stop_pct * (1 + self.config.volatility_adjustment_factor)
        elif volatility < self.config.low_volatility_threshold:
            # Decrease stop for low volatility
            adjusted_stop_pct = base_stop_pct * (1 - self.config.volatility_adjustment_factor)
        else:
            adjusted_stop_pct = base_stop_pct
        
        # Apply volatility multiplier
        volatility_stop_distance = volatility * volatility_multiplier
        
        if direction == 'BUY':
            stop_price = entry_price * (1 - max(adjusted_stop_pct / 100, volatility_stop_distance))
        else:
            stop_price = entry_price * (1 + max(adjusted_stop_pct / 100, volatility_stop_distance))
        
        risk_amount = abs(entry_price - stop_price)
        
        return {
            'stop_price': stop_price,
            'volatility': volatility,
            'adjusted_stop_percentage': adjusted_stop_pct,
            'volatility_multiplier': volatility_multiplier,
            'risk_per_share': risk_amount,
            'risk_percentage': (risk_amount / entry_price) * 100
        }
    
    def _apply_time_adjustments(self, stop_data: Dict, trade_type: str = "swing", **kwargs) -> Dict:
        """Apply time-based adjustments to stop loss"""
        original_stop = stop_data['stop_price']
        entry_price = stop_data['entry_price']
        direction = stop_data['direction']
        
        if trade_type == "intraday":
            multiplier = self.config.intraday_stop_multiplier
        elif trade_type == "swing":
            multiplier = self.config.swing_stop_multiplier
        else:
            multiplier = 1.0
        
        # Adjust stop distance
        stop_distance = abs(original_stop - entry_price)
        adjusted_distance = stop_distance * multiplier
        
        if direction == 'BUY':
            adjusted_stop = entry_price - adjusted_distance
        else:
            adjusted_stop = entry_price + adjusted_distance
        
        stop_data['stop_price'] = adjusted_stop
        stop_data['time_adjustment_factor'] = multiplier
        stop_data['trade_type'] = trade_type
        
        # Recalculate risk metrics
        stop_data['risk_per_share'] = adjusted_distance
        stop_data['risk_percentage'] = (adjusted_distance / entry_price) * 100
        
        return stop_data
    
    def _apply_gap_protection(self, stop_data: Dict) -> Dict:
        """Apply gap protection to stop loss"""
        entry_price = stop_data['entry_price']
        direction = stop_data['direction']
        max_gap_pct = self.config.max_gap_protection_pct
        
        # Calculate maximum gap protection distance
        max_gap_distance = entry_price * (max_gap_pct / 100)
        
        # Ensure stop is not too far from entry (gap protection)
        current_stop_distance = abs(stop_data['stop_price'] - entry_price)
        
        if current_stop_distance > max_gap_distance:
            logger.warning(f"Stop loss too far ({current_stop_distance:.2f}), applying gap protection")
            
            if direction == 'BUY':
                protected_stop = entry_price - max_gap_distance
            else:
                protected_stop = entry_price + max_gap_distance
            
            stop_data['stop_price'] = protected_stop
            stop_data['gap_protection_applied'] = True
            stop_data['max_gap_protection_pct'] = max_gap_pct
            
            # Recalculate risk metrics
            stop_data['risk_per_share'] = max_gap_distance
            stop_data['risk_percentage'] = max_gap_pct
        else:
            stop_data['gap_protection_applied'] = False
        
        return stop_data
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def add_stop_loss(
        self,
        symbol: str,
        stop_data: Dict,
        position_id: str = None
    ):
        """Add stop loss to tracking"""
        stop_id = position_id or f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_stops[stop_id] = {
            'symbol': symbol,
            'stop_data': stop_data,
            'created_at': datetime.now(),
            'status': 'active'
        }
        
        logger.info(f"Stop loss added for {symbol}: ₹{stop_data['stop_price']:.2f}")
        
        return stop_id
    
    def update_stop_loss(
        self,
        stop_id: str,
        new_stop_price: float,
        reason: str = "manual_update"
    ) -> bool:
        """Update existing stop loss"""
        if stop_id not in self.active_stops:
            logger.warning(f"Stop loss {stop_id} not found")
            return False
        
        old_stop = self.active_stops[stop_id]['stop_data']['stop_price']
        self.active_stops[stop_id]['stop_data']['stop_price'] = new_stop_price
        self.active_stops[stop_id]['last_updated'] = datetime.now()
        self.active_stops[stop_id]['update_reason'] = reason
        
        symbol = self.active_stops[stop_id]['symbol']
        logger.info(f"Stop loss updated for {symbol}: ₹{old_stop:.2f} → ₹{new_stop_price:.2f} ({reason})")
        
        return True
    
    def remove_stop_loss(self, stop_id: str, reason: str = "position_closed"):
        """Remove stop loss from tracking"""
        if stop_id in self.active_stops:
            self.active_stops[stop_id]['status'] = 'removed'
            self.active_stops[stop_id]['removed_at'] = datetime.now()
            self.active_stops[stop_id]['remove_reason'] = reason
            
            symbol = self.active_stops[stop_id]['symbol']
            logger.info(f"Stop loss removed for {symbol}: {reason}")
    
    def get_active_stops(self) -> Dict:
        """Get all active stop losses"""
        return {k: v for k, v in self.active_stops.items() if v['status'] == 'active'}
    
    def check_stop_hit(
        self,
        stop_id: str,
        current_price: float,
        current_time: datetime = None
    ) -> Dict[str, Union[bool, str, float]]:
        """Check if stop loss has been hit"""
        if stop_id not in self.active_stops:
            return {'hit': False, 'reason': 'stop_not_found'}
        
        stop_info = self.active_stops[stop_id]
        
        if stop_info['status'] != 'active':
            return {'hit': False, 'reason': 'stop_not_active'}
        
        stop_data = stop_info['stop_data']
        stop_price = stop_data['stop_price']
        direction = stop_data['direction']
        
        current_time = current_time or datetime.now()
        
        # Check if stop is hit
        hit = False
        if direction == 'BUY':
            hit = current_price <= stop_price
        else:  # SELL
            hit = current_price >= stop_price
        
        if hit:
            # Mark stop as hit
            self.active_stops[stop_id]['status'] = 'hit'
            self.active_stops[stop_id]['hit_at'] = current_time
            self.active_stops[stop_id]['hit_price'] = current_price
            
            symbol = stop_info['symbol']
            logger.info(f"Stop loss HIT for {symbol}: ₹{current_price:.2f} vs ₹{stop_price:.2f}")
            
            return {
                'hit': True,
                'stop_price': stop_price,
                'hit_price': current_price,
                'symbol': symbol,
                'direction': direction,
                'hit_time': current_time
            }
        
        return {'hit': False, 'reason': 'price_not_hit'}


# Utility functions
def quick_stop_loss(
    entry_price: float,
    direction: str,
    stop_pct: float = 1.5
) -> float:
    """Quick stop loss calculation"""
    direction = direction.upper()
    
    if direction == 'BUY':
        return entry_price * (1 - stop_pct / 100)
    else:
        return entry_price * (1 + stop_pct / 100)


def calculate_position_risk(
    entry_price: float,
    stop_price: float,
    quantity: int
) -> Dict[str, float]:
    """Calculate position risk metrics"""
    risk_per_share = abs(entry_price - stop_price)
    total_risk = risk_per_share * quantity
    position_value = entry_price * quantity
    risk_percentage = (total_risk / position_value) * 100
    
    return {
        'risk_per_share': risk_per_share,
        'total_risk': total_risk,
        'position_value': position_value,
        'risk_percentage': risk_percentage
    }