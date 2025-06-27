"""
Gap Handler - Overnight and Weekend Gap Management
Handles price gaps that can affect stop losses and position management
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class GapType(Enum):
    """Types of price gaps"""
    NO_GAP = "no_gap"
    OVERNIGHT = "overnight"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    EARNINGS = "earnings"
    NEWS = "news"


class GapDirection(Enum):
    """Direction of price gap relative to position"""
    FAVORABLE = "favorable"  # Gap in favor of position
    UNFAVORABLE = "unfavorable"  # Gap against position
    NEUTRAL = "neutral"


@dataclass
class GapConfig:
    """Configuration for gap handling"""
    # Gap detection settings
    min_gap_threshold_pct: float = 0.5  # Minimum 0.5% to consider a gap
    significant_gap_threshold_pct: float = 2.0  # 2%+ is significant gap
    extreme_gap_threshold_pct: float = 5.0  # 5%+ is extreme gap
    
    # Gap protection settings
    enable_gap_protection: bool = True
    max_gap_exposure_pct: float = 3.0  # Maximum gap exposure
    
    # Position size adjustments for gap risk - reduced aggressiveness
    overnight_position_reduction: float = 0.9  # Reduce overnight positions by 10%
    weekend_position_reduction: float = 0.8  # Reduce weekend positions by 20%
    earnings_position_reduction: float = 0.7  # Reduce positions before earnings by 30%
    
    # Stop loss adjustments
    gap_stop_buffer_pct: float = 1.0  # Additional buffer for gap protection
    max_gap_stop_distance_pct: float = 4.0  # Maximum stop distance for gap protection
    
    # Time-based settings
    market_open_time: str = "09:15"  # NSE opening time
    market_close_time: str = "15:30"  # NSE closing time
    
    # Risk management
    max_gap_loss_pct: float = 2.0  # Maximum loss allowed due to gaps
    enable_pre_market_monitoring: bool = True
    
    # Weekend and holiday handling
    reduce_friday_positions: bool = True
    friday_reduction_factor: float = 0.9  # Reduce Friday positions by 10%


@dataclass
class GapEvent:
    """Represents a detected gap event"""
    symbol: str
    gap_type: GapType
    gap_direction: GapDirection
    
    # Price information
    previous_close: float
    current_open: float
    gap_size_points: float
    gap_size_percentage: float
    
    # Timing
    previous_session_end: datetime
    current_session_start: datetime
    gap_duration_hours: float
    
    # Impact assessment
    severity: str  # 'minor', 'moderate', 'significant', 'extreme'
    estimated_impact: float  # Estimated impact on position
    
    # Market context
    market_sentiment: Optional[str] = None
    volume_context: Optional[str] = None
    news_events: List[str] = None


class GapHandler:
    """
    Advanced gap handling for overnight and weekend price movements
    
    Features:
    - Gap detection and classification
    - Position size adjustments for gap risk
    - Stop loss modifications for gap protection
    - Pre-market risk assessment
    - Historical gap analysis
    """
    
    def __init__(self, config: GapConfig = None):
        """Initialize gap handler"""
        self.config = config or GapConfig()
        
        # Track gap events and impacts
        self.gap_history = []
        self.position_adjustments = {}
        
        # Market hours for gap detection
        self.market_hours = {
            'open': self.config.market_open_time,
            'close': self.config.market_close_time
        }
        
        logger.info(f"Gap handler initialized with {self.config.min_gap_threshold_pct}% minimum gap threshold")
    
    def detect_gap(
        self,
        symbol: str,
        previous_close: float,
        current_open: float,
        previous_session_end: datetime,
        current_session_start: datetime,
        volume_data: Dict = None,
        news_events: List[str] = None
    ) -> Optional[GapEvent]:
        """
        Detect and classify price gaps
        
        Args:
            symbol: Trading symbol
            previous_close: Previous session closing price
            current_open: Current session opening price
            previous_session_end: Previous session end time
            current_session_start: Current session start time
            volume_data: Optional volume information
            news_events: Optional news events list
            
        Returns:
            GapEvent if gap detected, None otherwise
        """
        # Calculate gap metrics
        gap_points = current_open - previous_close
        gap_percentage = (gap_points / previous_close) * 100
        
        # Check if gap meets minimum threshold
        if abs(gap_percentage) < self.config.min_gap_threshold_pct:
            return None
        
        # Determine gap type based on time duration
        gap_duration = (current_session_start - previous_session_end).total_seconds() / 3600
        gap_type = self._classify_gap_type(gap_duration, previous_session_end, current_session_start)
        
        # Determine gap direction (we'll set this when checking against positions)
        gap_direction = GapDirection.NEUTRAL
        
        # Assess gap severity
        severity = self._assess_gap_severity(abs(gap_percentage))
        
        gap_event = GapEvent(
            symbol=symbol,
            gap_type=gap_type,
            gap_direction=gap_direction,
            previous_close=previous_close,
            current_open=current_open,
            gap_size_points=gap_points,
            gap_size_percentage=gap_percentage,
            previous_session_end=previous_session_end,
            current_session_start=current_session_start,
            gap_duration_hours=gap_duration,
            severity=severity,
            estimated_impact=0.0,  # Will be calculated when checking positions
            news_events=news_events or []
        )
        
        # Add market context
        gap_event = self._add_market_context(gap_event, volume_data)
        
        # Store in history
        self.gap_history.append(gap_event)
        
        logger.info(f"Gap detected: {symbol} - {gap_percentage:.2f}% ({gap_type.value}, {severity})")
        
        return gap_event
    
    def assess_gap_impact_on_position(
        self,
        gap_event: GapEvent,
        position_entry_price: float,
        position_quantity: int,
        position_direction: str,
        current_stop_price: float
    ) -> Dict[str, Union[float, str, bool]]:
        """
        Assess gap impact on a specific position
        
        Args:
            gap_event: Detected gap event
            position_entry_price: Position entry price
            position_quantity: Position quantity
            position_direction: 'BUY' or 'SELL'
            current_stop_price: Current stop loss price
            
        Returns:
            Dictionary with gap impact analysis
        """
        position_direction = position_direction.upper()
        
        # Determine if gap is favorable or unfavorable for position
        if position_direction == 'BUY':
            # For long positions, positive gaps are favorable
            if gap_event.gap_size_points > 0:
                gap_direction = GapDirection.FAVORABLE
            else:
                gap_direction = GapDirection.UNFAVORABLE
        else:
            # For short positions, negative gaps are favorable
            if gap_event.gap_size_points < 0:
                gap_direction = GapDirection.FAVORABLE
            else:
                gap_direction = GapDirection.UNFAVORABLE
        
        # Update gap event with direction
        gap_event.gap_direction = gap_direction
        
        # Calculate potential P&L impact
        position_value = position_entry_price * position_quantity
        
        if position_direction == 'BUY':
            gap_pnl = gap_event.gap_size_points * position_quantity
        else:
            gap_pnl = -gap_event.gap_size_points * position_quantity
        
        gap_impact_pct = (gap_pnl / position_value) * 100
        
        # Check if gap would hit stop loss
        stop_hit = False
        if position_direction == 'BUY':
            stop_hit = gap_event.current_open <= current_stop_price
        else:
            stop_hit = gap_event.current_open >= current_stop_price
        
        # Calculate gap slippage (difference between stop and gap open)
        gap_slippage = 0.0
        if stop_hit:
            gap_slippage = abs(gap_event.current_open - current_stop_price)
        
        # Assess risk level
        risk_level = self._assess_position_risk_level(
            gap_impact_pct, gap_event.severity, gap_direction
        )
        
        return {
            'gap_direction': gap_direction.value,
            'gap_pnl': gap_pnl,
            'gap_impact_percentage': gap_impact_pct,
            'stop_hit': stop_hit,
            'gap_slippage': gap_slippage,
            'gap_slippage_pct': (gap_slippage / position_entry_price) * 100 if gap_slippage > 0 else 0,
            'risk_level': risk_level,
            'position_value': position_value,
            'requires_action': risk_level in ['high', 'extreme'] or stop_hit
        }
    
    def calculate_gap_adjusted_position_size(
        self,
        base_position_size: int,
        symbol: str,
        entry_time: datetime,
        position_direction: str,
        earnings_date: datetime = None,
        news_risk_level: str = "normal"
    ) -> Dict[str, Union[int, float, str]]:
        """
        Calculate position size adjusted for gap risk
        
        Args:
            base_position_size: Original calculated position size
            symbol: Trading symbol
            entry_time: Planned entry time
            position_direction: 'BUY' or 'SELL'
            earnings_date: Upcoming earnings date if any
            news_risk_level: 'low', 'normal', 'high'
            
        Returns:
            Dictionary with adjusted position size and reasoning
        """
        adjustments = []
        adjustment_factor = 1.0
        
        # Check for overnight risk
        market_close = self._get_next_market_close(entry_time)
        hours_to_close = (market_close - entry_time).total_seconds() / 3600
        
        if hours_to_close < 24:  # Will hold overnight
            adjustment_factor *= self.config.overnight_position_reduction
            adjustments.append(f"Overnight reduction: {(1-self.config.overnight_position_reduction)*100:.0f}%")
        
        # Check for weekend risk
        if entry_time.weekday() == 4:  # Friday
            if self.config.reduce_friday_positions:
                adjustment_factor *= self.config.friday_reduction_factor
                adjustments.append(f"Friday reduction: {(1-self.config.friday_reduction_factor)*100:.0f}%")
        
        # Check for earnings risk
        if earnings_date:
            days_to_earnings = (earnings_date - entry_time).days
            if 0 <= days_to_earnings <= 7:  # Within a week of earnings
                adjustment_factor *= self.config.earnings_position_reduction
                adjustments.append(f"Earnings reduction: {(1-self.config.earnings_position_reduction)*100:.0f}%")
        
        # News risk adjustment
        if news_risk_level == "high":
            adjustment_factor *= 0.7  # 30% reduction for high news risk
            adjustments.append("High news risk reduction: 30%")
        elif news_risk_level == "low":
            adjustment_factor *= 1.1  # 10% increase for low news risk
            adjustments.append("Low news risk increase: 10%")
        
        # Calculate historical gap risk for this symbol
        historical_gap_risk = self._calculate_historical_gap_risk(symbol)
        if historical_gap_risk > 0.02:  # High historical gap risk
            adjustment_factor *= 0.9  # Reduced from 0.8 to 0.9
            adjustments.append(f"Historical gap risk reduction: 10%")
        
        # Apply adjustments with minimum position size protection
        adjusted_size = max(1, int(base_position_size * adjustment_factor))
        total_reduction_pct = (1 - adjustment_factor) * 100
        
        return {
            'original_size': base_position_size,
            'adjusted_size': adjusted_size,
            'adjustment_factor': adjustment_factor,
            'total_reduction_pct': total_reduction_pct,
            'adjustments': adjustments,
            'gap_risk_level': self._classify_overall_gap_risk(adjustment_factor)
        }
    
    def calculate_gap_protected_stop_loss(
        self,
        entry_price: float,
        normal_stop_price: float,
        position_direction: str,
        symbol: str = None
    ) -> Dict[str, Union[float, str]]:
        """
        Calculate stop loss with gap protection
        
        Args:
            entry_price: Position entry price
            normal_stop_price: Normal calculated stop loss
            position_direction: 'BUY' or 'SELL'
            symbol: Trading symbol for historical analysis
            
        Returns:
            Dictionary with gap-protected stop loss
        """
        position_direction = position_direction.upper()
        
        # Calculate normal stop distance
        normal_stop_distance = abs(entry_price - normal_stop_price)
        normal_stop_pct = (normal_stop_distance / entry_price) * 100
        
        # Get historical gap statistics for this symbol
        avg_gap_size = self._get_average_gap_size(symbol) if symbol else 2.0
        max_recent_gap = self._get_max_recent_gap(symbol) if symbol else 3.0
        
        # Calculate gap buffer
        gap_buffer_pct = max(
            self.config.gap_stop_buffer_pct,
            avg_gap_size * 0.5,  # Half of average gap
            max_recent_gap * 0.3  # 30% of max recent gap
        )
        
        # Ensure total stop distance doesn't exceed maximum
        total_stop_pct = normal_stop_pct + gap_buffer_pct
        if total_stop_pct > self.config.max_gap_stop_distance_pct:
            gap_buffer_pct = self.config.max_gap_stop_distance_pct - normal_stop_pct
            total_stop_pct = self.config.max_gap_stop_distance_pct
        
        # Calculate gap-protected stop price
        if position_direction == 'BUY':
            gap_protected_stop = entry_price * (1 - total_stop_pct / 100)
        else:
            gap_protected_stop = entry_price * (1 + total_stop_pct / 100)
        
        protection_level = "normal"
        if gap_buffer_pct > 2.0:
            protection_level = "high"
        elif gap_buffer_pct > 1.0:
            protection_level = "moderate"
        
        return {
            'normal_stop_price': normal_stop_price,
            'gap_protected_stop': gap_protected_stop,
            'normal_stop_pct': normal_stop_pct,
            'gap_buffer_pct': gap_buffer_pct,
            'total_stop_pct': total_stop_pct,
            'additional_risk': (gap_protected_stop - normal_stop_price) if position_direction == 'BUY' else (normal_stop_price - gap_protected_stop),
            'protection_level': protection_level,
            'historical_avg_gap': avg_gap_size,
            'max_recent_gap': max_recent_gap
        }
    
    def _classify_gap_type(
        self,
        gap_duration_hours: float,
        previous_end: datetime,
        current_start: datetime
    ) -> GapType:
        """Classify the type of gap based on duration and timing"""
        if gap_duration_hours < 20:
            return GapType.OVERNIGHT
        elif gap_duration_hours > 60:  # More than 2.5 days
            return GapType.HOLIDAY
        elif previous_end.weekday() == 4:  # Friday
            return GapType.WEEKEND
        else:
            return GapType.OVERNIGHT
    
    def _assess_gap_severity(self, gap_percentage: float) -> str:
        """Assess gap severity based on percentage"""
        if gap_percentage >= self.config.extreme_gap_threshold_pct:
            return "extreme"
        elif gap_percentage >= self.config.significant_gap_threshold_pct:
            return "significant"
        elif gap_percentage >= self.config.min_gap_threshold_pct * 2:
            return "moderate"
        else:
            return "minor"
    
    def _add_market_context(self, gap_event: GapEvent, volume_data: Dict = None) -> GapEvent:
        """Add market context to gap event"""
        # Volume context
        if volume_data:
            if volume_data.get('relative_volume', 1.0) > 2.0:
                gap_event.volume_context = "high_volume"
            elif volume_data.get('relative_volume', 1.0) < 0.5:
                gap_event.volume_context = "low_volume"
            else:
                gap_event.volume_context = "normal_volume"
        
        # Market sentiment (simplified)
        if gap_event.gap_size_percentage > 1.0:
            gap_event.market_sentiment = "bullish"
        elif gap_event.gap_size_percentage < -1.0:
            gap_event.market_sentiment = "bearish"
        else:
            gap_event.market_sentiment = "neutral"
        
        return gap_event
    
    def _assess_position_risk_level(
        self,
        gap_impact_pct: float,
        gap_severity: str,
        gap_direction: GapDirection
    ) -> str:
        """Assess risk level for position given gap"""
        if gap_direction == GapDirection.FAVORABLE:
            return "low"  # Favorable gaps are low risk
        
        # For unfavorable gaps
        abs_impact = abs(gap_impact_pct)
        
        if gap_severity == "extreme" or abs_impact > 5.0:
            return "extreme"
        elif gap_severity == "significant" or abs_impact > 3.0:
            return "high"
        elif gap_severity == "moderate" or abs_impact > 1.5:
            return "moderate"
        else:
            return "low"
    
    def _calculate_historical_gap_risk(self, symbol: str) -> float:
        """Calculate historical gap risk for a symbol"""
        if not symbol:
            return 0.02  # Default 2% gap risk
        
        # Get historical gaps for this symbol
        symbol_gaps = [gap for gap in self.gap_history if gap.symbol == symbol]
        
        if len(symbol_gaps) < 5:
            return 0.02  # Default if insufficient history
        
        # Calculate average absolute gap size
        avg_gap_size = np.mean([abs(gap.gap_size_percentage) for gap in symbol_gaps])
        
        return avg_gap_size / 100  # Convert to decimal
    
    def _get_average_gap_size(self, symbol: str) -> float:
        """Get average gap size for symbol"""
        if not symbol:
            return 2.0
        
        symbol_gaps = [gap for gap in self.gap_history if gap.symbol == symbol]
        
        if not symbol_gaps:
            return 2.0
        
        return np.mean([abs(gap.gap_size_percentage) for gap in symbol_gaps])
    
    def _get_max_recent_gap(self, symbol: str, days: int = 30) -> float:
        """Get maximum gap in recent period"""
        if not symbol:
            return 3.0
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_gaps = [
            gap for gap in self.gap_history 
            if gap.symbol == symbol and gap.current_session_start >= cutoff_date
        ]
        
        if not recent_gaps:
            return 3.0
        
        return max([abs(gap.gap_size_percentage) for gap in recent_gaps])
    
    def _get_next_market_close(self, from_time: datetime) -> datetime:
        """Get next market close time"""
        # Simplified - assumes same day close at 15:30
        close_time = from_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        if from_time.time() > close_time.time():
            # Market already closed, next close is tomorrow
            close_time += timedelta(days=1)
        
        return close_time
    
    def _classify_overall_gap_risk(self, adjustment_factor: float) -> str:
        """Classify overall gap risk level"""
        if adjustment_factor < 0.5:
            return "extreme"
        elif adjustment_factor < 0.7:
            return "high"
        elif adjustment_factor < 0.9:
            return "moderate"
        else:
            return "low"
    
    def get_gap_statistics(self, symbol: str = None, days: int = 90) -> Dict:
        """Get gap statistics for analysis"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        if symbol:
            gaps = [gap for gap in self.gap_history 
                   if gap.symbol == symbol and gap.current_session_start >= cutoff_date]
        else:
            gaps = [gap for gap in self.gap_history 
                   if gap.current_session_start >= cutoff_date]
        
        if not gaps:
            return {'total_gaps': 0}
        
        gap_sizes = [abs(gap.gap_size_percentage) for gap in gaps]
        
        return {
            'total_gaps': len(gaps),
            'average_gap_size': np.mean(gap_sizes),
            'median_gap_size': np.median(gap_sizes),
            'max_gap_size': max(gap_sizes),
            'std_gap_size': np.std(gap_sizes),
            'gaps_by_type': {gap_type.value: sum(1 for gap in gaps if gap.gap_type == gap_type) 
                           for gap_type in GapType},
            'gaps_by_severity': {severity: sum(1 for gap in gaps if gap.severity == severity) 
                               for severity in ['minor', 'moderate', 'significant', 'extreme']}
        }


# Utility functions
def is_market_hours(timestamp: datetime, market_open: str = "09:15", market_close: str = "15:30") -> bool:
    """Check if timestamp is during market hours"""
    time_str = timestamp.strftime("%H:%M")
    return market_open <= time_str <= market_close


def calculate_gap_exposure(
    positions: List[Dict],
    gap_events: List[GapEvent]
) -> Dict[str, float]:
    """Calculate total gap exposure across positions"""
    total_exposure = 0.0
    symbol_exposures = {}
    
    for position in positions:
        symbol = position['symbol']
        position_value = position.get('position_value', 0)
        
        # Find relevant gap events for this symbol
        symbol_gaps = [gap for gap in gap_events if gap.symbol == symbol]
        
        if symbol_gaps:
            # Use largest recent gap as exposure measure
            max_gap_pct = max([abs(gap.gap_size_percentage) for gap in symbol_gaps])
            exposure = position_value * (max_gap_pct / 100)
            
            symbol_exposures[symbol] = exposure
            total_exposure += exposure
    
    return {
        'total_exposure': total_exposure,
        'symbol_exposures': symbol_exposures
    }