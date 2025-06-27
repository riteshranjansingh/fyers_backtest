"""
Advanced Trailing Stop Management
Implements sophisticated trailing stop logic with profit booking and risk management
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TrailingStage(Enum):
    """Stages of trailing stop progression"""
    INITIAL = "initial"
    BREAKEVEN = "breakeven"
    PARTIAL_BOOK = "partial_book"
    FULL_TRAIL = "full_trail"
    FINAL = "final"


@dataclass
class TrailingConfig:
    """Configuration for advanced trailing stop logic"""
    # Core trailing settings
    enable_trailing: bool = True
    
    # Risk-Reward based triggers
    breakeven_trigger_rr: float = 1.0  # Move to breakeven at 1:1 R:R
    breakeven_buffer_points: float = 2.0  # Breakeven + 2 points
    
    partial_book_trigger_rr: float = 2.0  # Book partial profits at 1:2 R:R
    partial_book_percentage: float = 50.0  # Book 50% of position
    
    full_trail_trigger_rr: float = 2.5  # Start aggressive trailing at 2.5:1 R:R
    
    # Trailing parameters
    trailing_stop_distance_pct: float = 1.0  # Trail by 1% from highs/lows
    min_trailing_distance: float = 5.0  # Minimum trailing distance in points
    max_trailing_distance_pct: float = 3.0  # Maximum trailing distance
    
    # ATR-based trailing
    use_atr_trailing: bool = True
    atr_trailing_multiplier: float = 1.5  # Trail by 1.5x ATR
    atr_period: int = 14
    
    # Time-based settings
    min_time_before_trail: int = 5  # Minimum minutes before trailing starts
    max_trail_speed: float = 0.5  # Maximum trail adjustment per update (%)
    
    # Advanced features
    acceleration_factor: float = 0.02  # Parabolic SAR style acceleration
    max_acceleration: float = 0.2
    
    # Profit protection
    min_profit_protection_pct: float = 0.5  # Always protect minimum profit
    profit_lock_percentage: float = 80.0  # Lock in 80% of unrealized profits
    
    # Market condition adjustments
    volatile_market_multiplier: float = 1.5  # Wider stops in volatile markets
    trending_market_multiplier: float = 0.8  # Tighter stops in trending markets
    
    def validate(self) -> bool:
        """Validate trailing configuration"""
        if self.breakeven_trigger_rr <= 0 or self.partial_book_trigger_rr <= 0:
            raise ValueError("R:R triggers must be positive")
        
        if not (10 <= self.partial_book_percentage <= 90):
            raise ValueError("Partial book percentage must be between 10% and 90%")
        
        if self.partial_book_trigger_rr <= self.breakeven_trigger_rr:
            raise ValueError("Partial book trigger must be greater than breakeven trigger")
        
        return True


@dataclass
class TrailingState:
    """Current state of a trailing stop"""
    symbol: str
    position_id: str
    
    # Position details
    entry_price: float
    initial_stop_price: float
    current_quantity: int
    initial_quantity: int
    direction: str  # 'BUY' or 'SELL'
    
    # Current trailing state
    current_stop_price: float
    highest_favorable_price: float  # Highest price for long, lowest for short
    stage: TrailingStage = TrailingStage.INITIAL
    
    # Profit booking tracking
    partial_booked_quantity: int = 0
    partial_book_price: float = 0.0
    partial_book_time: Optional[datetime] = None
    
    # Trailing metrics
    current_rr_ratio: float = 0.0
    unrealized_profit: float = 0.0
    total_realized_profit: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Advanced tracking
    acceleration_factor: float = 0.02
    atr_value: float = 0.0
    trailing_history: List[Dict] = field(default_factory=list)


class AdvancedTrailingStop:
    """
    Advanced trailing stop management with multiple stages and profit booking
    
    Features:
    - Risk-reward based stage progression
    - Automatic partial profit booking
    - ATR and percentage-based trailing
    - Acceleration-based tightening
    - Market condition awareness
    """
    
    def __init__(self, config: TrailingConfig = None):
        """Initialize advanced trailing stop manager"""
        self.config = config or TrailingConfig()
        self.config.validate()
        
        # Track all trailing positions
        self.trailing_positions = {}
        
        # Performance metrics
        self.total_positions = 0
        self.successful_trails = 0
        self.profit_bookings = 0
        
        logger.info(f"Advanced trailing stop initialized: BE@{self.config.breakeven_trigger_rr}:1, Partial@{self.config.partial_book_trigger_rr}:1")
    
    def add_trailing_position(
        self,
        symbol: str,
        entry_price: float,
        initial_stop_price: float,
        quantity: int,
        direction: str,
        position_id: str = None,
        current_price: float = None
    ) -> str:
        """
        Add new position to trailing stop management
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price of the position
            initial_stop_price: Initial stop loss price
            quantity: Position quantity
            direction: 'BUY' or 'SELL'
            position_id: Optional position identifier
            current_price: Current market price
            
        Returns:
            Position ID for tracking
        """
        direction = direction.upper()
        if direction not in ['BUY', 'SELL']:
            raise ValueError("Direction must be 'BUY' or 'SELL'")
        
        # Generate position ID if not provided
        if position_id is None:
            position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize current price
        if current_price is None:
            current_price = entry_price
        
        # Create trailing state
        trailing_state = TrailingState(
            symbol=symbol,
            position_id=position_id,
            entry_price=entry_price,
            initial_stop_price=initial_stop_price,
            current_quantity=quantity,
            initial_quantity=quantity,
            direction=direction,
            current_stop_price=initial_stop_price,
            highest_favorable_price=current_price
        )
        
        # Calculate initial metrics
        trailing_state = self._update_metrics(trailing_state, current_price)
        
        # Store position
        self.trailing_positions[position_id] = trailing_state
        self.total_positions += 1
        
        logger.info(f"Trailing position added: {symbol} - {quantity} @ ₹{entry_price:.2f}, Stop: ₹{initial_stop_price:.2f}")
        
        return position_id
    
    def update_trailing_stop(
        self,
        position_id: str,
        current_price: float,
        current_time: datetime = None,
        atr_value: float = None
    ) -> Dict[str, Union[bool, float, str]]:
        """
        Update trailing stop based on current market conditions
        
        Args:
            position_id: Position identifier
            current_price: Current market price
            current_time: Current timestamp
            atr_value: Current ATR value for dynamic trailing
            
        Returns:
            Dictionary with update results and any actions required
        """
        if position_id not in self.trailing_positions:
            return {'success': False, 'error': 'Position not found'}
        
        current_time = current_time or datetime.now()
        trailing_state = self.trailing_positions[position_id]
        
        try:
            # Update price tracking
            previous_stage = trailing_state.stage
            trailing_state = self._update_price_tracking(trailing_state, current_price)
            
            # Update ATR if provided
            if atr_value:
                trailing_state.atr_value = atr_value
            
            # Update metrics
            trailing_state = self._update_metrics(trailing_state, current_price)
            
            # Process stage transitions
            actions = self._process_stage_transitions(trailing_state, current_price, current_time)
            
            # Update trailing stop price
            new_stop = self._calculate_new_trailing_stop(trailing_state, current_price)
            
            # Apply trailing stop update
            if new_stop != trailing_state.current_stop_price:
                trailing_state = self._update_stop_price(trailing_state, new_stop, current_time)
                actions['stop_updated'] = True
                actions['new_stop_price'] = new_stop
            else:
                actions['stop_updated'] = False
            
            # Record trailing history
            trailing_state.trailing_history.append({
                'timestamp': current_time,
                'price': current_price,
                'stop_price': trailing_state.current_stop_price,
                'stage': trailing_state.stage.value,
                'rr_ratio': trailing_state.current_rr_ratio,
                'actions': actions.copy()
            })
            
            # Update timestamp
            trailing_state.last_updated = current_time
            
            # Check for stage change
            if trailing_state.stage != previous_stage:
                actions['stage_changed'] = True
                actions['new_stage'] = trailing_state.stage.value
                actions['previous_stage'] = previous_stage.value
            
            # Store updated state
            self.trailing_positions[position_id] = trailing_state
            
            return {
                'success': True,
                'actions': actions,
                'current_stop': trailing_state.current_stop_price,
                'current_stage': trailing_state.stage.value,
                'rr_ratio': trailing_state.current_rr_ratio,
                'unrealized_profit': trailing_state.unrealized_profit
            }
            
        except Exception as e:
            logger.error(f"Error updating trailing stop for {position_id}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _update_price_tracking(self, state: TrailingState, current_price: float) -> TrailingState:
        """Update highest/lowest favorable price tracking"""
        if state.direction == 'BUY':
            # For long positions, track highest price
            if current_price > state.highest_favorable_price:
                state.highest_favorable_price = current_price
        else:
            # For short positions, track lowest price
            if current_price < state.highest_favorable_price:
                state.highest_favorable_price = current_price
        
        return state
    
    def _update_metrics(self, state: TrailingState, current_price: float) -> TrailingState:
        """Update position metrics"""
        # Calculate current R:R ratio
        initial_risk = abs(state.entry_price - state.initial_stop_price)
        
        if state.direction == 'BUY':
            current_profit = max(0, current_price - state.entry_price)
        else:
            current_profit = max(0, state.entry_price - current_price)
        
        if initial_risk > 0:
            state.current_rr_ratio = current_profit / initial_risk
        else:
            state.current_rr_ratio = 0
        
        # Calculate unrealized profit
        state.unrealized_profit = current_profit * state.current_quantity
        
        return state
    
    def _process_stage_transitions(
        self,
        state: TrailingState,
        current_price: float,
        current_time: datetime
    ) -> Dict[str, Union[bool, float]]:
        """Process transitions between trailing stages"""
        actions = {}
        
        # Stage 1: Move to breakeven
        if (state.stage == TrailingStage.INITIAL and 
            state.current_rr_ratio >= self.config.breakeven_trigger_rr):
            
            state.stage = TrailingStage.BREAKEVEN
            actions['moved_to_breakeven'] = True
            actions['breakeven_rr'] = state.current_rr_ratio
            
            logger.info(f"Position {state.symbol} moved to breakeven stage at {state.current_rr_ratio:.2f}:1 R:R")
        
        # Stage 2: Partial profit booking
        if (state.stage in [TrailingStage.INITIAL, TrailingStage.BREAKEVEN] and
            state.current_rr_ratio >= self.config.partial_book_trigger_rr and
            state.partial_booked_quantity == 0):
            
            # Calculate partial booking
            book_quantity = int(state.initial_quantity * (self.config.partial_book_percentage / 100))
            
            if book_quantity > 0:
                state.stage = TrailingStage.PARTIAL_BOOK
                state.partial_booked_quantity = book_quantity
                state.current_quantity -= book_quantity
                state.partial_book_price = current_price
                state.partial_book_time = current_time
                
                # Calculate realized profit
                if state.direction == 'BUY':
                    partial_profit = (current_price - state.entry_price) * book_quantity
                else:
                    partial_profit = (state.entry_price - current_price) * book_quantity
                
                state.total_realized_profit += partial_profit
                
                actions['partial_booking'] = True
                actions['book_quantity'] = book_quantity
                actions['book_price'] = current_price
                actions['partial_profit'] = partial_profit
                
                self.profit_bookings += 1
                
                logger.info(f"Partial booking: {state.symbol} - {book_quantity} shares @ ₹{current_price:.2f}, Profit: ₹{partial_profit:.2f}")
        
        # Stage 3: Full trailing mode
        if (state.stage in [TrailingStage.PARTIAL_BOOK] and
            state.current_rr_ratio >= self.config.full_trail_trigger_rr):
            
            state.stage = TrailingStage.FULL_TRAIL
            actions['full_trail_mode'] = True
            
            logger.info(f"Position {state.symbol} entered full trailing mode at {state.current_rr_ratio:.2f}:1 R:R")
        
        return actions
    
    def _calculate_new_trailing_stop(self, state: TrailingState, current_price: float) -> float:
        """Calculate new trailing stop price based on current stage"""
        current_stop = state.current_stop_price
        
        if state.stage == TrailingStage.INITIAL:
            # No trailing in initial stage
            return current_stop
        
        elif state.stage == TrailingStage.BREAKEVEN:
            # Move to breakeven + buffer
            breakeven_stop = state.entry_price
            
            if state.direction == 'BUY':
                breakeven_stop += self.config.breakeven_buffer_points
                return max(current_stop, breakeven_stop)  # Only move up
            else:
                breakeven_stop -= self.config.breakeven_buffer_points
                return min(current_stop, breakeven_stop)  # Only move down
        
        elif state.stage in [TrailingStage.PARTIAL_BOOK, TrailingStage.FULL_TRAIL]:
            # Active trailing
            return self._calculate_active_trailing_stop(state, current_price)
        
        return current_stop
    
    def _calculate_active_trailing_stop(self, state: TrailingState, current_price: float) -> float:
        """Calculate trailing stop for active trailing stages"""
        if self.config.use_atr_trailing and state.atr_value > 0:
            # ATR-based trailing
            trailing_distance = state.atr_value * self.config.atr_trailing_multiplier
        else:
            # Percentage-based trailing
            trailing_distance = state.highest_favorable_price * (self.config.trailing_stop_distance_pct / 100)
            
            # Apply min/max limits
            trailing_distance = max(trailing_distance, self.config.min_trailing_distance)
            max_distance = state.highest_favorable_price * (self.config.max_trailing_distance_pct / 100)
            trailing_distance = min(trailing_distance, max_distance)
        
        # Apply acceleration factor for aggressive trailing
        if state.stage == TrailingStage.FULL_TRAIL:
            acceleration = min(
                state.acceleration_factor * state.current_rr_ratio,
                self.config.max_acceleration
            )
            trailing_distance *= (1 - acceleration)
        
        # Calculate new stop
        if state.direction == 'BUY':
            new_stop = state.highest_favorable_price - trailing_distance
            return max(state.current_stop_price, new_stop)  # Only move up
        else:
            new_stop = state.highest_favorable_price + trailing_distance
            return min(state.current_stop_price, new_stop)  # Only move down
    
    def _update_stop_price(
        self,
        state: TrailingState,
        new_stop_price: float,
        current_time: datetime
    ) -> TrailingState:
        """Update stop price and log the change"""
        old_stop = state.current_stop_price
        state.current_stop_price = new_stop_price
        
        # Update acceleration factor (Parabolic SAR style)
        if state.stage == TrailingStage.FULL_TRAIL:
            state.acceleration_factor = min(
                state.acceleration_factor + self.config.acceleration_factor,
                self.config.max_acceleration
            )
        
        logger.debug(f"Trailing stop updated for {state.symbol}: ₹{old_stop:.2f} → ₹{new_stop_price:.2f}")
        
        return state
    
    def check_stop_hit(
        self,
        position_id: str,
        current_price: float,
        current_time: datetime = None
    ) -> Dict[str, Union[bool, str, float]]:
        """Check if trailing stop has been hit"""
        if position_id not in self.trailing_positions:
            return {'hit': False, 'error': 'Position not found'}
        
        state = self.trailing_positions[position_id]
        current_time = current_time or datetime.now()
        
        # Check if stop is hit
        stop_hit = False
        if state.direction == 'BUY':
            stop_hit = current_price <= state.current_stop_price
        else:
            stop_hit = current_price >= state.current_stop_price
        
        if stop_hit:
            # Calculate final profit/loss
            if state.direction == 'BUY':
                remaining_pnl = (current_price - state.entry_price) * state.current_quantity
            else:
                remaining_pnl = (state.entry_price - current_price) * state.current_quantity
            
            total_pnl = state.total_realized_profit + remaining_pnl
            
            # Mark position as stopped out
            state.stage = TrailingStage.FINAL
            
            self.successful_trails += 1
            
            logger.info(f"Trailing stop HIT for {state.symbol}: ₹{current_price:.2f} vs ₹{state.current_stop_price:.2f}")
            logger.info(f"Final P&L: ₹{total_pnl:.2f} (Realized: ₹{state.total_realized_profit:.2f}, Remaining: ₹{remaining_pnl:.2f})")
            
            return {
                'hit': True,
                'symbol': state.symbol,
                'stop_price': state.current_stop_price,
                'hit_price': current_price,
                'remaining_quantity': state.current_quantity,
                'total_realized_profit': state.total_realized_profit,
                'remaining_pnl': remaining_pnl,
                'total_pnl': total_pnl,
                'final_rr_ratio': state.current_rr_ratio,
                'hit_time': current_time
            }
        
        return {'hit': False}
    
    def get_position_status(self, position_id: str) -> Dict:
        """Get comprehensive status of a trailing position"""
        if position_id not in self.trailing_positions:
            return {'exists': False}
        
        state = self.trailing_positions[position_id]
        
        return {
            'exists': True,
            'symbol': state.symbol,
            'stage': state.stage.value,
            'entry_price': state.entry_price,
            'current_stop': state.current_stop_price,
            'highest_favorable': state.highest_favorable_price,
            'current_quantity': state.current_quantity,
            'initial_quantity': state.initial_quantity,
            'partial_booked': state.partial_booked_quantity,
            'rr_ratio': state.current_rr_ratio,
            'unrealized_profit': state.unrealized_profit,
            'realized_profit': state.total_realized_profit,
            'direction': state.direction,
            'created_at': state.created_at,
            'last_updated': state.last_updated
        }
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get status of all trailing positions"""
        return {pos_id: self.get_position_status(pos_id) 
                for pos_id in self.trailing_positions.keys()}
    
    def get_performance_metrics(self) -> Dict[str, Union[int, float]]:
        """Get trailing stop performance metrics"""
        active_positions = sum(1 for state in self.trailing_positions.values() 
                              if state.stage != TrailingStage.FINAL)
        
        total_realized_profit = sum(state.total_realized_profit 
                                   for state in self.trailing_positions.values())
        
        return {
            'total_positions': self.total_positions,
            'active_positions': active_positions,
            'completed_positions': self.successful_trails,
            'profit_bookings': self.profit_bookings,
            'total_realized_profit': total_realized_profit,
            'success_rate': (self.successful_trails / max(1, self.total_positions)) * 100
        }
    
    def remove_position(self, position_id: str):
        """Remove position from tracking"""
        if position_id in self.trailing_positions:
            del self.trailing_positions[position_id]
            logger.info(f"Trailing position removed: {position_id}")


# Utility functions
def calculate_risk_reward_levels(
    entry_price: float,
    stop_loss_price: float,
    direction: str,
    target_rr_ratios: List[float] = [1.0, 2.0, 3.0]
) -> Dict[float, float]:
    """Calculate price levels for different R:R ratios"""
    direction = direction.upper()
    initial_risk = abs(entry_price - stop_loss_price)
    
    levels = {}
    for rr_ratio in target_rr_ratios:
        if direction == 'BUY':
            target_price = entry_price + (initial_risk * rr_ratio)
        else:
            target_price = entry_price - (initial_risk * rr_ratio)
        
        levels[rr_ratio] = target_price
    
    return levels