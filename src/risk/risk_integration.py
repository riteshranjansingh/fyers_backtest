"""
Risk Management Integration Module
Integrates position sizing, stop losses, trailing stops, and gap handling
with strategy signals for complete risk management
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from .position_sizer import PositionSizer, RiskConfig
from .stop_loss import StopLossManager, StopLossConfig
from .advanced_trailing import AdvancedTrailingStop, TrailingConfig
from .gap_handler import GapHandler, GapConfig
from ..strategies.base_strategy import Signal, StrategyResult

logger = logging.getLogger(__name__)


@dataclass
class RiskManagerConfig:
    """Complete risk management configuration"""
    # Component configurations
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    stop_loss_config: StopLossConfig = field(default_factory=StopLossConfig)
    trailing_config: TrailingConfig = field(default_factory=TrailingConfig)
    gap_config: GapConfig = field(default_factory=GapConfig)
    
    # Integration settings
    enable_position_sizing: bool = True
    enable_stop_loss_management: bool = True
    enable_trailing_stops: bool = True
    enable_gap_protection: bool = True
    
    # Risk profile
    risk_profile: str = "moderate"  # conservative, moderate, aggressive
    
    # Portfolio-level settings
    max_portfolio_risk: float = 10.0  # Maximum 10% portfolio risk
    max_correlated_positions: int = 3  # Max positions in correlated assets
    
    # Emergency settings
    emergency_stop_loss_pct: float = 5.0  # Emergency stop at 5% account loss
    max_daily_loss_pct: float = 3.0  # Maximum daily loss


@dataclass
class TradeRecommendation:
    """Complete trade recommendation with risk management"""
    # Original signal
    original_signal: Signal
    
    # Position sizing
    recommended_quantity: int
    position_value: float
    risk_amount: float
    risk_percentage: float
    
    # Stop loss management
    initial_stop_price: float
    stop_loss_method: str
    
    # Trailing stop settings
    enable_trailing: bool
    trailing_config: TrailingConfig
    
    # Gap protection
    gap_adjusted_size: int
    gap_protection_applied: bool
    
    # Risk assessment
    overall_risk_score: float  # 0-100 scale
    risk_warnings: List[str] = field(default_factory=list)
    
    # Execution details
    trade_valid: bool = True
    rejection_reason: str = ""
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)


class RiskManager:
    """
    Comprehensive risk management system that integrates:
    - Position sizing
    - Stop loss management  
    - Advanced trailing stops
    - Gap handling
    - Portfolio-level risk management
    """
    
    def __init__(self, config: RiskManagerConfig = None):
        """Initialize comprehensive risk manager"""
        self.config = config or RiskManagerConfig()
        
        # Initialize component managers
        self.position_sizer = PositionSizer(self.config.risk_config)
        self.stop_loss_manager = StopLossManager(self.config.stop_loss_config)
        self.trailing_stop_manager = AdvancedTrailingStop(self.config.trailing_config)
        self.gap_handler = GapHandler(self.config.gap_config)
        
        # Track active trades and portfolio
        self.active_trades = {}
        self.portfolio_metrics = {
            'total_risk': 0.0,
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
        
        logger.info(f"Risk manager initialized with {self.config.risk_profile} profile")
    
    def process_strategy_signal(
        self,
        signal: Signal,
        market_data: pd.DataFrame,
        symbol: str,
        current_price: float = None,
        market_context: Dict = None
    ) -> TradeRecommendation:
        """
        Process strategy signal and generate complete trade recommendation
        
        Args:
            signal: Strategy signal from MA crossover or other strategy
            market_data: Historical market data for calculations
            symbol: Trading symbol
            current_price: Current market price
            market_context: Additional market context (volatility, news, etc.)
            
        Returns:
            Complete trade recommendation with risk management
        """
        try:
            # Use signal price if current price not provided
            if current_price is None:
                current_price = signal.price
            
            # Initialize recommendation
            recommendation = TradeRecommendation(
                original_signal=signal,
                recommended_quantity=0,
                position_value=0,
                risk_amount=0,
                risk_percentage=0,
                initial_stop_price=0,
                stop_loss_method="",
                enable_trailing=False,
                trailing_config=self.config.trailing_config,
                gap_adjusted_size=0,
                gap_protection_applied=False,
                overall_risk_score=0,
                risk_warnings=[]
            )
            
            # Step 1: Calculate initial stop loss
            stop_result = self._calculate_initial_stop_loss(
                signal, current_price, market_data, market_context
            )
            
            if not stop_result['valid']:
                recommendation.trade_valid = False
                recommendation.rejection_reason = f"Stop loss calculation failed: {stop_result.get('error', 'Unknown error')}"
                return recommendation
            
            recommendation.initial_stop_price = stop_result['stop_price']
            recommendation.stop_loss_method = stop_result['method']
            
            # Step 2: Calculate position size
            position_result = self._calculate_position_size(
                signal, current_price, recommendation.initial_stop_price, symbol
            )
            
            if not position_result['valid']:
                recommendation.trade_valid = False
                recommendation.rejection_reason = f"Position sizing failed: {position_result.get('error', 'Unknown error')}"
                return recommendation
            
            recommendation.recommended_quantity = position_result['quantity']
            recommendation.position_value = position_result['position_value']
            recommendation.risk_amount = position_result['risk_amount']
            recommendation.risk_percentage = position_result['risk_percentage']
            
            # Step 3: Apply gap protection
            gap_result = self._apply_gap_protection(
                recommendation.recommended_quantity, symbol, signal, market_context
            )
            
            recommendation.gap_adjusted_size = gap_result['adjusted_size']
            recommendation.gap_protection_applied = gap_result['protection_applied']
            
            # Update final quantities
            final_quantity = recommendation.gap_adjusted_size
            recommendation.recommended_quantity = final_quantity
            recommendation.position_value = final_quantity * current_price
            recommendation.risk_amount = final_quantity * abs(current_price - recommendation.initial_stop_price)
            
            # Step 4: Configure trailing stops
            trailing_result = self._configure_trailing_stops(
                signal, recommendation, market_data
            )
            
            recommendation.enable_trailing = trailing_result['enable']
            recommendation.trailing_config = trailing_result['config']
            
            # Step 5: Portfolio-level risk checks
            portfolio_check = self._check_portfolio_risk(recommendation, symbol)
            
            if not portfolio_check['approved']:
                recommendation.trade_valid = False
                recommendation.rejection_reason = portfolio_check['reason']
                return recommendation
            
            # Step 6: Calculate overall risk score
            recommendation.overall_risk_score = self._calculate_risk_score(recommendation, market_context)
            
            # Step 7: Generate risk warnings
            recommendation.risk_warnings = self._generate_risk_warnings(recommendation, market_context)
            
            # Final validation
            if recommendation.recommended_quantity == 0:
                recommendation.trade_valid = False
                recommendation.rejection_reason = "Position size calculated as zero"
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {str(e)}")
            recommendation.trade_valid = False
            recommendation.rejection_reason = f"Processing error: {str(e)}"
            return recommendation
    
    def execute_trade_recommendation(
        self,
        recommendation: TradeRecommendation,
        execution_price: float = None
    ) -> Dict[str, Union[bool, str, Dict]]:
        """
        Execute trade recommendation and set up risk management
        
        Args:
            recommendation: Trade recommendation to execute
            execution_price: Actual execution price
            
        Returns:
            Execution result with risk management setup
        """
        if not recommendation.trade_valid:
            return {
                'success': False,
                'error': f"Invalid recommendation: {recommendation.rejection_reason}"
            }
        
        execution_price = execution_price or recommendation.original_signal.price
        symbol = recommendation.original_signal.metadata.get('symbol', 'UNKNOWN')
        
        try:
            # Generate unique trade ID
            trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Add position to position sizer tracking
            self.position_sizer.add_position(
                symbol=symbol,
                quantity=recommendation.recommended_quantity,
                entry_price=execution_price,
                sector=recommendation.original_signal.metadata.get('sector'),
                position_type='LONG' if recommendation.original_signal.signal_type == 'BUY' else 'SHORT'
            )
            
            # Set up stop loss management
            stop_data = self.stop_loss_manager.calculate_initial_stop_loss(
                entry_price=execution_price,
                direction=recommendation.original_signal.signal_type,
                method=recommendation.stop_loss_method
            )
            
            stop_id = self.stop_loss_manager.add_stop_loss(symbol, stop_data, trade_id)
            
            # Set up trailing stops if enabled
            trailing_id = None
            if recommendation.enable_trailing:
                trailing_id = self.trailing_stop_manager.add_trailing_position(
                    symbol=symbol,
                    entry_price=execution_price,
                    initial_stop_price=recommendation.initial_stop_price,
                    quantity=recommendation.recommended_quantity,
                    direction=recommendation.original_signal.signal_type,
                    position_id=trade_id,
                    current_price=execution_price
                )
            
            # Store active trade
            self.active_trades[trade_id] = {
                'symbol': symbol,
                'recommendation': recommendation,
                'execution_price': execution_price,
                'stop_id': stop_id,
                'trailing_id': trailing_id,
                'entry_time': datetime.now(),
                'status': 'active'
            }
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            logger.info(f"Trade executed: {symbol} - {recommendation.recommended_quantity} @ ₹{execution_price:.2f}")
            
            return {
                'success': True,
                'trade_id': trade_id,
                'stop_id': stop_id,
                'trailing_id': trailing_id,
                'execution_details': {
                    'symbol': symbol,
                    'quantity': recommendation.recommended_quantity,
                    'price': execution_price,
                    'stop_price': recommendation.initial_stop_price,
                    'risk_amount': recommendation.risk_amount,
                    'trailing_enabled': recommendation.enable_trailing
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing trade recommendation: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def update_positions(
        self,
        market_data: Dict[str, float],
        current_time: datetime = None
    ) -> Dict[str, List[Dict]]:
        """
        Update all active positions with current market data
        
        Args:
            market_data: Dictionary of symbol -> current_price
            current_time: Current timestamp
            
        Returns:
            Dictionary with update results and any triggered actions
        """
        current_time = current_time or datetime.now()
        
        updates = {
            'stop_hits': [],
            'trailing_updates': [],
            'partial_bookings': [],
            'warnings': []
        }
        
        for trade_id, trade_info in self.active_trades.items():
            if trade_info['status'] != 'active':
                continue
            
            symbol = trade_info['symbol']
            current_price = market_data.get(symbol)
            
            if current_price is None:
                continue
            
            try:
                # Update stop loss
                if trade_info['stop_id']:
                    stop_hit = self.stop_loss_manager.check_stop_hit(
                        trade_info['stop_id'], current_price, current_time
                    )
                    
                    if stop_hit['hit']:
                        updates['stop_hits'].append({
                            'trade_id': trade_id,
                            'symbol': symbol,
                            'stop_hit': stop_hit
                        })
                        
                        # Mark trade as stopped out
                        trade_info['status'] = 'stopped'
                        trade_info['exit_time'] = current_time
                        trade_info['exit_price'] = stop_hit['hit_price']
                        
                        continue  # Don't update trailing if stopped out
                
                # Update trailing stops
                if trade_info['trailing_id']:
                    trailing_result = self.trailing_stop_manager.update_trailing_stop(
                        trade_info['trailing_id'], current_price, current_time
                    )
                    
                    if trailing_result['success']:
                        updates['trailing_updates'].append({
                            'trade_id': trade_id,
                            'symbol': symbol,
                            'update': trailing_result
                        })
                        
                        # Check for partial bookings
                        if trailing_result.get('actions', {}).get('partial_booking'):
                            updates['partial_bookings'].append({
                                'trade_id': trade_id,
                                'symbol': symbol,
                                'booking_details': trailing_result['actions']
                            })
                
            except Exception as e:
                logger.error(f"Error updating position {trade_id}: {str(e)}")
                updates['warnings'].append({
                    'trade_id': trade_id,
                    'symbol': symbol,
                    'error': str(e)
                })
        
        # Update portfolio metrics
        self._update_portfolio_metrics()
        
        return updates
    
    def _calculate_initial_stop_loss(
        self,
        signal: Signal,
        current_price: float,
        market_data: pd.DataFrame,
        market_context: Dict = None
    ) -> Dict[str, Union[bool, float, str]]:
        """Calculate initial stop loss for the signal"""
        try:
            # Determine stop loss method based on available data and market conditions
            if market_context and 'atr' in market_context:
                method = "atr"
                kwargs = {'atr_value': market_context['atr']}
            elif market_context and 'volatility' in market_context:
                method = "volatility"
                kwargs = {'data': market_data}
            else:
                method = "percentage"
                kwargs = {}
            
            # Calculate stop loss
            stop_result = self.stop_loss_manager.calculate_initial_stop_loss(
                entry_price=current_price,
                direction=signal.signal_type,
                method=method,
                **kwargs
            )
            
            return stop_result
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        stop_price: float,
        symbol: str
    ) -> Dict[str, Union[bool, int, float]]:
        """Calculate position size based on risk management"""
        try:
            position_result = self.position_sizer.calculate_position_size(
                entry_price=current_price,
                stop_loss_price=stop_price,
                symbol=symbol
            )
            
            if 'error' in position_result:
                return {'valid': False, 'error': position_result['error']}
            
            position_result['valid'] = True
            return position_result
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _apply_gap_protection(
        self,
        base_quantity: int,
        symbol: str,
        signal: Signal,
        market_context: Dict = None
    ) -> Dict[str, Union[int, bool]]:
        """Apply gap protection to position size"""
        try:
            gap_result = self.gap_handler.calculate_gap_adjusted_position_size(
                base_position_size=base_quantity,
                symbol=symbol,
                entry_time=signal.timestamp,
                position_direction=signal.signal_type,
                earnings_date=market_context.get('earnings_date') if market_context else None,
                news_risk_level=market_context.get('news_risk', 'normal') if market_context else 'normal'
            )
            
            return {
                'adjusted_size': gap_result['adjusted_size'],
                'protection_applied': gap_result['adjusted_size'] != base_quantity,
                'gap_result': gap_result
            }
            
        except Exception as e:
            logger.warning(f"Gap protection failed for {symbol}: {str(e)}")
            return {
                'adjusted_size': base_quantity,
                'protection_applied': False
            }
    
    def _configure_trailing_stops(
        self,
        signal: Signal,
        recommendation: TradeRecommendation,
        market_data: pd.DataFrame
    ) -> Dict[str, Union[bool, TrailingConfig]]:
        """Configure trailing stops for the position"""
        # Enable trailing if configured and position size is adequate
        enable_trailing = (
            self.config.enable_trailing_stops and
            recommendation.recommended_quantity > 0 and
            recommendation.risk_percentage <= 3.0  # Don't trail very risky positions
        )
        
        # Customize trailing config based on signal confidence and market conditions
        trailing_config = self.config.trailing_config
        
        if enable_trailing:
            # Adjust trailing parameters based on signal strength
            if signal.confidence > 0.8:
                # High confidence signals get tighter trailing
                trailing_config.trailing_stop_distance_pct *= 0.8
            elif signal.confidence < 0.5:
                # Low confidence signals get wider trailing
                trailing_config.trailing_stop_distance_pct *= 1.2
        
        return {
            'enable': enable_trailing,
            'config': trailing_config
        }
    
    def _check_portfolio_risk(self, recommendation: TradeRecommendation, symbol: str) -> Dict[str, Union[bool, str]]:
        """Check portfolio-level risk constraints"""
        # Calculate total portfolio risk
        current_risk = sum(
            trade['recommendation'].risk_percentage 
            for trade in self.active_trades.values() 
            if trade['status'] == 'active'
        )
        
        total_risk = current_risk + recommendation.risk_percentage
        
        if total_risk > self.config.max_portfolio_risk:
            return {
                'approved': False,
                'reason': f"Portfolio risk limit exceeded: {total_risk:.1f}% > {self.config.max_portfolio_risk}%"
            }
        
        # Check correlation limits (simplified)
        same_sector_positions = sum(
            1 for trade in self.active_trades.values()
            if trade['status'] == 'active' and 
            trade['recommendation'].original_signal.metadata.get('sector') == 
            recommendation.original_signal.metadata.get('sector')
        )
        
        if same_sector_positions >= self.config.max_correlated_positions:
            return {
                'approved': False,
                'reason': f"Too many correlated positions: {same_sector_positions} >= {self.config.max_correlated_positions}"
            }
        
        return {'approved': True}
    
    def _calculate_risk_score(self, recommendation: TradeRecommendation, market_context: Dict = None) -> float:
        """Calculate overall risk score (0-100)"""
        score = 0
        
        # Position size risk (0-30 points)
        if recommendation.risk_percentage <= 1.0:
            score += 5  # Very low risk
        elif recommendation.risk_percentage <= 2.0:
            score += 15  # Low risk
        elif recommendation.risk_percentage <= 3.0:
            score += 25  # Moderate risk
        else:
            score += 30  # High risk
        
        # Signal confidence (0-20 points)
        confidence_score = (1 - recommendation.original_signal.confidence) * 20
        score += confidence_score
        
        # Gap protection (0-15 points)
        if recommendation.gap_protection_applied:
            score += 5  # Lower risk with gap protection
        else:
            score += 15  # Higher risk without gap protection
        
        # Market volatility (0-20 points)
        if market_context and 'volatility' in market_context:
            vol = market_context['volatility']
            if vol > 0.03:  # High volatility
                score += 20
            elif vol > 0.02:  # Moderate volatility
                score += 10
            else:  # Low volatility
                score += 5
        else:
            score += 10  # Default moderate risk
        
        # Portfolio concentration (0-15 points)
        active_positions = len([t for t in self.active_trades.values() if t['status'] == 'active'])
        if active_positions >= 4:
            score += 15  # High concentration risk
        elif active_positions >= 2:
            score += 8   # Moderate concentration
        else:
            score += 3   # Low concentration
        
        return min(score, 100)  # Cap at 100
    
    def _generate_risk_warnings(self, recommendation: TradeRecommendation, market_context: Dict = None) -> List[str]:
        """Generate risk warnings for the recommendation"""
        warnings = []
        
        if recommendation.risk_percentage > 2.5:
            warnings.append(f"High position risk: {recommendation.risk_percentage:.1f}%")
        
        if recommendation.original_signal.confidence < 0.6:
            warnings.append(f"Low signal confidence: {recommendation.original_signal.confidence:.2f}")
        
        if not recommendation.gap_protection_applied and recommendation.gap_adjusted_size < recommendation.recommended_quantity:
            warnings.append("Position size reduced due to gap risk")
        
        if recommendation.overall_risk_score > 70:
            warnings.append("High overall risk score")
        
        if market_context:
            if market_context.get('volatility', 0) > 0.03:
                warnings.append("High market volatility detected")
            
            if market_context.get('news_risk') == 'high':
                warnings.append("High news risk environment")
        
        return warnings
    
    def _update_portfolio_metrics(self):
        """Update portfolio-level metrics"""
        active_trades = [t for t in self.active_trades.values() if t['status'] == 'active']
        
        # Calculate total risk
        total_risk = sum(t['recommendation'].risk_percentage for t in active_trades)
        
        # Calculate current P&L (simplified)
        # In real implementation, this would use current market prices
        
        self.portfolio_metrics.update({
            'total_risk': total_risk,
            'active_positions': len(active_trades),
            'total_position_value': sum(t['recommendation'].position_value for t in active_trades)
        })
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        return {
            'portfolio_metrics': self.portfolio_metrics,
            'position_sizer_summary': self.position_sizer.get_portfolio_summary(),
            'active_stops': len(self.stop_loss_manager.get_active_stops()),
            'trailing_positions': len(self.trailing_stop_manager.get_all_positions()),
            'risk_manager_config': {
                'risk_profile': self.config.risk_profile,
                'max_portfolio_risk': self.config.max_portfolio_risk,
                'components_enabled': {
                    'position_sizing': self.config.enable_position_sizing,
                    'stop_loss': self.config.enable_stop_loss_management,
                    'trailing_stops': self.config.enable_trailing_stops,
                    'gap_protection': self.config.enable_gap_protection
                }
            }
        }


# Utility functions
def create_risk_manager(
    account_balance: float,
    risk_percentage: float = 2.0,
    risk_profile: str = "moderate"
) -> RiskManager:
    """Create a pre-configured risk manager"""
    from . import RISK_PROFILES
    
    profile_settings = RISK_PROFILES.get(risk_profile, RISK_PROFILES['moderate'])
    
    # Create risk config
    risk_config = RiskConfig(
        account_balance=account_balance,
        risk_percentage=profile_settings['risk_percentage'],
        max_position_size=profile_settings['max_position_size']
    )
    
    # Create manager config
    manager_config = RiskManagerConfig(
        risk_config=risk_config,
        risk_profile=risk_profile
    )
    
    return RiskManager(manager_config)


def quick_trade_analysis(
    signal: Signal,
    current_price: float,
    account_balance: float
) -> Dict:
    """Quick trade analysis without full risk manager setup"""
    # Create temporary risk manager
    risk_manager = create_risk_manager(account_balance)
    
    # Create minimal market data
    market_data = pd.DataFrame({
        'close': [current_price] * 20,
        'high': [current_price * 1.01] * 20,
        'low': [current_price * 0.99] * 20,
        'volume': [1000000] * 20
    }, index=pd.date_range('2024-01-01', periods=20, freq='D'))
    
    # Process signal
    recommendation = risk_manager.process_strategy_signal(
        signal=signal,
        market_data=market_data,
        symbol='TEST',
        current_price=current_price
    )
    
    return {
        'valid': recommendation.trade_valid,
        'quantity': recommendation.recommended_quantity,
        'position_value': recommendation.position_value,
        'risk_amount': recommendation.risk_amount,
        'risk_percentage': recommendation.risk_percentage,
        'stop_price': recommendation.initial_stop_price,
        'risk_score': recommendation.overall_risk_score,
        'warnings': recommendation.risk_warnings
    }