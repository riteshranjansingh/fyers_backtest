"""
Trade Execution Simulator - Realistic trade execution with slippage and market impact
Simulates real-world trading conditions for accurate backtesting
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging

from ..strategies.base_strategy import Signal
from ..risk.risk_integration import TradeRecommendation

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for trade execution simulation"""
    enable_slippage: bool = True
    slippage_bps: float = 2.0  # 2 basis points default slippage
    min_slippage: float = 0.10  # Minimum ₹0.10 slippage
    max_slippage_pct: float = 0.5  # Maximum 0.5% slippage
    
    # Market impact modeling
    enable_market_impact: bool = False
    impact_factor: float = 0.1  # Impact per ₹1L position size
    
    # Execution timing
    execution_delay_bars: int = 0  # Execute on same bar (0) or next bar (1)
    
    # Liquidity constraints
    max_volume_participation: float = 0.1  # Max 10% of bar volume
    min_volume_for_execution: int = 1000  # Minimum volume required


class TradeExecutor:
    """
    Simulates realistic trade execution with slippage, market impact,
    and liquidity constraints for accurate backtesting
    """
    
    def __init__(self, config: ExecutionConfig = None, **kwargs):
        """
        Initialize trade executor
        
        Args:
            config: Execution configuration
            **kwargs: Configuration overrides
        """
        self.config = config or ExecutionConfig()
        
        # Apply any keyword overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.execution_history = []
        
        logger.info(f"Trade executor initialized with {self.config.slippage_bps:.1f}bps slippage")
    
    def execute_trade(
        self,
        signal: Signal,
        recommendation: TradeRecommendation,
        market_data: pd.Series,
        volume_data: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Execute a trade with realistic market conditions
        
        Args:
            signal: Original trading signal
            recommendation: Risk management recommendation
            market_data: Current market data (OHLCV)
            volume_data: Volume data for liquidity checks
            
        Returns:
            Execution result with actual fill price and quantities
        """
        try:
            # Extract market information
            current_price = signal.price
            volume = market_data.get('volume', 1000000)  # Default 1M volume
            high = market_data.get('high', current_price * 1.002)
            low = market_data.get('low', current_price * 0.998)
            
            # Check liquidity constraints
            liquidity_check = self._check_liquidity(
                requested_quantity=recommendation.recommended_quantity,
                current_volume=volume,
                current_price=current_price
            )
            
            if not liquidity_check['executable']:
                return {
                    'success': False,
                    'error': f"Liquidity constraint: {liquidity_check['reason']}",
                    'liquidity_info': liquidity_check
                }
            
            # Calculate execution price with slippage
            execution_price = self._calculate_execution_price(
                signal_price=current_price,
                signal_type=signal.signal_type,
                quantity=recommendation.recommended_quantity,
                high=high,
                low=low,
                volume=volume
            )
            
            # Validate execution price is within reasonable bounds
            price_check = self._validate_execution_price(
                signal_price=current_price,
                execution_price=execution_price,
                signal_type=signal.signal_type,
                high=high,
                low=low
            )
            
            if not price_check['valid']:
                return {
                    'success': False,
                    'error': f"Price validation failed: {price_check['reason']}",
                    'price_info': price_check
                }
            
            # Calculate final quantities (may be adjusted for liquidity)
            final_quantity = liquidity_check.get('adjusted_quantity', recommendation.recommended_quantity)
            
            # Record execution
            execution_record = {
                'timestamp': signal.timestamp,
                'symbol': recommendation.original_signal.metadata.get('symbol', 'UNKNOWN'),
                'signal_type': signal.signal_type,
                'signal_price': current_price,
                'execution_price': execution_price,
                'requested_quantity': recommendation.recommended_quantity,
                'executed_quantity': final_quantity,
                'slippage': execution_price - current_price if signal.signal_type == 'BUY' else current_price - execution_price,
                'slippage_bps': abs(execution_price - current_price) / current_price * 10000,
                'volume': volume,
                'recommendation': recommendation
            }
            
            self.execution_history.append(execution_record)
            
            # Log execution
            slippage_bps = execution_record['slippage_bps']
            logger.debug(f"Trade executed: {signal.signal_type} {final_quantity}@₹{execution_price:.2f} (slippage: {slippage_bps:.1f}bps)")
            
            return {
                'success': True,
                'execution_price': execution_price,
                'quantity': final_quantity,
                'slippage': execution_record['slippage'],
                'slippage_bps': slippage_bps,
                'execution_record': execution_record,
                'liquidity_info': liquidity_check
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _check_liquidity(
        self,
        requested_quantity: int,
        current_volume: int,
        current_price: float
    ) -> Dict[str, Any]:
        """Check if trade can be executed given liquidity constraints"""
        
        # Check minimum volume requirement
        if current_volume < self.config.min_volume_for_execution:
            return {
                'executable': False,
                'reason': f"Insufficient volume: {current_volume} < {self.config.min_volume_for_execution}",
                'volume_available': current_volume,
                'volume_required': self.config.min_volume_for_execution
            }
        
        # Check volume participation limit
        max_allowed_quantity = int(current_volume * self.config.max_volume_participation)
        
        if requested_quantity > max_allowed_quantity:
            return {
                'executable': True,
                'adjusted_quantity': max_allowed_quantity,
                'original_quantity': requested_quantity,
                'reason': f"Quantity reduced due to volume limit: {requested_quantity} -> {max_allowed_quantity}",
                'volume_participation': max_allowed_quantity / current_volume * 100
            }
        
        return {
            'executable': True,
            'adjusted_quantity': requested_quantity,
            'volume_participation': requested_quantity / current_volume * 100,
            'liquidity_adequate': True
        }
    
    def _calculate_execution_price(
        self,
        signal_price: float,
        signal_type: str,
        quantity: int,
        high: float,
        low: float,
        volume: int
    ) -> float:
        """Calculate realistic execution price with slippage"""
        
        base_slippage = 0.0
        
        # Calculate base slippage
        if self.config.enable_slippage:
            # Base slippage in basis points
            base_slippage_bps = self.config.slippage_bps
            
            # Adjust for quantity (larger orders get more slippage)
            quantity_factor = 1.0 + (quantity / 10000.0)  # Extra slippage for large orders
            
            # Adjust for volume (low volume = higher slippage)
            volume_factor = max(0.5, min(2.0, 1000000 / max(volume, 100000)))
            
            # Calculate final slippage in rupees
            slippage_bps = base_slippage_bps * quantity_factor * volume_factor
            base_slippage = signal_price * (slippage_bps / 10000.0)
            
            # Apply minimum and maximum slippage limits
            base_slippage = max(self.config.min_slippage, base_slippage)
            max_slippage = signal_price * (self.config.max_slippage_pct / 100.0)
            base_slippage = min(base_slippage, max_slippage)
        
        # Calculate market impact if enabled
        market_impact = 0.0
        if self.config.enable_market_impact:
            position_value = quantity * signal_price
            impact_units = position_value / 100000  # Per ₹1L
            market_impact = impact_units * self.config.impact_factor
        
        # Combine slippage and impact
        total_slippage = base_slippage + market_impact
        
        # Apply slippage based on trade direction
        if signal_type == 'BUY':
            execution_price = signal_price + total_slippage
            # Ensure we don't exceed the high
            execution_price = min(execution_price, high)
        else:  # SELL
            execution_price = signal_price - total_slippage
            # Ensure we don't go below the low
            execution_price = max(execution_price, low)
        
        # Add some randomness for realism (±10% of slippage)
        if total_slippage > 0:
            randomness = np.random.uniform(-0.1, 0.1) * total_slippage
            if signal_type == 'BUY':
                execution_price += randomness
                execution_price = max(signal_price, min(execution_price, high))
            else:
                execution_price -= randomness
                execution_price = min(signal_price, max(execution_price, low))
        
        return round(execution_price, 2)
    
    def _validate_execution_price(
        self,
        signal_price: float,
        execution_price: float,
        signal_type: str,
        high: float,
        low: float
    ) -> Dict[str, Any]:
        """Validate that execution price is reasonable"""
        
        # Check if price is within the bar's range
        if execution_price > high or execution_price < low:
            return {
                'valid': False,
                'reason': f"Execution price ₹{execution_price:.2f} outside bar range [₹{low:.2f}, ₹{high:.2f}]",
                'execution_price': execution_price,
                'bar_range': (low, high)
            }
        
        # Check slippage is not excessive
        slippage = abs(execution_price - signal_price)
        slippage_pct = (slippage / signal_price) * 100
        
        if slippage_pct > self.config.max_slippage_pct:
            return {
                'valid': False,
                'reason': f"Excessive slippage: {slippage_pct:.2f}% > {self.config.max_slippage_pct}%",
                'slippage_pct': slippage_pct,
                'max_allowed': self.config.max_slippage_pct
            }
        
        # Check direction makes sense
        if signal_type == 'BUY' and execution_price < signal_price - 0.01:
            return {
                'valid': False,
                'reason': f"BUY execution price ₹{execution_price:.2f} below signal price ₹{signal_price:.2f}",
                'signal_price': signal_price,
                'execution_price': execution_price
            }
        
        if signal_type == 'SELL' and execution_price > signal_price + 0.01:
            return {
                'valid': False,
                'reason': f"SELL execution price ₹{execution_price:.2f} above signal price ₹{signal_price:.2f}",
                'signal_price': signal_price,
                'execution_price': execution_price
            }
        
        return {
            'valid': True,
            'slippage_pct': slippage_pct,
            'slippage_amount': slippage
        }
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about trade executions"""
        if not self.execution_history:
            return {'total_executions': 0}
        
        executions_df = pd.DataFrame(self.execution_history)
        
        return {
            'total_executions': len(executions_df),
            'avg_slippage_bps': executions_df['slippage_bps'].mean(),
            'max_slippage_bps': executions_df['slippage_bps'].max(),
            'min_slippage_bps': executions_df['slippage_bps'].min(),
            'total_slippage_cost': executions_df['slippage'].sum(),
            'avg_execution_size': executions_df['executed_quantity'].mean(),
            'liquidity_adjustments': len(executions_df[executions_df['executed_quantity'] != executions_df['requested_quantity']]),
            'buy_orders': len(executions_df[executions_df['signal_type'] == 'BUY']),
            'sell_orders': len(executions_df[executions_df['signal_type'] == 'SELL'])
        }
    
    def reset_history(self):
        """Reset execution history"""
        self.execution_history = []
        logger.info("Execution history reset")


# Utility functions for different execution models

def create_conservative_executor() -> TradeExecutor:
    """Create executor with conservative (higher) slippage assumptions"""
    config = ExecutionConfig(
        enable_slippage=True,
        slippage_bps=5.0,  # 5 bps
        min_slippage=0.25,
        max_slippage_pct=1.0,
        enable_market_impact=True,
        impact_factor=0.2
    )
    return TradeExecutor(config)


def create_optimistic_executor() -> TradeExecutor:
    """Create executor with optimistic (lower) slippage assumptions"""
    config = ExecutionConfig(
        enable_slippage=True,
        slippage_bps=1.0,  # 1 bp
        min_slippage=0.05,
        max_slippage_pct=0.25,
        enable_market_impact=False
    )
    return TradeExecutor(config)


def create_realistic_executor() -> TradeExecutor:
    """Create executor with realistic slippage for Indian markets"""
    config = ExecutionConfig(
        enable_slippage=True,
        slippage_bps=2.5,  # 2.5 bps
        min_slippage=0.10,
        max_slippage_pct=0.5,
        enable_market_impact=True,
        impact_factor=0.1,
        max_volume_participation=0.05  # 5% of volume
    )
    return TradeExecutor(config)