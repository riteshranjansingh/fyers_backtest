"""
Position Sizer - 1-3% Risk Per Trade Calculator
Calculates optimal position sizes based on account risk management
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Configuration for risk management parameters"""
    # Core risk settings
    account_balance: float = 100000.0  # Account size in rupees
    risk_percentage: float = 2.0  # Risk per trade (1-3%)
    max_position_size: float = 0.1  # Max 10% of account per position
    
    # Stop loss settings
    default_stop_loss_pct: float = 1.5  # Default 1.5% stop loss
    min_stop_loss_pct: float = 0.5  # Minimum 0.5% stop loss
    max_stop_loss_pct: float = 5.0  # Maximum 5% stop loss
    
    # Position limits
    max_concurrent_positions: int = 5  # Maximum open positions
    sector_concentration_limit: float = 0.3  # Max 30% in one sector
    
    # Currency and costs
    currency: str = "INR"
    brokerage_per_trade: float = 20.0  # Flat ₹20 per trade
    taxes_and_charges_pct: float = 0.1  # 0.1% for taxes/charges
    
    def validate(self) -> bool:
        """Validate risk configuration"""
        if not (1.0 <= self.risk_percentage <= 5.0):
            raise ValueError("Risk percentage must be between 1% and 5%")
        
        if not (0.01 <= self.max_position_size <= 0.5):
            raise ValueError("Max position size must be between 1% and 50%")
        
        if self.account_balance <= 0:
            raise ValueError("Account balance must be positive")
        
        return True


class PositionSizer:
    """
    Advanced position sizing calculator for risk-based trading
    
    Calculates position sizes based on:
    - Account risk percentage (1-3%)
    - Stop loss distance
    - Account balance
    - Transaction costs
    """
    
    def __init__(self, risk_config: RiskConfig):
        """Initialize position sizer with risk configuration"""
        self.config = risk_config
        self.config.validate()
        
        # Track current positions for risk management
        self.current_positions = {}
        self.sector_exposure = {}
        
        logger.info(f"Position sizer initialized: {self.config.risk_percentage}% risk, ₹{self.config.account_balance:,.0f} account")
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        symbol: str = None,
        sector: str = None,
        custom_risk_pct: float = None
    ) -> Dict[str, Union[float, int]]:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            symbol: Symbol being traded (for tracking)
            sector: Sector classification (for concentration limits)
            custom_risk_pct: Override default risk percentage
            
        Returns:
            Dictionary with position sizing details
        """
        try:
            # Validate inputs
            if entry_price <= 0 or stop_loss_price <= 0:
                raise ValueError("Entry and stop loss prices must be positive")
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss_price)
            
            if risk_per_share == 0:
                raise ValueError("Stop loss must be different from entry price")
            
            # Determine risk percentage to use
            risk_pct = custom_risk_pct or self.config.risk_percentage
            
            # Calculate base position size
            risk_amount = self.config.account_balance * (risk_pct / 100)
            base_quantity = int(risk_amount / risk_per_share)
            
            # Calculate transaction costs
            transaction_costs = self._calculate_transaction_costs(entry_price, base_quantity)
            
            # Adjust for transaction costs
            adjusted_risk_amount = risk_amount - transaction_costs
            adjusted_quantity = max(1, int(adjusted_risk_amount / risk_per_share))
            
            # Apply position size limits
            max_quantity_by_balance = int(
                (self.config.account_balance * self.config.max_position_size) / entry_price
            )
            
            final_quantity = min(adjusted_quantity, max_quantity_by_balance)
            
            # Calculate actual position value and risk
            position_value = final_quantity * entry_price
            actual_risk_amount = final_quantity * risk_per_share
            actual_risk_pct = (actual_risk_amount / self.config.account_balance) * 100
            
            # Check concentration limits
            concentration_warning = self._check_concentration_limits(
                position_value, sector
            )
            
            result = {
                'quantity': final_quantity,
                'position_value': position_value,
                'risk_amount': actual_risk_amount,
                'risk_percentage': actual_risk_pct,
                'risk_per_share': risk_per_share,
                'transaction_costs': transaction_costs,
                'max_loss': actual_risk_amount + transaction_costs,
                'position_size_pct': (position_value / self.config.account_balance) * 100,
                'concentration_warning': concentration_warning,
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'stop_loss_pct': (risk_per_share / entry_price) * 100
            }
            
            logger.debug(f"Position size calculated: {final_quantity} shares, ₹{position_value:,.0f} value, {actual_risk_pct:.2f}% risk")
            
            return result
            
        except Exception as e:
            logger.error(f"Position sizing error: {str(e)}")
            return self._get_error_result(str(e))
    
    def calculate_stop_loss_price(
        self,
        entry_price: float,
        direction: str,
        stop_loss_pct: float = None,
        atr_value: float = None,
        support_resistance: float = None
    ) -> Dict[str, float]:
        """
        Calculate stop loss price using multiple methods
        
        Args:
            entry_price: Entry price for the trade
            direction: 'BUY' or 'SELL'
            stop_loss_pct: Percentage-based stop loss
            atr_value: ATR value for volatility-based stop
            support_resistance: Support/resistance level
            
        Returns:
            Dictionary with different stop loss calculations
        """
        direction = direction.upper()
        if direction not in ['BUY', 'SELL']:
            raise ValueError("Direction must be 'BUY' or 'SELL'")
        
        results = {'entry_price': entry_price, 'direction': direction}
        
        # Percentage-based stop loss
        stop_pct = stop_loss_pct or self.config.default_stop_loss_pct
        stop_pct = max(self.config.min_stop_loss_pct, min(stop_pct, self.config.max_stop_loss_pct))
        
        if direction == 'BUY':
            results['percentage_stop'] = entry_price * (1 - stop_pct / 100)
        else:
            results['percentage_stop'] = entry_price * (1 + stop_pct / 100)
        
        results['stop_loss_pct'] = stop_pct
        
        # ATR-based stop loss (if provided)
        if atr_value:
            atr_multiplier = 2.0  # Standard 2x ATR
            if direction == 'BUY':
                results['atr_stop'] = entry_price - (atr_value * atr_multiplier)
            else:
                results['atr_stop'] = entry_price + (atr_value * atr_multiplier)
            
            results['atr_value'] = atr_value
            results['atr_multiplier'] = atr_multiplier
        
        # Support/Resistance-based stop loss
        if support_resistance:
            if direction == 'BUY':
                # For long positions, stop below support with small buffer
                results['sr_stop'] = support_resistance * 0.995  # 0.5% buffer below support
            else:
                # For short positions, stop above resistance with small buffer
                results['sr_stop'] = support_resistance * 1.005  # 0.5% buffer above resistance
            
            results['support_resistance'] = support_resistance
        
        # Recommend best stop loss
        results['recommended_stop'] = self._get_recommended_stop_loss(results, direction)
        
        return results
    
    def _get_recommended_stop_loss(self, stops: Dict, direction: str) -> float:
        """Determine the best stop loss from available methods"""
        percentage_stop = stops['percentage_stop']
        
        # Start with percentage-based stop
        recommended = percentage_stop
        
        # If ATR stop is available and more conservative, use it
        if 'atr_stop' in stops:
            atr_stop = stops['atr_stop']
            if direction == 'BUY':
                # For long, use higher stop (less risk)
                recommended = max(percentage_stop, atr_stop)
            else:
                # For short, use lower stop (less risk)
                recommended = min(percentage_stop, atr_stop)
        
        # If S/R stop is available and reasonable, consider it
        if 'sr_stop' in stops:
            sr_stop = stops['sr_stop']
            entry_price = stops['entry_price']
            
            # Check if S/R stop is within reasonable range (0.5% to 5%)
            sr_risk_pct = abs(sr_stop - entry_price) / entry_price * 100
            
            if 0.5 <= sr_risk_pct <= 5.0:
                if direction == 'BUY':
                    recommended = max(recommended, sr_stop)
                else:
                    recommended = min(recommended, sr_stop)
        
        return recommended
    
    def update_account_balance(self, new_balance: float):
        """Update account balance"""
        if new_balance <= 0:
            raise ValueError("Account balance must be positive")
        
        old_balance = self.config.account_balance
        self.config.account_balance = new_balance
        
        logger.info(f"Account balance updated: ₹{old_balance:,.0f} → ₹{new_balance:,.0f}")
    
    def add_position(self, symbol: str, quantity: int, entry_price: float, 
                    sector: str = None, position_type: str = 'LONG'):
        """Add position to tracking"""
        position_value = quantity * entry_price
        
        self.current_positions[symbol] = {
            'quantity': quantity,
            'entry_price': entry_price,
            'position_value': position_value,
            'sector': sector,
            'position_type': position_type,
            'timestamp': pd.Timestamp.now()
        }
        
        # Update sector exposure
        if sector:
            if sector not in self.sector_exposure:
                self.sector_exposure[sector] = 0
            self.sector_exposure[sector] += position_value
        
        logger.info(f"Position added: {symbol} - {quantity} shares @ ₹{entry_price}")
    
    def remove_position(self, symbol: str):
        """Remove position from tracking"""
        if symbol in self.current_positions:
            position = self.current_positions[symbol]
            sector = position.get('sector')
            
            # Update sector exposure
            if sector and sector in self.sector_exposure:
                self.sector_exposure[sector] -= position['position_value']
                if self.sector_exposure[sector] <= 0:
                    del self.sector_exposure[sector]
            
            del self.current_positions[symbol]
            logger.info(f"Position removed: {symbol}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        total_exposure = sum(pos['position_value'] for pos in self.current_positions.values())
        
        return {
            'account_balance': self.config.account_balance,
            'total_positions': len(self.current_positions),
            'total_exposure': total_exposure,
            'exposure_percentage': (total_exposure / self.config.account_balance) * 100,
            'available_balance': self.config.account_balance - total_exposure,
            'sector_exposure': self.sector_exposure.copy(),
            'positions': self.current_positions.copy()
        }
    
    def _calculate_transaction_costs(self, entry_price: float, quantity: int) -> float:
        """Calculate total transaction costs"""
        position_value = entry_price * quantity
        
        # Flat brokerage
        brokerage = self.config.brokerage_per_trade
        
        # Percentage-based charges (STT, exchange charges, GST, etc.)
        percentage_charges = position_value * (self.config.taxes_and_charges_pct / 100)
        
        return brokerage + percentage_charges
    
    def _check_concentration_limits(self, position_value: float, sector: str = None) -> Optional[str]:
        """Check if position violates concentration limits"""
        # Check maximum position size
        position_pct = (position_value / self.config.account_balance) * 100
        max_position_pct = self.config.max_position_size * 100
        
        if position_pct > max_position_pct:
            return f"Position size {position_pct:.1f}% exceeds limit of {max_position_pct:.1f}%"
        
        # Check sector concentration
        if sector:
            current_sector_exposure = self.sector_exposure.get(sector, 0)
            new_sector_exposure = current_sector_exposure + position_value
            sector_pct = (new_sector_exposure / self.config.account_balance) * 100
            limit_pct = self.config.sector_concentration_limit * 100
            
            if sector_pct > limit_pct:
                return f"Sector exposure {sector_pct:.1f}% exceeds limit of {limit_pct:.1f}%"
        
        # Check maximum concurrent positions
        if len(self.current_positions) >= self.config.max_concurrent_positions:
            return f"Maximum concurrent positions ({self.config.max_concurrent_positions}) reached"
        
        return None
    
    def _get_error_result(self, error_message: str) -> Dict:
        """Return error result with safe defaults"""
        return {
            'quantity': 0,
            'position_value': 0,
            'risk_amount': 0,
            'risk_percentage': 0,
            'error': error_message,
            'valid': False
        }


# Utility functions
def quick_position_size(
    account_balance: float,
    entry_price: float, 
    stop_loss_price: float,
    risk_percentage: float = 2.0
) -> int:
    """Quick position size calculation"""
    risk_amount = account_balance * (risk_percentage / 100)
    risk_per_share = abs(entry_price - stop_loss_price)
    
    if risk_per_share == 0:
        return 0
    
    return int(risk_amount / risk_per_share)


def calculate_risk_reward_ratio(
    entry_price: float,
    stop_loss_price: float, 
    target_price: float
) -> float:
    """Calculate risk-reward ratio"""
    risk = abs(entry_price - stop_loss_price)
    reward = abs(target_price - entry_price)
    
    if risk == 0:
        return 0
    
    return reward / risk