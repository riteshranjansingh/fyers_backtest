"""
🎯 Enhanced Position Sizing Engine

Comprehensive position sizing system supporting multiple methods:
- Risk-based sizing (current method)
- Portfolio percentage sizing
- Fixed quantity sizing  
- Hybrid sizing (max of risk-based and portfolio %)

Author: Fyers Backtesting System
Date: 2025-06-27
"""

from dataclasses import dataclass
from typing import Dict, Union, Optional, Any
import numpy as np
import logging

from .risk_profiles import RiskConfiguration, PositionSizingMethod

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingResult:
    """Result of position sizing calculation"""
    quantity: int
    position_value: float
    risk_amount: float
    risk_percentage: float
    method_used: str
    details: Dict[str, Any]
    valid: bool = True
    warnings: list = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class EnhancedPositionSizer:
    """
    Enhanced position sizing engine supporting multiple methods
    """
    
    def __init__(self, config: RiskConfiguration):
        """Initialize position sizer with configuration"""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Enhanced position sizer initialized with {config.position_sizing_method.value} method")
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        symbol: str = "UNKNOWN",
        signal_confidence: float = 1.0,
        custom_config: Optional[RiskConfiguration] = None
    ) -> PositionSizingResult:
        """
        Calculate position size using configured method
        
        Args:
            entry_price: Entry price per share
            stop_loss_price: Stop loss price per share
            symbol: Trading symbol
            signal_confidence: Signal confidence (0.0 to 1.0)
            custom_config: Optional custom configuration override
            
        Returns:
            PositionSizingResult with calculated position details
        """
        config = custom_config or self.config
        
        try:
            # Validate inputs
            if entry_price <= 0:
                raise ValueError(f"Entry price must be positive, got {entry_price}")
            if stop_loss_price <= 0:
                raise ValueError(f"Stop loss price must be positive, got {stop_loss_price}")
            if not 0.0 <= signal_confidence <= 1.0:
                raise ValueError(f"Signal confidence must be between 0.0 and 1.0, got {signal_confidence}")
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss_price)
            if risk_per_share == 0:
                raise ValueError("Entry price and stop loss price cannot be the same")
            
            # Route to appropriate sizing method
            if config.position_sizing_method == PositionSizingMethod.RISK_BASED:
                result = self._calculate_risk_based_size(
                    entry_price, risk_per_share, signal_confidence, config
                )
            elif config.position_sizing_method == PositionSizingMethod.PORTFOLIO_PERCENT:
                result = self._calculate_portfolio_percent_size(
                    entry_price, risk_per_share, signal_confidence, config
                )
            elif config.position_sizing_method == PositionSizingMethod.FIXED_QUANTITY:
                result = self._calculate_fixed_quantity_size(
                    entry_price, risk_per_share, signal_confidence, config
                )
            elif config.position_sizing_method == PositionSizingMethod.HYBRID:
                result = self._calculate_hybrid_size(
                    entry_price, risk_per_share, signal_confidence, config
                )
            else:
                raise ValueError(f"Unknown position sizing method: {config.position_sizing_method}")
            
            # Apply position limits and validation
            result = self._apply_position_limits(result, config)
            
            # Log the result
            self.logger.info(
                f"Position calculated for {symbol}: {result.quantity} shares, "
                f"₹{result.position_value:,.0f} value, {result.risk_percentage:.2f}% risk"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Position sizing error for {symbol}: {str(e)}")
            return PositionSizingResult(
                quantity=0,
                position_value=0.0,
                risk_amount=0.0,
                risk_percentage=0.0,
                method_used="error",
                details={"error": str(e)},
                valid=False,
                warnings=[f"Position sizing failed: {str(e)}"]
            )
    
    def _calculate_risk_based_size(
        self,
        entry_price: float,
        risk_per_share: float,
        signal_confidence: float,
        config: RiskConfiguration
    ) -> PositionSizingResult:
        """Calculate position size based on risk percentage"""
        
        # Calculate base risk amount
        base_risk_amount = config.account_balance * (config.risk_percent / 100)
        
        # Apply confidence scaling if enabled
        if config.enable_position_scaling:
            confidence_adjusted_risk = base_risk_amount * signal_confidence * config.confidence_multiplier
            confidence_adjusted_risk = min(confidence_adjusted_risk, base_risk_amount * 1.5)  # Cap at 150%
        else:
            confidence_adjusted_risk = base_risk_amount
        
        # Calculate quantity
        base_quantity = confidence_adjusted_risk / risk_per_share
        quantity = max(1, int(base_quantity))
        
        # Calculate actual values
        position_value = quantity * entry_price
        actual_risk_amount = quantity * risk_per_share
        risk_percentage = (actual_risk_amount / config.account_balance) * 100
        
        details = {
            "base_risk_amount": base_risk_amount,
            "confidence_adjusted_risk": confidence_adjusted_risk,
            "signal_confidence": signal_confidence,
            "risk_per_share": risk_per_share,
            "base_quantity": base_quantity
        }
        
        warnings = []
        if base_quantity < 1:
            warnings.append(f"Calculated quantity {base_quantity:.2f} rounded up to minimum 1 share")
        
        if config.enable_position_scaling and signal_confidence < 0.5:
            warnings.append(f"Low signal confidence ({signal_confidence:.2f}) reduced position size")
        
        return PositionSizingResult(
            quantity=quantity,
            position_value=position_value,
            risk_amount=actual_risk_amount,
            risk_percentage=risk_percentage,
            method_used="risk_based",
            details=details,
            warnings=warnings
        )
    
    def _calculate_portfolio_percent_size(
        self,
        entry_price: float,
        risk_per_share: float,
        signal_confidence: float,
        config: RiskConfiguration
    ) -> PositionSizingResult:
        """Calculate position size based on portfolio percentage"""
        
        # Calculate base position value
        base_position_value = config.account_balance * (config.portfolio_percent / 100)
        
        # Apply confidence scaling if enabled
        if config.enable_position_scaling:
            confidence_adjusted_value = base_position_value * signal_confidence * config.confidence_multiplier
            confidence_adjusted_value = min(confidence_adjusted_value, base_position_value * 1.5)  # Cap at 150%
        else:
            confidence_adjusted_value = base_position_value
        
        # Calculate quantity
        base_quantity = confidence_adjusted_value / entry_price
        quantity = max(1, int(base_quantity))
        
        # Calculate actual values
        position_value = quantity * entry_price
        actual_risk_amount = quantity * risk_per_share
        risk_percentage = (actual_risk_amount / config.account_balance) * 100
        
        details = {
            "base_position_value": base_position_value,
            "confidence_adjusted_value": confidence_adjusted_value,
            "signal_confidence": signal_confidence,
            "portfolio_percent": config.portfolio_percent,
            "base_quantity": base_quantity
        }
        
        warnings = []
        if base_quantity < 1:
            warnings.append(f"Calculated quantity {base_quantity:.2f} rounded up to minimum 1 share")
        
        if risk_percentage > config.risk_percent * 2:
            warnings.append(f"High risk exposure ({risk_percentage:.2f}%) from portfolio % method")
        
        return PositionSizingResult(
            quantity=quantity,
            position_value=position_value,
            risk_amount=actual_risk_amount,
            risk_percentage=risk_percentage,
            method_used="portfolio_percent",
            details=details,
            warnings=warnings
        )
    
    def _calculate_fixed_quantity_size(
        self,
        entry_price: float,
        risk_per_share: float,
        signal_confidence: float,
        config: RiskConfiguration
    ) -> PositionSizingResult:
        """Calculate position size using fixed quantity"""
        
        # Base quantity from configuration
        base_quantity = config.fixed_quantity
        
        # Apply confidence scaling if enabled
        if config.enable_position_scaling:
            confidence_adjusted_quantity = base_quantity * signal_confidence * config.confidence_multiplier
            quantity = max(1, int(confidence_adjusted_quantity))
        else:
            quantity = base_quantity
        
        # Calculate actual values
        position_value = quantity * entry_price
        actual_risk_amount = quantity * risk_per_share
        risk_percentage = (actual_risk_amount / config.account_balance) * 100
        
        details = {
            "base_quantity": base_quantity,
            "signal_confidence": signal_confidence,
            "fixed_quantity_config": config.fixed_quantity
        }
        
        warnings = []
        if risk_percentage > config.risk_percent * 3:
            warnings.append(f"Very high risk exposure ({risk_percentage:.2f}%) from fixed quantity method")
        
        if config.enable_position_scaling and signal_confidence < 0.5:
            warnings.append(f"Low signal confidence ({signal_confidence:.2f}) reduced position size")
        
        return PositionSizingResult(
            quantity=quantity,
            position_value=position_value,
            risk_amount=actual_risk_amount,
            risk_percentage=risk_percentage,
            method_used="fixed_quantity",
            details=details,
            warnings=warnings
        )
    
    def _calculate_hybrid_size(
        self,
        entry_price: float,
        risk_per_share: float,
        signal_confidence: float,
        config: RiskConfiguration
    ) -> PositionSizingResult:
        """Calculate position size using hybrid method (max of risk-based and portfolio %)"""
        
        # Calculate both methods
        risk_based = self._calculate_risk_based_size(entry_price, risk_per_share, signal_confidence, config)
        portfolio_based = self._calculate_portfolio_percent_size(entry_price, risk_per_share, signal_confidence, config)
        
        # Choose the larger position (but cap it for safety)
        if risk_based.quantity >= portfolio_based.quantity:
            chosen_result = risk_based
            other_method = "portfolio_percent"
            other_quantity = portfolio_based.quantity
        else:
            chosen_result = portfolio_based
            other_method = "risk_based"
            other_quantity = risk_based.quantity
        
        # Update method name and details
        chosen_result.method_used = "hybrid"
        chosen_result.details.update({
            "hybrid_comparison": {
                "risk_based_quantity": risk_based.quantity,
                "portfolio_based_quantity": portfolio_based.quantity,
                "chosen_method": chosen_result.method_used.replace("hybrid", 
                    "risk_based" if chosen_result.quantity == risk_based.quantity else "portfolio_percent"
                )
            }
        })
        
        # Add hybrid-specific warnings
        if abs(risk_based.quantity - portfolio_based.quantity) / max(risk_based.quantity, portfolio_based.quantity) > 0.5:
            chosen_result.warnings.append(
                f"Large difference between methods: {risk_based.quantity} vs {portfolio_based.quantity} shares"
            )
        
        return chosen_result
    
    def _apply_position_limits(
        self,
        result: PositionSizingResult,
        config: RiskConfiguration
    ) -> PositionSizingResult:
        """Apply position limits and final validation"""
        
        original_quantity = result.quantity
        
        # Apply minimum position value limit
        if result.position_value < config.min_position_value:
            new_quantity = max(1, int(config.min_position_value / (result.position_value / result.quantity)))
            if new_quantity != result.quantity:
                result.warnings.append(
                    f"Position increased from {result.quantity} to {new_quantity} shares to meet minimum value ₹{config.min_position_value:,.0f}"
                )
                result.quantity = new_quantity
        
        # Apply maximum position value limit
        if result.position_value > config.max_position_value:
            new_quantity = int(config.max_position_value / (result.position_value / result.quantity))
            if new_quantity != result.quantity and new_quantity > 0:
                result.warnings.append(
                    f"Position reduced from {result.quantity} to {new_quantity} shares to respect maximum value ₹{config.max_position_value:,.0f}"
                )
                result.quantity = new_quantity
            elif new_quantity <= 0:
                result.valid = False
                result.warnings.append("Position size would exceed maximum value limit")
                return result
        
        # Recalculate values if quantity changed
        if result.quantity != original_quantity:
            entry_price = result.position_value / original_quantity  # Back-calculate entry price
            risk_per_share = result.risk_amount / original_quantity   # Back-calculate risk per share
            
            result.position_value = result.quantity * entry_price
            result.risk_amount = result.quantity * risk_per_share
            result.risk_percentage = (result.risk_amount / config.account_balance) * 100
        
        # Final risk validation
        if result.risk_percentage > config.max_portfolio_risk:
            result.warnings.append(
                f"Position risk ({result.risk_percentage:.2f}%) exceeds maximum portfolio risk ({config.max_portfolio_risk:.2f}%)"
            )
        
        # Ensure minimum quantity
        if result.quantity < 1:
            result.quantity = 1
            result.warnings.append("Position size set to minimum 1 share")
        
        return result
    
    def get_position_sizing_summary(self) -> Dict[str, Any]:
        """Get summary of current position sizing configuration"""
        return {
            "method": self.config.position_sizing_method.value,
            "risk_percent": self.config.risk_percent,
            "portfolio_percent": self.config.portfolio_percent,
            "fixed_quantity": self.config.fixed_quantity,
            "min_position_value": self.config.min_position_value,
            "max_position_value": self.config.max_position_value,
            "enable_position_scaling": self.config.enable_position_scaling,
            "confidence_multiplier": self.config.confidence_multiplier,
            "account_balance": self.config.account_balance
        }


def compare_position_sizing_methods(
    entry_price: float,
    stop_loss_price: float,
    config: RiskConfiguration,
    signal_confidence: float = 1.0
) -> Dict[str, PositionSizingResult]:
    """
    Compare all position sizing methods for given parameters
    
    Returns:
        Dictionary with results from all methods
    """
    results = {}
    
    # Test each method
    methods = [
        PositionSizingMethod.RISK_BASED,
        PositionSizingMethod.PORTFOLIO_PERCENT,
        PositionSizingMethod.FIXED_QUANTITY,
        PositionSizingMethod.HYBRID
    ]
    
    for method in methods:
        # Create temporary config with this method
        temp_config = RiskConfiguration(
            **{**config.to_dict(), 'position_sizing_method': method}
        )
        temp_config.position_sizing_method = method  # Ensure enum is set correctly
        
        # Calculate position size
        sizer = EnhancedPositionSizer(temp_config)
        result = sizer.calculate_position_size(entry_price, stop_loss_price, signal_confidence=signal_confidence)
        results[method.value] = result
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    from .risk_profiles import RiskProfileManager
    
    print("🎯 Enhanced Position Sizing Engine Test")
    print("=" * 50)
    
    # Test parameters
    entry_price = 2500.0
    stop_loss_price = 2450.0  # 2% stop loss
    signal_confidence = 0.8
    
    # Test with different profiles
    profiles = RiskProfileManager.get_all_profiles()
    
    for profile_name, config in profiles.items():
        print(f"\n📊 Testing {profile_name.upper()} Profile:")
        print("-" * 30)
        
        sizer = EnhancedPositionSizer(config)
        result = sizer.calculate_position_size(entry_price, stop_loss_price, signal_confidence=signal_confidence)
        
        print(f"Method: {result.method_used}")
        print(f"Quantity: {result.quantity} shares")
        print(f"Position Value: ₹{result.position_value:,.0f}")
        print(f"Risk Amount: ₹{result.risk_amount:,.0f}")
        print(f"Risk Percentage: {result.risk_percentage:.2f}%")
        
        if result.warnings:
            print(f"Warnings: {', '.join(result.warnings)}")
    
    # Test method comparison
    print(f"\n🔍 Comparing All Methods (Entry: ₹{entry_price}, Stop: ₹{stop_loss_price}):")
    print("-" * 60)
    
    moderate_config = RiskProfileManager.get_moderate_profile()
    comparison = compare_position_sizing_methods(entry_price, stop_loss_price, moderate_config, signal_confidence)
    
    for method, result in comparison.items():
        print(f"{method:15s}: {result.quantity:3d} shares, ₹{result.position_value:8,.0f}, {result.risk_percentage:5.2f}% risk")
    
    print("\n✅ Enhanced position sizing tests completed!")