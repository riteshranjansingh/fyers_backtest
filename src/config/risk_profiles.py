"""
🎯 Risk Profile Configuration System

This module provides comprehensive configuration options for risk management,
position sizing, stop losses, and trading parameters. Designed to be UI-ready
and user-friendly with preset profiles and custom configurations.

Author: Fyers Backtesting System
Date: 2025-06-27
"""

from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional, Any
from enum import Enum
import json
import os


class RiskLevel(Enum):
    """Risk level enumeration"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class PositionSizingMethod(Enum):
    """Position sizing method enumeration"""
    RISK_BASED = "risk_based"           # Current method: % of account at risk
    PORTFOLIO_PERCENT = "portfolio_percent"  # % of total portfolio
    FIXED_QUANTITY = "fixed_quantity"   # Fixed number of shares
    HYBRID = "hybrid"                   # Max of risk-based and portfolio %


class StopLossMethod(Enum):
    """Stop loss method enumeration"""
    PERCENTAGE = "percentage"           # Fixed percentage
    ATR = "atr"                        # ATR-based
    SUPPORT_RESISTANCE = "support_resistance"  # Support/Resistance levels
    ADAPTIVE = "adaptive"              # Adaptive based on market conditions


class GapProtectionLevel(Enum):
    """Gap protection level enumeration"""
    MINIMAL = "minimal"      # Light protection
    STANDARD = "standard"    # Normal protection
    MAXIMUM = "maximum"      # Heavy protection
    DISABLED = "disabled"    # No gap protection


@dataclass
class RiskConfiguration:
    """
    Comprehensive risk management configuration
    """
    # Core Risk Settings
    risk_percent: float = 2.0                    # Risk per trade (0.5% to 3.0%)
    max_portfolio_risk: float = 10.0             # Max total portfolio risk
    max_daily_loss: float = 5.0                  # Max daily loss limit
    
    # Position Sizing
    position_sizing_method: PositionSizingMethod = PositionSizingMethod.RISK_BASED
    portfolio_percent: float = 10.0              # If using portfolio % method
    fixed_quantity: int = 100                    # If using fixed quantity method
    min_position_value: float = 1000.0           # Minimum position value
    max_position_value: float = 100000.0         # Maximum position value
    
    # Stop Loss Configuration
    stop_loss_method: StopLossMethod = StopLossMethod.PERCENTAGE
    stop_loss_percent: float = 1.5               # Default stop loss %
    atr_multiplier: float = 2.0                  # ATR multiplier for ATR stops
    atr_period: int = 14                         # ATR calculation period
    adaptive_volatility_factor: float = 1.5      # Volatility adjustment factor
    
    # Trailing Stop Configuration
    enable_trailing_stops: bool = True
    breakeven_trigger_ratio: float = 1.0         # R:R ratio to move to breakeven
    breakeven_buffer_points: float = 2.0         # Points above breakeven
    partial_booking_enabled: bool = True
    partial_booking_ratio: float = 2.0           # R:R ratio for partial booking
    partial_booking_percent: float = 50.0        # % of position to book
    trailing_step_percent: float = 0.5           # Trailing step size
    
    # Gap Protection
    gap_protection_level: GapProtectionLevel = GapProtectionLevel.STANDARD
    overnight_reduction_factor: float = 0.9      # Overnight position reduction
    weekend_reduction_factor: float = 0.8        # Weekend position reduction
    friday_reduction_factor: float = 0.9         # Friday position reduction
    gap_stop_buffer_percent: float = 1.0         # Additional gap stop buffer
    
    # Advanced Settings
    enable_position_scaling: bool = False        # Scale positions based on confidence
    confidence_multiplier: float = 1.0           # Signal confidence multiplier
    volatility_adjustment: bool = True           # Adjust for market volatility
    correlation_limit: float = 0.7               # Max correlation between positions
    
    # Account Settings
    account_balance: float = 100000.0            # Account balance
    currency: str = "INR"                        # Account currency
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        # Risk validation
        if not 0.1 <= self.risk_percent <= 5.0:
            raise ValueError(f"Risk percent must be between 0.1% and 5.0%, got {self.risk_percent}%")
        
        if not 1.0 <= self.max_portfolio_risk <= 50.0:
            raise ValueError(f"Max portfolio risk must be between 1% and 50%, got {self.max_portfolio_risk}%")
        
        # Position sizing validation
        if self.position_sizing_method == PositionSizingMethod.PORTFOLIO_PERCENT:
            if not 1.0 <= self.portfolio_percent <= 50.0:
                raise ValueError(f"Portfolio percent must be between 1% and 50%, got {self.portfolio_percent}%")
        
        if self.position_sizing_method == PositionSizingMethod.FIXED_QUANTITY:
            if self.fixed_quantity <= 0:
                raise ValueError(f"Fixed quantity must be positive, got {self.fixed_quantity}")
        
        # Stop loss validation
        if not 0.1 <= self.stop_loss_percent <= 10.0:
            raise ValueError(f"Stop loss percent must be between 0.1% and 10%, got {self.stop_loss_percent}%")
        
        # ATR validation
        if self.stop_loss_method == StopLossMethod.ATR:
            if not 0.5 <= self.atr_multiplier <= 5.0:
                raise ValueError(f"ATR multiplier must be between 0.5 and 5.0, got {self.atr_multiplier}")
            if not 5 <= self.atr_period <= 50:
                raise ValueError(f"ATR period must be between 5 and 50, got {self.atr_period}")
        
        # Trailing stops validation
        if self.enable_trailing_stops:
            if not 0.5 <= self.breakeven_trigger_ratio <= 5.0:
                raise ValueError(f"Breakeven trigger ratio must be between 0.5 and 5.0, got {self.breakeven_trigger_ratio}")
            
            if self.partial_booking_enabled:
                if not 1.0 <= self.partial_booking_ratio <= 10.0:
                    raise ValueError(f"Partial booking ratio must be between 1.0 and 10.0, got {self.partial_booking_ratio}")
                if not 10.0 <= self.partial_booking_percent <= 90.0:
                    raise ValueError(f"Partial booking percent must be between 10% and 90%, got {self.partial_booking_percent}%")
        
        # Account validation
        if self.account_balance <= 0:
            raise ValueError(f"Account balance must be positive, got {self.account_balance}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                config_dict[key] = value.value
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RiskConfiguration':
        """Create configuration from dictionary"""
        # Convert enum strings back to enums
        if 'position_sizing_method' in config_dict:
            config_dict['position_sizing_method'] = PositionSizingMethod(config_dict['position_sizing_method'])
        if 'stop_loss_method' in config_dict:
            config_dict['stop_loss_method'] = StopLossMethod(config_dict['stop_loss_method'])
        if 'gap_protection_level' in config_dict:
            config_dict['gap_protection_level'] = GapProtectionLevel(config_dict['gap_protection_level'])
        
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'RiskConfiguration':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class RiskProfileManager:
    """
    Manages predefined risk profiles and custom configurations
    """
    
    @staticmethod
    def get_conservative_profile() -> RiskConfiguration:
        """Conservative risk profile - Capital preservation focused"""
        return RiskConfiguration(
            # Conservative risk settings
            risk_percent=1.0,
            max_portfolio_risk=5.0,
            max_daily_loss=2.0,
            
            # Conservative position sizing
            position_sizing_method=PositionSizingMethod.RISK_BASED,
            portfolio_percent=5.0,
            min_position_value=2000.0,
            max_position_value=50000.0,
            
            # Tight stop losses
            stop_loss_method=StopLossMethod.PERCENTAGE,
            stop_loss_percent=1.0,
            atr_multiplier=1.5,
            
            # Conservative trailing
            enable_trailing_stops=True,
            breakeven_trigger_ratio=0.75,
            breakeven_buffer_points=3.0,
            partial_booking_enabled=True,
            partial_booking_ratio=1.5,
            partial_booking_percent=50.0,
            trailing_step_percent=0.25,
            
            # Maximum gap protection
            gap_protection_level=GapProtectionLevel.MAXIMUM,
            overnight_reduction_factor=0.8,
            weekend_reduction_factor=0.7,
            friday_reduction_factor=0.8,
            gap_stop_buffer_percent=1.5,
            
            # Conservative advanced settings
            enable_position_scaling=False,
            volatility_adjustment=True,
            correlation_limit=0.5
        )
    
    @staticmethod
    def get_moderate_profile() -> RiskConfiguration:
        """Moderate risk profile - Balanced approach"""
        return RiskConfiguration(
            # Moderate risk settings
            risk_percent=2.0,
            max_portfolio_risk=10.0,
            max_daily_loss=4.0,
            
            # Balanced position sizing
            position_sizing_method=PositionSizingMethod.RISK_BASED,
            portfolio_percent=10.0,
            min_position_value=1500.0,
            max_position_value=75000.0,
            
            # Standard stop losses
            stop_loss_method=StopLossMethod.ATR,
            stop_loss_percent=1.5,
            atr_multiplier=2.0,
            
            # Standard trailing
            enable_trailing_stops=True,
            breakeven_trigger_ratio=1.0,
            breakeven_buffer_points=2.0,
            partial_booking_enabled=True,
            partial_booking_ratio=2.0,
            partial_booking_percent=50.0,
            trailing_step_percent=0.5,
            
            # Standard gap protection
            gap_protection_level=GapProtectionLevel.STANDARD,
            overnight_reduction_factor=0.9,
            weekend_reduction_factor=0.8,
            friday_reduction_factor=0.9,
            gap_stop_buffer_percent=1.0,
            
            # Moderate advanced settings
            enable_position_scaling=True,
            confidence_multiplier=1.2,
            volatility_adjustment=True,
            correlation_limit=0.7
        )
    
    @staticmethod
    def get_aggressive_profile() -> RiskConfiguration:
        """Aggressive risk profile - Growth focused"""
        return RiskConfiguration(
            # Aggressive risk settings
            risk_percent=3.0,
            max_portfolio_risk=15.0,
            max_daily_loss=6.0,
            
            # Aggressive position sizing
            position_sizing_method=PositionSizingMethod.HYBRID,
            portfolio_percent=15.0,
            min_position_value=1000.0,
            max_position_value=100000.0,
            
            # Wider stop losses
            stop_loss_method=StopLossMethod.ADAPTIVE,
            stop_loss_percent=2.5,
            atr_multiplier=2.5,
            adaptive_volatility_factor=2.0,
            
            # Aggressive trailing
            enable_trailing_stops=True,
            breakeven_trigger_ratio=1.25,
            breakeven_buffer_points=1.0,
            partial_booking_enabled=True,
            partial_booking_ratio=2.5,
            partial_booking_percent=40.0,
            trailing_step_percent=0.75,
            
            # Minimal gap protection
            gap_protection_level=GapProtectionLevel.MINIMAL,
            overnight_reduction_factor=0.95,
            weekend_reduction_factor=0.9,
            friday_reduction_factor=0.95,
            gap_stop_buffer_percent=0.5,
            
            # Aggressive advanced settings
            enable_position_scaling=True,
            confidence_multiplier=1.5,
            volatility_adjustment=True,
            correlation_limit=0.8
        )
    
    @staticmethod
    def get_all_profiles() -> Dict[str, RiskConfiguration]:
        """Get all predefined risk profiles"""
        return {
            "conservative": RiskProfileManager.get_conservative_profile(),
            "moderate": RiskProfileManager.get_moderate_profile(),
            "aggressive": RiskProfileManager.get_aggressive_profile()
        }
    
    @staticmethod
    def get_profile_by_name(profile_name: str) -> RiskConfiguration:
        """Get risk profile by name"""
        profiles = RiskProfileManager.get_all_profiles()
        if profile_name.lower() not in profiles:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(profiles.keys())}")
        return profiles[profile_name.lower()]


# Configuration Options for UI
class ConfigurationOptions:
    """Static configuration options for UI components"""
    
    # Risk percentage options
    RISK_PERCENT_OPTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    
    # Portfolio percentage options
    PORTFOLIO_PERCENT_OPTIONS = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 30]
    
    # Stop loss percentage options
    STOP_LOSS_PERCENT_OPTIONS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    # ATR multiplier options
    ATR_MULTIPLIER_OPTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0]
    
    # Breakeven trigger ratio options
    BREAKEVEN_TRIGGER_OPTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    
    # Partial booking ratio options
    PARTIAL_BOOKING_RATIO_OPTIONS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0]
    
    # Partial booking percentage options
    PARTIAL_BOOKING_PERCENT_OPTIONS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 90]
    
    # Fixed quantity options (common share quantities)
    FIXED_QUANTITY_OPTIONS = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 500, 1000]
    
    # Gap protection factor options
    GAP_PROTECTION_FACTORS = {
        GapProtectionLevel.DISABLED: {
            "overnight": 1.0,
            "weekend": 1.0, 
            "friday": 1.0
        },
        GapProtectionLevel.MINIMAL: {
            "overnight": 0.95,
            "weekend": 0.9,
            "friday": 0.95
        },
        GapProtectionLevel.STANDARD: {
            "overnight": 0.9,
            "weekend": 0.8,
            "friday": 0.9
        },
        GapProtectionLevel.MAXIMUM: {
            "overnight": 0.8,
            "weekend": 0.7,
            "friday": 0.8
        }
    }


if __name__ == "__main__":
    # Example usage and testing
    print("🎯 Risk Profile Configuration System Test")
    print("=" * 50)
    
    # Test conservative profile
    conservative = RiskProfileManager.get_conservative_profile()
    print(f"Conservative Profile - Risk: {conservative.risk_percent}%, Stop Loss: {conservative.stop_loss_percent}%")
    
    # Test moderate profile
    moderate = RiskProfileManager.get_moderate_profile()
    print(f"Moderate Profile - Risk: {moderate.risk_percent}%, Stop Loss: {moderate.stop_loss_percent}%")
    
    # Test aggressive profile
    aggressive = RiskProfileManager.get_aggressive_profile()
    print(f"Aggressive Profile - Risk: {aggressive.risk_percent}%, Stop Loss: {aggressive.stop_loss_percent}%")
    
    # Test custom configuration
    custom = RiskConfiguration(
        risk_percent=1.8,
        position_sizing_method=PositionSizingMethod.PORTFOLIO_PERCENT,
        portfolio_percent=12.0,
        stop_loss_method=StopLossMethod.ATR,
        stop_loss_percent=2.2
    )
    print(f"Custom Profile - Risk: {custom.risk_percent}%, Portfolio: {custom.portfolio_percent}%")
    
    # Test save/load
    custom.save_to_file("test_config.json")
    loaded = RiskConfiguration.load_from_file("test_config.json")
    print(f"Loaded Profile - Risk: {loaded.risk_percent}%, Portfolio: {loaded.portfolio_percent}%")
    
    print("\n✅ All configuration tests passed!")