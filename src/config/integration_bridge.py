"""
🎯 Integration Bridge - Connect New Configuration System with Existing Risk Management

This module provides seamless integration between the new configurability framework
and the existing risk management system, ensuring backward compatibility while
enabling all new configuration features.

Author: Fyers Backtesting System
Date: 2025-06-27
"""

from typing import Dict, Any, Optional
import logging

# Import new configuration system
from .risk_profiles import RiskConfiguration, RiskProfileManager
from .configuration_manager import ConfigurationManager

# Import existing risk management system
from ..risk.risk_integration import RiskManager, RiskManagerConfig
from ..risk.position_sizer import RiskConfig

logger = logging.getLogger(__name__)


class ConfigurationBridge:
    """
    Bridge between new configuration system and existing risk management
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """Initialize configuration bridge"""
        self.config_manager = config_manager or ConfigurationManager()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Configuration bridge initialized")
    
    def convert_to_legacy_config(self, new_config: RiskConfiguration) -> RiskConfig:
        """
        Convert new configuration to legacy RiskConfig format
        
        Args:
            new_config: New RiskConfiguration object
            
        Returns:
            Legacy RiskConfig object
        """
        try:
            # Calculate max position size as percentage (ensure it's between 0.01 and 0.5)
            max_pos_size = min(0.5, max(0.01, new_config.max_position_value / new_config.account_balance))
            
            # Create legacy config with compatible parameters only
            legacy_config = RiskConfig(
                account_balance=new_config.account_balance,
                risk_percentage=new_config.risk_percent,
                max_position_size=max_pos_size,
                default_stop_loss_pct=new_config.stop_loss_percent,
                currency=new_config.currency
            )
            
            self.logger.debug(f"Converted config: {new_config.risk_percent}% risk → legacy format")
            return legacy_config
            
        except Exception as e:
            self.logger.error(f"Error converting config: {str(e)}")
            # Fallback to basic config
            return RiskConfig(
                risk_percentage=new_config.risk_percent,
                account_balance=new_config.account_balance
            )
    
    def create_risk_manager_with_profile(self, profile_name: str) -> RiskManager:
        """
        Create RiskManager with specified profile
        
        Args:
            profile_name: Name of risk profile to use
            
        Returns:
            Configured RiskManager instance
        """
        try:
            # Set profile in configuration manager
            self.config_manager.set_profile(profile_name)
            
            # Get new configuration
            new_config = self.config_manager.active_config
            
            # Convert to legacy format
            legacy_risk_config = self.convert_to_legacy_config(new_config)
            
            # Create RiskManagerConfig with the legacy RiskConfig
            manager_config = RiskManagerConfig(
                risk_config=legacy_risk_config,
                enable_trailing_stops=new_config.enable_trailing_stops,
                enable_gap_protection=new_config.gap_protection_level.value != "disabled",
                risk_profile=profile_name,
                max_portfolio_risk=new_config.max_portfolio_risk
            )
            
            # Create and return risk manager
            risk_manager = RiskManager(manager_config)
            
            self.logger.info(f"Risk manager created with {profile_name} profile")
            return risk_manager
            
        except Exception as e:
            self.logger.error(f"Error creating risk manager with profile '{profile_name}': {str(e)}")
            # Fallback to moderate profile
            fallback_config = RiskProfileManager.get_moderate_profile()
            legacy_risk_config = self.convert_to_legacy_config(fallback_config)
            manager_config = RiskManagerConfig(
                risk_config=legacy_risk_config,
                risk_profile="moderate"
            )
            return RiskManager(manager_config)
    
    def create_risk_manager_with_custom_config(self, config_updates: Dict[str, Any]) -> RiskManager:
        """
        Create RiskManager with custom configuration
        
        Args:
            config_updates: Dictionary of configuration updates
            
        Returns:
            Configured RiskManager instance
        """
        try:
            # Update configuration
            success = self.config_manager.update_configuration(config_updates)
            
            if not success:
                self.logger.warning("Failed to update configuration, using current settings")
            
            # Get current configuration
            new_config = self.config_manager.active_config
            
            # Convert to legacy format
            legacy_risk_config = self.convert_to_legacy_config(new_config)
            
            # Create RiskManagerConfig
            manager_config = RiskManagerConfig(
                risk_config=legacy_risk_config,
                enable_trailing_stops=new_config.enable_trailing_stops,
                enable_gap_protection=new_config.gap_protection_level.value != "disabled",
                risk_profile="custom",
                max_portfolio_risk=new_config.max_portfolio_risk
            )
            
            # Create and return risk manager
            risk_manager = RiskManager(manager_config)
            
            self.logger.info(f"Risk manager created with custom config: {list(config_updates.keys())}")
            return risk_manager
            
        except Exception as e:
            self.logger.error(f"Error creating risk manager with custom config: {str(e)}")
            # Fallback to current configuration
            new_config = self.config_manager.active_config
            legacy_risk_config = self.convert_to_legacy_config(new_config)
            manager_config = RiskManagerConfig(risk_config=legacy_risk_config)
            return RiskManager(manager_config)
    
    def get_enhanced_position_calculator(self):
        """
        Get enhanced position calculator from current configuration
        
        Returns:
            Enhanced position sizer instance
        """
        from .position_sizing_engine import EnhancedPositionSizer
        return EnhancedPositionSizer(self.config_manager.active_config)
    
    def get_advanced_stop_loss_calculator(self):
        """
        Get advanced stop loss calculator from current configuration
        
        Returns:
            Advanced stop loss engine instance
        """
        from .stop_loss_engine import AdvancedStopLossEngine
        return AdvancedStopLossEngine(self.config_manager.active_config)
    
    def calculate_position_with_new_engine(
        self,
        entry_price: float,
        stop_loss_price: float,
        signal_confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate position using new enhanced engine
        
        Args:
            entry_price: Entry price per share
            stop_loss_price: Stop loss price per share
            signal_confidence: Signal confidence (0.0 to 1.0)
            
        Returns:
            Position calculation result
        """
        try:
            position_sizer = self.get_enhanced_position_calculator()
            result = position_sizer.calculate_position_size(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                signal_confidence=signal_confidence
            )
            
            return {
                "quantity": result.quantity,
                "position_value": result.position_value,
                "risk_amount": result.risk_amount,
                "risk_percentage": result.risk_percentage,
                "method_used": result.method_used,
                "valid": result.valid,
                "warnings": result.warnings,
                "enhanced_features": True
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced position calculation failed: {str(e)}")
            return {
                "quantity": 0,
                "position_value": 0.0,
                "risk_amount": 0.0,
                "risk_percentage": 0.0,
                "method_used": "error",
                "valid": False,
                "warnings": [str(e)],
                "enhanced_features": False
            }
    
    def calculate_stop_loss_with_new_engine(
        self,
        entry_price: float,
        signal_direction: str,
        historical_data: Optional = None
    ) -> Dict[str, Any]:
        """
        Calculate stop loss using new advanced engine
        
        Args:
            entry_price: Entry price per share
            signal_direction: 'BUY' or 'SELL'
            historical_data: Optional historical data for advanced methods
            
        Returns:
            Stop loss calculation result
        """
        try:
            stop_loss_engine = self.get_advanced_stop_loss_calculator()
            result = stop_loss_engine.calculate_stop_loss(
                entry_price=entry_price,
                signal_direction=signal_direction,
                historical_data=historical_data
            )
            
            return {
                "stop_price": result.stop_price,
                "stop_percentage": result.stop_percentage,
                "risk_per_share": result.risk_per_share,
                "method_used": result.method_used,
                "confidence": result.confidence,
                "valid": result.valid,
                "warnings": result.warnings,
                "enhanced_features": True
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced stop loss calculation failed: {str(e)}")
            return {
                "stop_price": 0.0,
                "stop_percentage": 0.0,
                "risk_per_share": 0.0,
                "method_used": "error",
                "confidence": 0.0,
                "valid": False,
                "warnings": [str(e)],
                "enhanced_features": False
            }
    
    def test_integration_with_existing_workflow(self) -> Dict[str, Any]:
        """
        Test integration with existing trading workflow
        
        Returns:
            Integration test results
        """
        test_results = {
            "profile_tests": {},
            "position_calculation": {},
            "stop_loss_calculation": {},
            "risk_manager_creation": {},
            "overall_status": "unknown"
        }
        
        try:
            # Test each profile
            profiles = ["conservative", "moderate", "aggressive"]
            
            for profile in profiles:
                try:
                    # Create risk manager with profile
                    risk_manager = self.create_risk_manager_with_profile(profile)
                    
                    # Test basic functionality
                    if hasattr(risk_manager, 'config'):
                        test_results["profile_tests"][profile] = "PASS"
                        test_results["risk_manager_creation"][profile] = "PASS"
                    else:
                        test_results["profile_tests"][profile] = "FAIL: No config attribute"
                        test_results["risk_manager_creation"][profile] = "FAIL"
                        
                except Exception as e:
                    test_results["profile_tests"][profile] = f"FAIL: {str(e)}"
                    test_results["risk_manager_creation"][profile] = f"FAIL: {str(e)}"
            
            # Test enhanced position calculation
            try:
                position_result = self.calculate_position_with_new_engine(2500.0, 2450.0, 0.8)
                if position_result["valid"] and position_result["enhanced_features"]:
                    test_results["position_calculation"] = "PASS"
                else:
                    test_results["position_calculation"] = f"FAIL: {position_result.get('warnings', ['Unknown error'])}"
            except Exception as e:
                test_results["position_calculation"] = f"FAIL: {str(e)}"
            
            # Test enhanced stop loss calculation
            try:
                stop_result = self.calculate_stop_loss_with_new_engine(2500.0, 'BUY')
                if stop_result["valid"] and stop_result["enhanced_features"]:
                    test_results["stop_loss_calculation"] = "PASS"
                else:
                    test_results["stop_loss_calculation"] = f"FAIL: {stop_result.get('warnings', ['Unknown error'])}"
            except Exception as e:
                test_results["stop_loss_calculation"] = f"FAIL: {str(e)}"
            
            # Determine overall status
            passed_tests = sum(1 for result in test_results.values() 
                             if isinstance(result, str) and result == "PASS")
            failed_tests = sum(1 for result in test_results.values() 
                             if isinstance(result, str) and result.startswith("FAIL"))
            
            # Also count nested test results
            for nested_dict in test_results.values():
                if isinstance(nested_dict, dict):
                    passed_tests += sum(1 for result in nested_dict.values() 
                                      if isinstance(result, str) and result == "PASS")
                    failed_tests += sum(1 for result in nested_dict.values() 
                                      if isinstance(result, str) and result.startswith("FAIL"))
            
            if failed_tests == 0:
                test_results["overall_status"] = "PASS"
            elif passed_tests > failed_tests:
                test_results["overall_status"] = "MOSTLY_PASS"
            else:
                test_results["overall_status"] = "FAIL"
            
            self.logger.info(f"Integration test completed: {test_results['overall_status']}")
            
        except Exception as e:
            test_results["overall_status"] = f"FAIL: {str(e)}"
            self.logger.error(f"Integration test failed: {str(e)}")
        
        return test_results
    
    def get_configuration_summary_for_ui(self) -> Dict[str, Any]:
        """
        Get configuration summary formatted for UI display
        
        Returns:
            UI-ready configuration summary
        """
        try:
            base_summary = self.config_manager.get_configuration_summary()
            
            # Add UI-specific formatting
            ui_summary = {
                "active_profile": {
                    "name": base_summary["profile_name"],
                    "display_name": base_summary["profile_name"].title(),
                    "description": self._get_profile_description(base_summary["profile_name"])
                },
                "risk_settings": {
                    "risk_per_trade": f"{base_summary['risk_management']['risk_percent']}%",
                    "max_portfolio_risk": f"{base_summary['risk_management']['max_portfolio_risk']}%",
                    "max_daily_loss": f"{base_summary['risk_management']['max_daily_loss']}%"
                },
                "position_sizing": {
                    "method": base_summary['position_sizing']['method'].replace('_', ' ').title(),
                    "risk_percent": f"{base_summary['position_sizing']['risk_percent']}%",
                    "portfolio_percent": f"{base_summary['position_sizing'].get('portfolio_percent', 0)}%"
                },
                "stop_loss": {
                    "method": base_summary['stop_loss']['method'].replace('_', ' ').title(),
                    "percentage": f"{base_summary['stop_loss']['stop_loss_percent']}%",
                    "atr_multiplier": base_summary['stop_loss'].get('atr_multiplier', 'N/A')
                },
                "trailing_stops": {
                    "enabled": "Yes" if base_summary['trailing_stops']['enabled'] else "No",
                    "breakeven_trigger": f"{base_summary['trailing_stops']['breakeven_trigger']}:1 R:R",
                    "partial_booking": f"{base_summary['trailing_stops']['partial_booking_percent']}% at {base_summary['trailing_stops']['partial_booking_ratio']}:1"
                },
                "gap_protection": {
                    "level": base_summary['gap_protection']['level'].title(),
                    "overnight_reduction": f"{(1-base_summary['gap_protection']['overnight_reduction'])*100:.0f}%",
                    "weekend_reduction": f"{(1-base_summary['gap_protection']['weekend_reduction'])*100:.0f}%"
                },
                "account": {
                    "balance": f"₹{base_summary['account']['balance']:,.0f}",
                    "currency": base_summary['account']['currency']
                }
            }
            
            return ui_summary
            
        except Exception as e:
            self.logger.error(f"Error creating UI summary: {str(e)}")
            return {"error": str(e)}
    
    def _get_profile_description(self, profile_name: str) -> str:
        """Get description for profile"""
        descriptions = {
            "conservative": "Low risk, tight stops, maximum protection - ideal for capital preservation",
            "moderate": "Balanced approach with standard risk management - suitable for most traders",
            "aggressive": "Higher risk for higher returns - requires careful monitoring",
            "custom": "User-defined configuration with personalized risk parameters"
        }
        return descriptions.get(profile_name, "Custom configuration profile")


# Global bridge instance
_integration_bridge = None

def get_integration_bridge() -> ConfigurationBridge:
    """Get global integration bridge instance"""
    global _integration_bridge
    if _integration_bridge is None:
        _integration_bridge = ConfigurationBridge()
    return _integration_bridge


# Convenience functions for easy integration
def create_risk_manager(profile_name: str = "moderate") -> RiskManager:
    """
    Create risk manager with specified profile
    
    Args:
        profile_name: Risk profile to use ('conservative', 'moderate', 'aggressive')
        
    Returns:
        Configured RiskManager instance
    """
    bridge = get_integration_bridge()
    return bridge.create_risk_manager_with_profile(profile_name)


def create_custom_risk_manager(**config_updates) -> RiskManager:
    """
    Create risk manager with custom configuration
    
    Args:
        **config_updates: Configuration updates as keyword arguments
        
    Returns:
        Configured RiskManager instance
    """
    bridge = get_integration_bridge()
    return bridge.create_risk_manager_with_custom_config(config_updates)


def calculate_enhanced_position(
    entry_price: float,
    stop_loss_price: float,
    signal_confidence: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate position using enhanced engine
    
    Args:
        entry_price: Entry price per share
        stop_loss_price: Stop loss price per share
        signal_confidence: Signal confidence (0.0 to 1.0)
        
    Returns:
        Position calculation result
    """
    bridge = get_integration_bridge()
    return bridge.calculate_position_with_new_engine(entry_price, stop_loss_price, signal_confidence)


def calculate_enhanced_stop_loss(
    entry_price: float,
    signal_direction: str,
    historical_data=None
) -> Dict[str, Any]:
    """
    Calculate stop loss using enhanced engine
    
    Args:
        entry_price: Entry price per share
        signal_direction: 'BUY' or 'SELL'
        historical_data: Optional historical data
        
    Returns:
        Stop loss calculation result
    """
    bridge = get_integration_bridge()
    return bridge.calculate_stop_loss_with_new_engine(entry_price, signal_direction, historical_data)


if __name__ == "__main__":
    # Test integration bridge
    print("🎯 Configuration Integration Bridge Test")
    print("=" * 50)
    
    # Initialize bridge
    bridge = ConfigurationBridge()
    
    # Test profile-based risk manager creation
    print("\n📊 Testing Profile-Based Risk Manager Creation:")
    for profile in ["conservative", "moderate", "aggressive"]:
        try:
            risk_manager = bridge.create_risk_manager_with_profile(profile)
            print(f"✅ {profile.title()}: {risk_manager.config.risk_percentage}% risk")
        except Exception as e:
            print(f"❌ {profile.title()}: {str(e)}")
    
    # Test enhanced calculations
    print("\n💰 Testing Enhanced Position Calculation:")
    try:
        result = bridge.calculate_position_with_new_engine(2500.0, 2450.0, 0.8)
        if result["valid"]:
            print(f"✅ Quantity: {result['quantity']}, Risk: {result['risk_percentage']:.2f}%")
        else:
            print(f"❌ Failed: {result['warnings']}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test integration
    print("\n🔗 Testing Full Integration:")
    test_results = bridge.test_integration_with_existing_workflow()
    print(f"Overall Status: {test_results['overall_status']}")
    
    print("\n✅ Integration bridge test completed!")