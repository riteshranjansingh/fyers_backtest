"""
🎯 Configuration Manager - Central Hub for All Configuration

This module provides a unified interface for managing all configuration aspects:
- Risk profiles and custom settings
- Position sizing configuration  
- Stop loss configuration
- UI-ready configuration interface
- Configuration validation and persistence

Author: Fyers Backtesting System
Date: 2025-06-27
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
import json
import os
from pathlib import Path
import logging
import pandas as pd

from .risk_profiles import (
    RiskConfiguration, RiskProfileManager, PositionSizingMethod, 
    StopLossMethod, GapProtectionLevel, ConfigurationOptions
)
from .position_sizing_engine import EnhancedPositionSizer, compare_position_sizing_methods
from .stop_loss_engine import AdvancedStopLossEngine

logger = logging.getLogger(__name__)


@dataclass 
class ConfigurationPreview:
    """Preview of configuration settings for UI display"""
    profile_name: str
    risk_percent: float
    position_method: str
    stop_loss_method: str
    stop_loss_percent: float
    trailing_enabled: bool
    gap_protection: str
    description: str


class ConfigurationManager:
    """
    Central configuration management system
    """
    
    def __init__(self, config_dir: str = "config/saved_profiles"):
        """Initialize configuration manager"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Configuration manager initialized with directory: {self.config_dir}")
        
        # Current active configuration
        self._active_config = RiskProfileManager.get_moderate_profile()
        self._active_profile_name = "moderate"
        
        # Initialize engines with active config
        self._position_sizer = EnhancedPositionSizer(self._active_config)
        self._stop_loss_engine = AdvancedStopLossEngine(self._active_config)
    
    @property
    def active_config(self) -> RiskConfiguration:
        """Get currently active configuration"""
        return self._active_config
    
    @property
    def active_profile_name(self) -> str:
        """Get currently active profile name"""
        return self._active_profile_name
    
    def set_profile(self, profile_name: str) -> bool:
        """
        Set active profile by name
        
        Args:
            profile_name: Name of profile ('conservative', 'moderate', 'aggressive', or custom name)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if it's a predefined profile
            if profile_name.lower() in ['conservative', 'moderate', 'aggressive']:
                self._active_config = RiskProfileManager.get_profile_by_name(profile_name.lower())
                self._active_profile_name = profile_name.lower()
            else:
                # Try to load custom profile
                custom_config = self.load_custom_profile(profile_name)
                if custom_config:
                    self._active_config = custom_config
                    self._active_profile_name = profile_name
                else:
                    self.logger.error(f"Profile '{profile_name}' not found")
                    return False
            
            # Reinitialize engines with new config
            self._position_sizer = EnhancedPositionSizer(self._active_config)
            self._stop_loss_engine = AdvancedStopLossEngine(self._active_config)
            
            self.logger.info(f"Active profile changed to: {profile_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting profile '{profile_name}': {str(e)}")
            return False
    
    def update_configuration(self, updates: Dict[str, Any]) -> bool:
        """
        Update current configuration with new values
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create new configuration with updates
            current_dict = self._active_config.to_dict()
            current_dict.update(updates)
            
            # Create and validate new configuration
            new_config = RiskConfiguration.from_dict(current_dict)
            
            # If validation passes, update active config
            self._active_config = new_config
            self._active_profile_name = "custom"
            
            # Reinitialize engines
            self._position_sizer = EnhancedPositionSizer(self._active_config)
            self._stop_loss_engine = AdvancedStopLossEngine(self._active_config)
            
            self.logger.info(f"Configuration updated: {list(updates.keys())}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            return False
    
    def save_custom_profile(self, profile_name: str, description: str = "") -> bool:
        """
        Save current configuration as custom profile
        
        Args:
            profile_name: Name for the custom profile
            description: Optional description
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create profile data
            profile_data = {
                "config": self._active_config.to_dict(),
                "metadata": {
                    "name": profile_name,
                    "description": description,
                    "created_at": pd.Timestamp.now().isoformat(),
                    "version": "1.0"
                }
            }
            
            # Save to file
            profile_path = self.config_dir / f"{profile_name}.json"
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            self.logger.info(f"Custom profile '{profile_name}' saved to {profile_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving profile '{profile_name}': {str(e)}")
            return False
    
    def load_custom_profile(self, profile_name: str) -> Optional[RiskConfiguration]:
        """
        Load custom profile by name
        
        Args:
            profile_name: Name of profile to load
            
        Returns:
            RiskConfiguration if found, None otherwise
        """
        try:
            profile_path = self.config_dir / f"{profile_name}.json"
            
            if not profile_path.exists():
                return None
            
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)
            
            config = RiskConfiguration.from_dict(profile_data["config"])
            self.logger.info(f"Custom profile '{profile_name}' loaded")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading profile '{profile_name}': {str(e)}")
            return None
    
    def delete_custom_profile(self, profile_name: str) -> bool:
        """
        Delete custom profile
        
        Args:
            profile_name: Name of profile to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            profile_path = self.config_dir / f"{profile_name}.json"
            
            if profile_path.exists():
                profile_path.unlink()
                self.logger.info(f"Custom profile '{profile_name}' deleted")
                return True
            else:
                self.logger.warning(f"Profile '{profile_name}' not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting profile '{profile_name}': {str(e)}")
            return False
    
    def get_all_profiles(self) -> Dict[str, ConfigurationPreview]:
        """
        Get preview of all available profiles
        
        Returns:
            Dictionary of profile previews
        """
        profiles = {}
        
        # Add predefined profiles
        predefined = {
            "conservative": ("Conservative", "Low risk, tight stops, maximum protection"),
            "moderate": ("Moderate", "Balanced approach with standard risk management"),
            "aggressive": ("Aggressive", "Higher risk for higher returns")
        }
        
        for name, (display_name, description) in predefined.items():
            config = RiskProfileManager.get_profile_by_name(name)
            profiles[name] = ConfigurationPreview(
                profile_name=display_name,
                risk_percent=config.risk_percent,
                position_method=config.position_sizing_method.value,
                stop_loss_method=config.stop_loss_method.value,
                stop_loss_percent=config.stop_loss_percent,
                trailing_enabled=config.enable_trailing_stops,
                gap_protection=config.gap_protection_level.value,
                description=description
            )
        
        # Add custom profiles
        for profile_file in self.config_dir.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)
                
                config = RiskConfiguration.from_dict(profile_data["config"])
                metadata = profile_data.get("metadata", {})
                
                profile_name = profile_file.stem
                profiles[profile_name] = ConfigurationPreview(
                    profile_name=metadata.get("name", profile_name),
                    risk_percent=config.risk_percent,
                    position_method=config.position_sizing_method.value,
                    stop_loss_method=config.stop_loss_method.value,
                    stop_loss_percent=config.stop_loss_percent,
                    trailing_enabled=config.enable_trailing_stops,
                    gap_protection=config.gap_protection_level.value,
                    description=metadata.get("description", "Custom profile")
                )
                
            except Exception as e:
                self.logger.warning(f"Error loading profile {profile_file.name}: {str(e)}")
        
        return profiles
    
    def calculate_position_preview(
        self,
        entry_price: float,
        stop_loss_price: float,
        signal_confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate position sizing preview with current configuration
        
        Args:
            entry_price: Entry price per share
            stop_loss_price: Stop loss price per share  
            signal_confidence: Signal confidence (0.0 to 1.0)
            
        Returns:
            Position sizing result dictionary
        """
        try:
            result = self._position_sizer.calculate_position_size(
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
                "details": result.details
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position preview: {str(e)}")
            return {
                "quantity": 0,
                "position_value": 0.0,
                "risk_amount": 0.0,
                "risk_percentage": 0.0,
                "method_used": "error",
                "valid": False,
                "warnings": [str(e)],
                "details": {}
            }
    
    def calculate_stop_loss_preview(
        self,
        entry_price: float,
        signal_direction: str,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Calculate stop loss preview with current configuration
        
        Args:
            entry_price: Entry price per share
            signal_direction: 'BUY' or 'SELL'
            historical_data: Optional historical data for advanced methods
            
        Returns:
            Stop loss result dictionary
        """
        try:
            result = self._stop_loss_engine.calculate_stop_loss(
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
                "details": result.details
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss preview: {str(e)}")
            return {
                "stop_price": 0.0,
                "stop_percentage": 0.0,
                "risk_per_share": 0.0,
                "method_used": "error",
                "confidence": 0.0,
                "valid": False,
                "warnings": [str(e)],
                "details": {}
            }
    
    def compare_all_methods(
        self,
        entry_price: float,
        stop_loss_price: float,
        signal_confidence: float = 1.0
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare all position sizing methods with current configuration
        
        Args:
            entry_price: Entry price per share
            stop_loss_price: Stop loss price per share
            signal_confidence: Signal confidence (0.0 to 1.0)
            
        Returns:
            Dictionary comparing all methods
        """
        try:
            comparison = compare_position_sizing_methods(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                config=self._active_config,
                signal_confidence=signal_confidence
            )
            
            # Convert to serializable format
            result = {}
            for method, sizing_result in comparison.items():
                result[method] = {
                    "quantity": sizing_result.quantity,
                    "position_value": sizing_result.position_value,
                    "risk_amount": sizing_result.risk_amount,
                    "risk_percentage": sizing_result.risk_percentage,
                    "valid": sizing_result.valid,
                    "warnings": sizing_result.warnings
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error comparing methods: {str(e)}")
            return {}
    
    def get_configuration_options(self) -> Dict[str, List]:
        """
        Get all available configuration options for UI
        
        Returns:
            Dictionary of available options
        """
        return {
            "risk_percent_options": ConfigurationOptions.RISK_PERCENT_OPTIONS,
            "portfolio_percent_options": ConfigurationOptions.PORTFOLIO_PERCENT_OPTIONS,
            "stop_loss_percent_options": ConfigurationOptions.STOP_LOSS_PERCENT_OPTIONS,
            "atr_multiplier_options": ConfigurationOptions.ATR_MULTIPLIER_OPTIONS,
            "breakeven_trigger_options": ConfigurationOptions.BREAKEVEN_TRIGGER_OPTIONS,
            "partial_booking_ratio_options": ConfigurationOptions.PARTIAL_BOOKING_RATIO_OPTIONS,
            "partial_booking_percent_options": ConfigurationOptions.PARTIAL_BOOKING_PERCENT_OPTIONS,
            "fixed_quantity_options": ConfigurationOptions.FIXED_QUANTITY_OPTIONS,
            "position_sizing_methods": [method.value for method in PositionSizingMethod],
            "stop_loss_methods": [method.value for method in StopLossMethod],
            "gap_protection_levels": [level.value for level in GapProtectionLevel]
        }
    
    def export_configuration(self, filepath: str) -> bool:
        """
        Export current configuration to file
        
        Args:
            filepath: Path to export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = {
                "active_profile": self._active_profile_name,
                "configuration": self._active_config.to_dict(),
                "exported_at": pd.Timestamp.now().isoformat(),
                "version": "1.0"
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Configuration exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {str(e)}")
            return False
    
    def import_configuration(self, filepath: str) -> bool:
        """
        Import configuration from file
        
        Args:
            filepath: Path to import file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            config = RiskConfiguration.from_dict(import_data["configuration"])
            
            self._active_config = config
            self._active_profile_name = import_data.get("active_profile", "imported")
            
            # Reinitialize engines
            self._position_sizer = EnhancedPositionSizer(self._active_config)
            self._stop_loss_engine = AdvancedStopLossEngine(self._active_config)
            
            self.logger.info(f"Configuration imported from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing configuration: {str(e)}")
            return False
    
    def validate_configuration(self, config_dict: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate configuration parameters
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            Dictionary with validation errors (empty if valid)
        """
        errors = {"errors": [], "warnings": []}
        
        try:
            # Try to create configuration (this will run validation)
            RiskConfiguration.from_dict(config_dict)
            
        except ValueError as e:
            errors["errors"].append(str(e))
        except Exception as e:
            errors["errors"].append(f"Configuration validation error: {str(e)}")
        
        return errors
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get summary of current configuration
        
        Returns:
            Configuration summary dictionary
        """
        return {
            "profile_name": self._active_profile_name,
            "risk_management": {
                "risk_percent": self._active_config.risk_percent,
                "max_portfolio_risk": self._active_config.max_portfolio_risk,
                "max_daily_loss": self._active_config.max_daily_loss
            },
            "position_sizing": self._position_sizer.get_position_sizing_summary(),
            "stop_loss": self._stop_loss_engine.get_stop_loss_summary(),
            "trailing_stops": {
                "enabled": self._active_config.enable_trailing_stops,
                "breakeven_trigger": self._active_config.breakeven_trigger_ratio,
                "partial_booking_enabled": self._active_config.partial_booking_enabled,
                "partial_booking_ratio": self._active_config.partial_booking_ratio,
                "partial_booking_percent": self._active_config.partial_booking_percent
            },
            "gap_protection": {
                "level": self._active_config.gap_protection_level.value,
                "overnight_reduction": self._active_config.overnight_reduction_factor,
                "weekend_reduction": self._active_config.weekend_reduction_factor
            },
            "account": {
                "balance": self._active_config.account_balance,
                "currency": self._active_config.currency
            }
        }


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


if __name__ == "__main__":
    # Example usage and testing
    import pandas as pd
    
    print("🎯 Configuration Manager Test")
    print("=" * 50)
    
    # Initialize manager
    manager = ConfigurationManager("test_config")
    
    # Test profile switching
    print("\n📊 Testing Profile Switching:")
    for profile in ["conservative", "moderate", "aggressive"]:
        success = manager.set_profile(profile)
        if success:
            summary = manager.get_configuration_summary()
            print(f"{profile.title()}: {summary['risk_management']['risk_percent']}% risk")
    
    # Test custom configuration
    print("\n⚙️ Testing Custom Configuration:")
    updates = {
        "risk_percent": 2.5,
        "position_sizing_method": PositionSizingMethod.HYBRID,
        "stop_loss_percent": 1.8
    }
    manager.update_configuration(updates)
    
    # Test position preview
    print("\n💰 Testing Position Preview:")
    preview = manager.calculate_position_preview(2500.0, 2450.0, 0.8)
    print(f"Quantity: {preview['quantity']}, Risk: {preview['risk_percentage']:.2f}%")
    
    # Test saving custom profile
    print("\n💾 Testing Save/Load:")
    manager.save_custom_profile("my_strategy", "Custom strategy for testing")
    
    # Test getting all profiles
    print("\n📋 Available Profiles:")
    profiles = manager.get_all_profiles()
    for name, preview in profiles.items():
        print(f"{name}: {preview.risk_percent}% risk, {preview.stop_loss_method} stops")
    
    print("\n✅ Configuration manager tests completed!")