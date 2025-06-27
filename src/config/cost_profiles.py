"""
Cost Configuration Profiles - Configurable transaction cost settings
Allows users to customize brokerage rates, exchange charges, and other costs
"""
import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from ..backtesting.cost_model import CostConfig

logger = logging.getLogger(__name__)


@dataclass
class BrokerProfile:
    """Predefined broker cost profiles"""
    name: str
    display_name: str
    description: str
    cost_config: CostConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'cost_config': asdict(self.cost_config)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BrokerProfile':
        """Create from dictionary"""
        cost_config = CostConfig(**data['cost_config'])
        return cls(
            name=data['name'],
            display_name=data['display_name'],
            description=data['description'],
            cost_config=cost_config
        )


class CostConfigurationManager:
    """
    Manages cost configuration profiles with save/load capability
    """
    
    def __init__(self, config_directory: str = "config/cost_profiles"):
        """
        Initialize cost configuration manager
        
        Args:
            config_directory: Directory to store cost profiles
        """
        self.config_directory = config_directory
        self.config_file = os.path.join(config_directory, "cost_profiles.json")
        
        # Create directory if it doesn't exist
        Path(config_directory).mkdir(parents=True, exist_ok=True)
        
        # Load existing profiles or create defaults
        self.profiles = self._load_profiles()
        self.active_profile_name = "zerodha"  # Default
        
        logger.info(f"Cost configuration manager initialized with {len(self.profiles)} profiles")
    
    def _create_default_profiles(self) -> Dict[str, BrokerProfile]:
        """Create default broker profiles"""
        profiles = {}
        
        # Zerodha (Discount Broker)
        profiles["zerodha"] = BrokerProfile(
            name="zerodha",
            display_name="Zerodha",
            description="Popular discount broker - ₹20 flat brokerage",
            cost_config=CostConfig(
                equity_brokerage_per_trade=20.0,
                equity_brokerage_pct=0.0,
                max_brokerage_per_trade=20.0,
                equity_delivery_stt=0.001,
                equity_intraday_stt=0.00025,
                nse_equity_charges=0.0000325,
                gst_rate=0.18,
                sebi_charges=0.000001,
                stamp_duty=0.00003,
                intraday_discount=1.0
            )
        )
        
        # Upstox (Discount Broker)
        profiles["upstox"] = BrokerProfile(
            name="upstox",
            display_name="Upstox",
            description="Discount broker - ₹20 flat brokerage",
            cost_config=CostConfig(
                equity_brokerage_per_trade=20.0,
                equity_brokerage_pct=0.0,
                max_brokerage_per_trade=20.0,
                equity_delivery_stt=0.001,
                equity_intraday_stt=0.00025,
                nse_equity_charges=0.0000325,
                gst_rate=0.18,
                sebi_charges=0.000001,
                stamp_duty=0.00003,
                intraday_discount=1.0
            )
        )
        
        # Angel One (Discount Broker)
        profiles["angel_one"] = BrokerProfile(
            name="angel_one",
            display_name="Angel One",
            description="Discount broker - ₹20 flat brokerage",
            cost_config=CostConfig(
                equity_brokerage_per_trade=20.0,
                equity_brokerage_pct=0.0,
                max_brokerage_per_trade=20.0,
                equity_delivery_stt=0.001,
                equity_intraday_stt=0.00025,
                nse_equity_charges=0.0000325,
                gst_rate=0.18,
                sebi_charges=0.000001,
                stamp_duty=0.00003,
                intraday_discount=1.0
            )
        )
        
        # ICICI Direct (Full Service)
        profiles["icici_direct"] = BrokerProfile(
            name="icici_direct",
            display_name="ICICI Direct",
            description="Full-service broker - 0.55% brokerage",
            cost_config=CostConfig(
                equity_brokerage_per_trade=100.0,
                equity_brokerage_pct=0.0055,  # 0.55%
                max_brokerage_per_trade=500.0,
                equity_delivery_stt=0.001,
                equity_intraday_stt=0.00025,
                nse_equity_charges=0.0000325,
                gst_rate=0.18,
                sebi_charges=0.000001,
                stamp_duty=0.00003,
                intraday_discount=0.5
            )
        )
        
        # HDFC Securities (Full Service)
        profiles["hdfc_securities"] = BrokerProfile(
            name="hdfc_securities",
            display_name="HDFC Securities",
            description="Full-service broker - 0.5% brokerage",
            cost_config=CostConfig(
                equity_brokerage_per_trade=100.0,
                equity_brokerage_pct=0.005,  # 0.5%
                max_brokerage_per_trade=2500.0,
                equity_delivery_stt=0.001,
                equity_intraday_stt=0.00025,
                nse_equity_charges=0.0000325,
                gst_rate=0.18,
                sebi_charges=0.000001,
                stamp_duty=0.00003,
                intraday_discount=0.5
            )
        )
        
        # Custom template
        profiles["custom"] = BrokerProfile(
            name="custom",
            display_name="Custom Broker",
            description="User-customizable cost structure",
            cost_config=CostConfig(
                equity_brokerage_per_trade=20.0,
                equity_brokerage_pct=0.0,
                max_brokerage_per_trade=20.0,
                equity_delivery_stt=0.001,
                equity_intraday_stt=0.00025,
                nse_equity_charges=0.0000325,
                gst_rate=0.18,
                sebi_charges=0.000001,
                stamp_duty=0.00003,
                intraday_discount=1.0
            )
        )
        
        return profiles
    
    def _load_profiles(self) -> Dict[str, BrokerProfile]:
        """Load profiles from file or create defaults"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                profiles = {}
                for profile_data in data.get('profiles', []):
                    profile = BrokerProfile.from_dict(profile_data)
                    profiles[profile.name] = profile
                
                logger.info(f"Loaded {len(profiles)} cost profiles from {self.config_file}")
                return profiles
            else:
                logger.info("No existing cost profiles found, creating defaults")
                return self._create_default_profiles()
                
        except Exception as e:
            logger.error(f"Error loading cost profiles: {str(e)}, using defaults")
            return self._create_default_profiles()
    
    def save_profiles(self) -> bool:
        """Save current profiles to file"""
        try:
            data = {
                'version': '1.0',
                'active_profile': self.active_profile_name,
                'profiles': [profile.to_dict() for profile in self.profiles.values()]
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.profiles)} cost profiles to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving cost profiles: {str(e)}")
            return False
    
    def get_profile(self, profile_name: str) -> Optional[BrokerProfile]:
        """Get a specific broker profile"""
        return self.profiles.get(profile_name)
    
    def get_active_profile(self) -> BrokerProfile:
        """Get the currently active profile"""
        return self.profiles.get(self.active_profile_name, self.profiles["zerodha"])
    
    def set_active_profile(self, profile_name: str) -> bool:
        """Set the active profile"""
        if profile_name in self.profiles:
            self.active_profile_name = profile_name
            logger.info(f"Active cost profile changed to: {profile_name}")
            return True
        else:
            logger.error(f"Profile '{profile_name}' not found")
            return False
    
    def create_custom_profile(
        self,
        name: str,
        display_name: str,
        description: str,
        cost_config: CostConfig
    ) -> bool:
        """Create a new custom profile"""
        try:
            profile = BrokerProfile(
                name=name,
                display_name=display_name,
                description=description,
                cost_config=cost_config
            )
            
            self.profiles[name] = profile
            logger.info(f"Created custom cost profile: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating custom profile: {str(e)}")
            return False
    
    def update_profile_costs(
        self,
        profile_name: str,
        cost_updates: Dict[str, Any]
    ) -> bool:
        """Update cost parameters for a profile"""
        try:
            if profile_name not in self.profiles:
                logger.error(f"Profile '{profile_name}' not found")
                return False
            
            profile = self.profiles[profile_name]
            
            # Update cost config parameters
            for key, value in cost_updates.items():
                if hasattr(profile.cost_config, key):
                    setattr(profile.cost_config, key, value)
                    logger.debug(f"Updated {profile_name}.{key} = {value}")
                else:
                    logger.warning(f"Invalid cost parameter: {key}")
            
            logger.info(f"Updated cost parameters for profile: {profile_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating profile costs: {str(e)}")
            return False
    
    def get_profile_list(self) -> List[Dict[str, str]]:
        """Get list of available profiles for UI"""
        return [
            {
                'name': profile.name,
                'display_name': profile.display_name,
                'description': profile.description,
                'is_active': profile.name == self.active_profile_name
            }
            for profile in self.profiles.values()
        ]
    
    def get_cost_breakdown_preview(
        self,
        profile_name: str,
        quantity: int = 100,
        price: float = 1000.0
    ) -> Dict[str, Any]:
        """Get cost breakdown preview for a profile"""
        try:
            profile = self.get_profile(profile_name)
            if not profile:
                return {'error': f'Profile {profile_name} not found'}
            
            from ..backtesting.cost_model import TransactionCostModel
            
            cost_model = TransactionCostModel(profile.cost_config)
            
            # Calculate round trip costs
            result = cost_model.calculate_round_trip_costs(
                quantity=quantity,
                entry_price=price,
                exit_price=price,
                is_intraday=False
            )
            
            if result['success']:
                return {
                    'profile_name': profile.display_name,
                    'trade_value': quantity * price,
                    'total_costs': result['total_costs'],
                    'cost_percentage': result['cost_percentage'],
                    'breakeven_price': result['breakeven_price'],
                    'entry_costs': result['entry_costs']['breakdown'],
                    'exit_costs': result['exit_costs']['breakdown']
                }
            else:
                return {'error': 'Cost calculation failed'}
                
        except Exception as e:
            logger.error(f"Error getting cost preview: {str(e)}")
            return {'error': str(e)}
    
    def export_profile(self, profile_name: str, filename: Optional[str] = None) -> str:
        """Export a profile to JSON file"""
        try:
            profile = self.get_profile(profile_name)
            if not profile:
                raise ValueError(f"Profile '{profile_name}' not found")
            
            if not filename:
                filename = f"cost_profile_{profile_name}.json"
            
            filepath = os.path.join(self.config_directory, filename)
            
            with open(filepath, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Exported profile '{profile_name}' to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting profile: {str(e)}")
            return ""
    
    def import_profile(self, filepath: str) -> bool:
        """Import a profile from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            profile = BrokerProfile.from_dict(data)
            self.profiles[profile.name] = profile
            
            logger.info(f"Imported profile '{profile.name}' from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing profile: {str(e)}")
            return False


# Global cost configuration manager
_cost_config_manager = None

def get_cost_config_manager() -> CostConfigurationManager:
    """Get global cost configuration manager instance"""
    global _cost_config_manager
    if _cost_config_manager is None:
        _cost_config_manager = CostConfigurationManager()
    return _cost_config_manager


# Convenience functions for easy integration
def get_active_cost_config() -> CostConfig:
    """Get currently active cost configuration"""
    manager = get_cost_config_manager()
    return manager.get_active_profile().cost_config


def set_cost_profile(profile_name: str) -> bool:
    """Set active cost profile"""
    manager = get_cost_config_manager()
    return manager.set_active_profile(profile_name)


def create_cost_model_with_profile(profile_name: str = None):
    """Create cost model with specified profile"""
    from ..backtesting.cost_model import TransactionCostModel
    
    manager = get_cost_config_manager()
    
    if profile_name:
        profile = manager.get_profile(profile_name)
        if not profile:
            raise ValueError(f"Cost profile '{profile_name}' not found")
        cost_config = profile.cost_config
    else:
        cost_config = manager.get_active_profile().cost_config
    
    return TransactionCostModel(cost_config)


def get_cost_profiles_for_ui() -> List[Dict[str, str]]:
    """Get cost profiles formatted for UI display"""
    manager = get_cost_config_manager()
    return manager.get_profile_list()


if __name__ == "__main__":
    # Test cost configuration manager
    print("🏦 Cost Configuration Manager Test")
    print("=" * 40)
    
    # Initialize manager
    manager = CostConfigurationManager()
    
    # Show available profiles
    print(f"\n📋 Available Profiles ({len(manager.profiles)}):")
    for name, profile in manager.profiles.items():
        indicator = "🔥" if name == manager.active_profile_name else "  "
        print(f"{indicator} {profile.display_name}: {profile.description}")
    
    # Show cost breakdown for active profile
    print(f"\n💰 Cost Breakdown (Active: {manager.get_active_profile().display_name}):")
    preview = manager.get_cost_breakdown_preview(manager.active_profile_name)
    if 'error' not in preview:
        print(f"   Trade Value: ₹{preview['trade_value']:,.0f}")
        print(f"   Total Costs: ₹{preview['total_costs']:.2f}")
        print(f"   Cost %: {preview['cost_percentage']:.3f}%")
        print(f"   Breakeven: ₹{preview['breakeven_price']:.2f}")
    
    # Save profiles
    if manager.save_profiles():
        print(f"\n✅ Profiles saved to: {manager.config_file}")
    
    print("\n🎉 Cost configuration system ready for Phase 5 UI!")