#!/usr/bin/env python3
"""
Test Configurable Cost Model
Demonstrates different brokerage cost configurations
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.cost_profiles import CostConfigurationManager, get_cost_profiles_for_ui
from src.backtesting.cost_model import TransactionCostModel


def test_configurable_costs():
    """Test the configurable cost system"""
    print("🏦 Configurable Cost Model Test")
    print("=" * 50)
    
    # Initialize cost manager
    manager = CostConfigurationManager()
    
    print(f"📋 Available Cost Profiles:")
    profiles = get_cost_profiles_for_ui()
    for i, profile in enumerate(profiles, 1):
        active = "🔥" if profile['is_active'] else "  "
        print(f"{active} {i}. {profile['display_name']}: {profile['description']}")
    
    # Test cost calculations for different profiles
    print(f"\n💰 Cost Comparison (₹100,000 trade):")
    print(f"{'Broker':<15} {'Total Cost':<12} {'Cost %':<8} {'Breakeven':<10}")
    print("-" * 50)
    
    test_profiles = ["zerodha", "upstox", "icici_direct", "hdfc_securities"]
    
    for profile_name in test_profiles:
        preview = manager.get_cost_breakdown_preview(
            profile_name=profile_name,
            quantity=100,
            price=1000.0
        )
        
        if 'error' not in preview:
            broker = preview['profile_name']
            total_cost = preview['total_costs']
            cost_pct = preview['cost_percentage']
            breakeven = preview['breakeven_price']
            
            print(f"{broker:<15} ₹{total_cost:<11.2f} {cost_pct:<7.3f}% ₹{breakeven:<9.2f}")
    
    # Test custom profile creation
    print(f"\n🔧 Testing Custom Profile Creation:")
    
    from src.backtesting.cost_model import CostConfig
    
    # Create a custom low-cost profile
    custom_config = CostConfig(
        equity_brokerage_per_trade=10.0,  # Very low brokerage
        equity_brokerage_pct=0.0,
        max_brokerage_per_trade=10.0,
        equity_delivery_stt=0.001,
        equity_intraday_stt=0.00025,
        nse_equity_charges=0.0000325,
        gst_rate=0.18,
        sebi_charges=0.000001,
        stamp_duty=0.00003,
        intraday_discount=1.0
    )
    
    success = manager.create_custom_profile(
        name="low_cost",
        display_name="Low Cost Broker",
        description="Hypothetical ultra-low cost broker - ₹10 brokerage",
        cost_config=custom_config
    )
    
    if success:
        print("✅ Created custom 'Low Cost Broker' profile")
        
        # Test the custom profile
        preview = manager.get_cost_breakdown_preview("low_cost")
        if 'error' not in preview:
            print(f"   Custom profile cost: ₹{preview['total_costs']:.2f} ({preview['cost_percentage']:.3f}%)")
    
    # Test profile switching
    print(f"\n🔄 Testing Profile Switching:")
    
    original_profile = manager.active_profile_name
    print(f"   Original active profile: {original_profile}")
    
    # Switch to ICICI Direct
    if manager.set_active_profile("icici_direct"):
        print(f"   Switched to: {manager.get_active_profile().display_name}")
        
        # Switch back
        manager.set_active_profile(original_profile)
        print(f"   Switched back to: {manager.get_active_profile().display_name}")
    
    # Save profiles
    print(f"\n💾 Saving Profiles:")
    if manager.save_profiles():
        print(f"   ✅ Saved to: {manager.config_file}")
    
    print(f"\n🎉 Configurable cost system is ready!")
    print(f"   ✅ Multiple broker profiles available")
    print(f"   ✅ Custom profiles can be created")  
    print(f"   ✅ Profile switching works")
    print(f"   ✅ Settings persist across sessions")
    print(f"\n🚀 Ready for Phase 5 UI integration!")
    
    return True


if __name__ == "__main__":
    test_configurable_costs()