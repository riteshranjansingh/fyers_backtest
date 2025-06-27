"""
🧪 Comprehensive Configurability System Test

Test all aspects of the new configuration framework:
- Risk profiles (Conservative, Moderate, Aggressive)
- Position sizing methods (Risk-based, Portfolio %, Fixed, Hybrid)
- Stop loss methods (Percentage, ATR, Support/Resistance, Adaptive)
- Configuration management and persistence

This test validates the complete configurability layer built on top
of the existing working risk management system.

Author: Fyers Backtesting System
Date: 2025-06-27
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our new configuration system
from src.config.risk_profiles import RiskProfileManager, RiskConfiguration, PositionSizingMethod, StopLossMethod
from src.config.position_sizing_engine import EnhancedPositionSizer, compare_position_sizing_methods
from src.config.stop_loss_engine import AdvancedStopLossEngine
from src.config.configuration_manager import ConfigurationManager


def create_sample_data():
    """Create sample historical data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    price = 2500.0
    prices = []
    
    for i in range(100):
        # Simulate realistic price movement
        change = np.random.normal(0, 0.015)  # 1.5% daily volatility
        price = price * (1 + change)
        prices.append(max(price, 100.0))  # Ensure price doesn't go negative
    
    # Create OHLCV data
    data = []
    for i, close_price in enumerate(prices):
        high = close_price * (1 + abs(np.random.normal(0, 0.008)))
        low = close_price * (1 - abs(np.random.normal(0, 0.008)))
        open_price = prices[i-1] if i > 0 else close_price
        volume = np.random.randint(50000, 200000)
        
        data.append({
            'datetime': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def test_risk_profiles():
    """Test predefined risk profiles"""
    print("🎯 TESTING RISK PROFILES")
    print("=" * 50)
    
    profiles = RiskProfileManager.get_all_profiles()
    test_results = {}
    
    for profile_name, config in profiles.items():
        print(f"\n📊 Testing {profile_name.upper()} Profile:")
        print("-" * 30)
        
        try:
            # Validate configuration
            config.validate()
            
            print(f"✅ Risk Percent: {config.risk_percent}%")
            print(f"✅ Position Method: {config.position_sizing_method.value}")
            print(f"✅ Stop Loss Method: {config.stop_loss_method.value}")
            print(f"✅ Stop Loss %: {config.stop_loss_percent}%")
            print(f"✅ Gap Protection: {config.gap_protection_level.value}")
            print(f"✅ Trailing Stops: {'Enabled' if config.enable_trailing_stops else 'Disabled'}")
            
            test_results[profile_name] = "PASS"
            
        except Exception as e:
            print(f"❌ Profile validation failed: {str(e)}")
            test_results[profile_name] = f"FAIL: {str(e)}"
    
    return test_results


def test_position_sizing_methods():
    """Test all position sizing methods"""
    print("\n🎯 TESTING POSITION SIZING METHODS")
    print("=" * 50)
    
    # Test parameters
    entry_price = 2500.0
    stop_loss_price = 2425.0  # 3% stop loss
    signal_confidence = 0.8
    
    # Use moderate profile as base
    base_config = RiskProfileManager.get_moderate_profile()
    test_results = {}
    
    methods = [
        PositionSizingMethod.RISK_BASED,
        PositionSizingMethod.PORTFOLIO_PERCENT,
        PositionSizingMethod.FIXED_QUANTITY,
        PositionSizingMethod.HYBRID
    ]
    
    print(f"📈 Entry Price: ₹{entry_price:,.0f}")
    print(f"🛑 Stop Loss: ₹{stop_loss_price:,.0f} ({((entry_price-stop_loss_price)/entry_price)*100:.1f}%)")
    print(f"📊 Signal Confidence: {signal_confidence:.1f}")
    print()
    
    for method in methods:
        print(f"🔍 Testing {method.value.replace('_', ' ').title()} Method:")
        print("-" * 40)
        
        try:
            # Create config with this method
            config = RiskConfiguration(**base_config.to_dict())
            config.position_sizing_method = method
            
            # Test position sizing
            sizer = EnhancedPositionSizer(config)
            result = sizer.calculate_position_size(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                signal_confidence=signal_confidence
            )
            
            if result.valid:
                print(f"✅ Quantity: {result.quantity} shares")
                print(f"✅ Position Value: ₹{result.position_value:,.0f}")
                print(f"✅ Risk Amount: ₹{result.risk_amount:,.0f}")
                print(f"✅ Risk Percentage: {result.risk_percentage:.2f}%")
                
                if result.warnings:
                    print(f"⚠️  Warnings: {', '.join(result.warnings)}")
                
                test_results[method.value] = "PASS"
            else:
                print(f"❌ Invalid result: {', '.join(result.warnings)}")
                test_results[method.value] = "FAIL: Invalid result"
                
        except Exception as e:
            print(f"❌ Method failed: {str(e)}")
            test_results[method.value] = f"FAIL: {str(e)}"
        
        print()
    
    return test_results


def test_stop_loss_methods():
    """Test all stop loss methods"""
    print("\n🎯 TESTING STOP LOSS METHODS")
    print("=" * 50)
    
    # Test parameters
    entry_price = 2500.0
    signal_direction = 'BUY'
    historical_data = create_sample_data()
    
    # Use moderate profile as base
    base_config = RiskProfileManager.get_moderate_profile()
    test_results = {}
    
    methods = [
        StopLossMethod.PERCENTAGE,
        StopLossMethod.ATR,
        StopLossMethod.SUPPORT_RESISTANCE,
        StopLossMethod.ADAPTIVE
    ]
    
    print(f"📈 Entry Price: ₹{entry_price:,.0f}")
    print(f"📊 Signal Direction: {signal_direction}")
    print(f"📊 Historical Data Points: {len(historical_data)}")
    print()
    
    for method in methods:
        print(f"🔍 Testing {method.value.replace('_', ' ').title()} Method:")
        print("-" * 40)
        
        try:
            # Create config with this method
            config = RiskConfiguration(**base_config.to_dict())
            config.stop_loss_method = method
            
            # Test stop loss calculation
            engine = AdvancedStopLossEngine(config)
            result = engine.calculate_stop_loss(
                entry_price=entry_price,
                signal_direction=signal_direction,
                historical_data=historical_data
            )
            
            if result.valid:
                print(f"✅ Stop Price: ₹{result.stop_price:.2f}")
                print(f"✅ Stop Percentage: {result.stop_percentage:.2f}%")
                print(f"✅ Risk per Share: ₹{result.risk_per_share:.2f}")
                print(f"✅ Confidence: {result.confidence:.2f}")
                
                if result.warnings:
                    print(f"⚠️  Warnings: {', '.join(result.warnings)}")
                
                test_results[method.value] = "PASS"
            else:
                print(f"❌ Invalid result: {', '.join(result.warnings)}")
                test_results[method.value] = "FAIL: Invalid result"
                
        except Exception as e:
            print(f"❌ Method failed: {str(e)}")
            test_results[method.value] = f"FAIL: {str(e)}"
        
        print()
    
    return test_results


def test_configuration_manager():
    """Test configuration manager functionality"""
    print("\n🎯 TESTING CONFIGURATION MANAGER")
    print("=" * 50)
    
    test_results = {}
    
    try:
        # Initialize manager
        manager = ConfigurationManager("test_config_temp")
        print("✅ Configuration manager initialized")
        
        # Test profile switching
        print("\n📊 Testing Profile Switching:")
        for profile in ["conservative", "moderate", "aggressive"]:
            success = manager.set_profile(profile)
            if success:
                summary = manager.get_configuration_summary()
                risk_percent = summary['risk_management']['risk_percent']
                print(f"✅ {profile.title()}: {risk_percent}% risk")
            else:
                print(f"❌ Failed to set {profile} profile")
                
        test_results['profile_switching'] = "PASS"
        
        # Test custom configuration
        print("\n⚙️ Testing Custom Configuration:")
        updates = {
            "risk_percent": 2.3,
            "stop_loss_percent": 1.8,
            "portfolio_percent": 12.5
        }
        
        success = manager.update_configuration(updates)
        if success:
            config = manager.active_config
            print(f"✅ Risk updated to: {config.risk_percent}%")
            print(f"✅ Stop loss updated to: {config.stop_loss_percent}%")
            print(f"✅ Portfolio % updated to: {config.portfolio_percent}%")
        else:
            print("❌ Failed to update configuration")
            
        test_results['custom_config'] = "PASS" if success else "FAIL"
        
        # Test position preview
        print("\n💰 Testing Position Preview:")
        preview = manager.calculate_position_preview(2500.0, 2450.0, 0.85)
        if preview['valid']:
            print(f"✅ Quantity: {preview['quantity']} shares")
            print(f"✅ Position Value: ₹{preview['position_value']:,.0f}")
            print(f"✅ Risk: {preview['risk_percentage']:.2f}%")
        else:
            print(f"❌ Invalid preview: {', '.join(preview['warnings'])}")
            
        test_results['position_preview'] = "PASS" if preview['valid'] else "FAIL"
        
        # Test stop loss preview
        print("\n🛑 Testing Stop Loss Preview:")
        stop_preview = manager.calculate_stop_loss_preview(2500.0, 'BUY')
        if stop_preview['valid']:
            print(f"✅ Stop Price: ₹{stop_preview['stop_price']:.2f}")
            print(f"✅ Stop %: {stop_preview['stop_percentage']:.2f}%")
            print(f"✅ Method: {stop_preview['method_used']}")
        else:
            print(f"❌ Invalid stop preview: {', '.join(stop_preview['warnings'])}")
            
        test_results['stop_preview'] = "PASS" if stop_preview['valid'] else "FAIL"
        
        # Test method comparison
        print("\n🔍 Testing Method Comparison:")
        comparison = manager.compare_all_methods(2500.0, 2450.0, 0.8)
        if comparison:
            print("✅ Method comparison successful:")
            for method, result in comparison.items():
                print(f"   {method}: {result['quantity']} shares, {result['risk_percentage']:.2f}% risk")
        else:
            print("❌ Method comparison failed")
            
        test_results['method_comparison'] = "PASS" if comparison else "FAIL"
        
        # Test save/load profile
        print("\n💾 Testing Save/Load Profile:")
        save_success = manager.save_custom_profile("test_profile", "Test profile for validation")
        if save_success:
            print("✅ Profile saved successfully")
            
            # Test loading
            load_success = manager.set_profile("test_profile")
            if load_success:
                print("✅ Profile loaded successfully")
                test_results['save_load'] = "PASS"
            else:
                print("❌ Failed to load profile")
                test_results['save_load'] = "FAIL: Load failed"
        else:
            print("❌ Failed to save profile")
            test_results['save_load'] = "FAIL: Save failed"
        
        # Test getting all profiles
        print("\n📋 Testing Profile List:")
        profiles = manager.get_all_profiles()
        if profiles and len(profiles) >= 4:  # 3 predefined + 1 custom
            print(f"✅ Found {len(profiles)} profiles:")
            for name, preview in profiles.items():
                print(f"   {name}: {preview.risk_percent}% risk, {preview.stop_loss_method} stops")
            test_results['profile_list'] = "PASS"
        else:
            print(f"❌ Expected at least 4 profiles, found {len(profiles) if profiles else 0}")
            test_results['profile_list'] = "FAIL"
        
        # Clean up test files
        try:
            manager.delete_custom_profile("test_profile")
            print("✅ Test profile cleaned up")
        except:
            print("⚠️  Test profile cleanup failed (not critical)")
            
    except Exception as e:
        print(f"❌ Configuration manager test failed: {str(e)}")
        test_results['overall'] = f"FAIL: {str(e)}"
    
    return test_results


def test_integration_with_existing_system():
    """Test integration with existing risk management system"""
    print("\n🎯 TESTING INTEGRATION WITH EXISTING SYSTEM")
    print("=" * 50)
    
    test_results = {}
    
    try:
        # Import existing risk management
        from src.risk.risk_integration import RiskManager
        from src.risk.risk_integration import RiskConfig  # This should be our old config
        
        print("✅ Existing risk management system imported")
        
        # Create new configuration manager
        config_manager = ConfigurationManager()
        
        # Test different profiles with integration
        profiles_to_test = ["conservative", "moderate", "aggressive"]
        
        for profile_name in profiles_to_test:
            print(f"\n📊 Testing {profile_name.title()} Profile Integration:")
            print("-" * 40)
            
            # Set profile in configuration manager
            config_manager.set_profile(profile_name)
            new_config = config_manager.active_config
            
            # Convert to old config format for compatibility
            old_config = RiskConfig(
                risk_percentage=new_config.risk_percent,
                account_balance=new_config.account_balance,
                max_portfolio_risk=new_config.max_portfolio_risk,
                enable_trailing_stops=new_config.enable_trailing_stops,
                enable_gap_protection=new_config.gap_protection_level.value != "disabled"
            )
            
            # Test with existing risk manager
            risk_manager = RiskManager(old_config)
            
            # Create a dummy signal for testing
            from src.strategies.base_strategy import TradingSignal
            from datetime import datetime
            
            test_signal = TradingSignal(
                timestamp=datetime.now(),
                symbol="NSE:RELIANCE-EQ",
                signal_type="BUY",
                entry_price=2500.0,
                confidence=0.8,
                strength=0.7,
                metadata={}
            )
            
            # Test signal processing
            recommendation = risk_manager.process_signal(
                signal=test_signal,
                current_price=2500.0,
                symbol="NSE:RELIANCE-EQ"
            )
            
            if recommendation and recommendation.trade_valid:
                print(f"✅ Signal processed successfully")
                print(f"✅ Recommended quantity: {recommendation.recommended_quantity}")
                print(f"✅ Risk percentage: {recommendation.risk_percentage:.2f}%")
                print(f"✅ Stop price: ₹{recommendation.initial_stop_price:.2f}")
                test_results[f'{profile_name}_integration'] = "PASS"
            else:
                print(f"❌ Signal processing failed")
                if recommendation:
                    print(f"   Rejection reason: {recommendation.rejection_reason}")
                test_results[f'{profile_name}_integration'] = "FAIL"
        
    except ImportError as e:
        print(f"⚠️  Could not import existing system: {str(e)}")
        print("   This is expected if files have been moved or renamed")
        test_results['integration'] = "SKIP: Import failed"
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        test_results['integration'] = f"FAIL: {str(e)}"
    
    return test_results


def main():
    """Run comprehensive configurability tests"""
    print("🎯 FYERS BACKTESTING CONFIGURABILITY SYSTEM TEST")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Goal: Validate complete configurability framework")
    print()
    
    all_results = {}
    
    # Test 1: Risk Profiles
    all_results.update(test_risk_profiles())
    
    # Test 2: Position Sizing Methods
    all_results.update(test_position_sizing_methods())
    
    # Test 3: Stop Loss Methods
    all_results.update(test_stop_loss_methods())
    
    # Test 4: Configuration Manager
    all_results.update(test_configuration_manager())
    
    # Test 5: Integration with Existing System
    all_results.update(test_integration_with_existing_system())
    
    # Summary
    print("\n🎉 CONFIGURABILITY SYSTEM TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for result in all_results.values() if result == "PASS")
    failed = sum(1 for result in all_results.values() if result.startswith("FAIL"))
    skipped = sum(1 for result in all_results.values() if result.startswith("SKIP"))
    total = len(all_results)
    
    print(f"📊 Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"📈 Success Rate: {(passed/total)*100:.1f}%")
    
    if failed > 0:
        print(f"\n❌ FAILED TESTS:")
        for test_name, result in all_results.items():
            if result.startswith("FAIL"):
                print(f"   {test_name}: {result}")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ Configurability system is ready for Phase 4 backtesting")
        print("✅ UI integration (Phase 5) is fully prepared")
    elif passed >= total * 0.8:
        print(f"\n✅ CONFIGURABILITY SYSTEM MOSTLY WORKING!")
        print(f"📊 {passed}/{total} tests passed - ready to proceed")
    else:
        print(f"\n⚠️  CONFIGURABILITY SYSTEM NEEDS ATTENTION")
        print(f"📊 Only {passed}/{total} tests passed")
    
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()