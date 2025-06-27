#!/usr/bin/env python3
"""
Phase 4 Backtesting Engine Test
Test complete backtesting workflow with real NIFTY data
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_backtesting_system():
    """Test the complete backtesting system"""
    print("🚀 Phase 4 Backtesting Engine Test")
    print("=" * 50)
    
    try:
        # Import modules with absolute imports from src
        from src.strategies.ma_crossover_strategy import MACrossoverStrategy, MAConfig
        from src.backtesting import BacktestEngine, BacktestConfig, quick_backtest
        from src.config.integration_bridge import create_risk_manager
        
        print("✅ All modules imported successfully")
        
        # Test 1: Load real NIFTY data
        print("\n📊 Test 1: Loading Real NIFTY Data")
        try:
            # Try to load cached NIFTY data
            data_file = "data/raw/1d/NSE_NIFTY50_INDEX_20000101_20250625.csv"
            
            if os.path.exists(data_file):
                print(f"📁 Loading data from: {data_file}")
                data = pd.read_csv(data_file)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
                data = data.sort_index()
                
                # Use last 1000 points for testing
                data = data.tail(1000)
                print(f"✅ Loaded {len(data)} data points from {data.index[0]} to {data.index[-1]}")
                
            else:
                print("⚠️  Real data file not found, creating synthetic data")
                # Create synthetic NIFTY-like data for testing
                dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
                np.random.seed(42)
                
                price = 18000  # Starting price
                prices = [price]
                
                for _ in range(len(dates) - 1):
                    change = np.random.normal(0, 0.015)  # 1.5% daily volatility
                    price *= (1 + change)
                    prices.append(price)
                
                data = pd.DataFrame({
                    'open': prices,
                    'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                    'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                    'close': prices,
                    'volume': [np.random.randint(10000000, 50000000) for _ in prices]
                }, index=dates)
                
                print(f"✅ Created synthetic data: {len(data)} points from {data.index[0]} to {data.index[-1]}")
            
        except Exception as e:
            print(f"❌ Data loading failed: {str(e)}")
            return False
        
        # Test 2: Create strategy
        print("\n🎯 Test 2: Creating MA Crossover Strategy")
        try:
            from src.strategies.ma_crossover_strategy import MAType
            
            ma_config = MAConfig(
                fast_ma_type=MAType.EMA,
                fast_period=9,
                slow_ma_type=MAType.EMA, 
                slow_period=21
            )
            
            strategy = MACrossoverStrategy(config=ma_config)
            
            print(f"✅ Strategy created: {strategy.name}")
            
        except Exception as e:
            print(f"❌ Strategy creation failed: {str(e)}")
            return False
        
        # Test 3: Create risk manager
        print("\n⚠️  Test 3: Creating Risk Manager")
        try:
            risk_manager = create_risk_manager("moderate")
            print(f"✅ Risk manager created with moderate profile")
            
        except Exception as e:
            print(f"❌ Risk manager creation failed: {str(e)}")
            return False
        
        # Test 4: Quick backtest
        print("\n🔄 Test 4: Running Quick Backtest")
        try:
            results = quick_backtest(
                strategy=strategy,
                data=data,
                initial_capital=100000,
                risk_profile="moderate"
            )
            
            print("✅ Quick backtest completed!")
            print(f"📊 Results Summary:")
            print(f"   Total Return: {results.total_return_pct:.2f}%")
            print(f"   Total Trades: {results.total_trades}")
            print(f"   Win Rate: {results.win_rate:.2f}%")
            print(f"   Max Drawdown: {results.max_drawdown_pct:.2f}%")
            print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
            
        except Exception as e:
            print(f"❌ Quick backtest failed: {str(e)}")
            return False
        
        # Test 5: Full backtest with custom configuration
        print("\n🎛️  Test 5: Running Full Backtest with Custom Config")
        try:
            config = BacktestConfig(
                initial_capital=100000,
                risk_profile="moderate",
                enable_costs=True,
                enable_slippage=True,
                enable_logging=True,
                max_positions=3
            )
            
            engine = BacktestEngine(
                strategy=strategy,
                config=config,
                risk_manager=risk_manager
            )
            
            full_results = engine.run(data, symbol="NIFTY50")
            
            print("✅ Full backtest completed!")
            print(f"📊 Detailed Results:")
            
            summary = full_results.summary()
            for category, metrics in summary.items():
                print(f"\n   {category.upper()}:")
                for key, value in metrics.items():
                    print(f"     {key}: {value}")
            
        except Exception as e:
            print(f"❌ Full backtest failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 6: Trade history analysis
        print("\n📋 Test 6: Trade History Analysis")
        try:
            trade_history = full_results.trade_history
            
            if not trade_history.empty:
                print(f"✅ Trade history contains {len(trade_history)} trades")
                print(f"   Columns: {list(trade_history.columns)}")
                
                if len(trade_history) > 0:
                    print(f"   Sample trade:")
                    sample_trade = trade_history.iloc[0]
                    print(f"     Entry: {sample_trade.get('entry_date', 'N/A')} @ ₹{sample_trade.get('entry_price', 0):.2f}")
                    print(f"     Exit: {sample_trade.get('exit_date', 'N/A')} @ ₹{sample_trade.get('exit_price', 0):.2f}")
                    print(f"     P&L: ₹{sample_trade.get('net_pnl', 0):.2f}")
            else:
                print("⚠️  No trades in history (this might be normal with conservative strategy)")
            
        except Exception as e:
            print(f"❌ Trade history analysis failed: {str(e)}")
            return False
        
        # Test 7: Performance metrics validation
        print("\n📈 Test 7: Performance Metrics Validation")
        try:
            # Validate key metrics
            assert isinstance(full_results.total_return_pct, (int, float)), "Total return should be numeric"
            assert isinstance(full_results.max_drawdown_pct, (int, float)), "Max drawdown should be numeric"
            assert isinstance(full_results.sharpe_ratio, (int, float)), "Sharpe ratio should be numeric"
            assert full_results.initial_capital > 0, "Initial capital should be positive"
            assert full_results.final_capital > 0, "Final capital should be positive"
            
            print("✅ All performance metrics are valid")
            
            # Check equity curve
            if not full_results.equity_curve.empty:
                print(f"✅ Equity curve has {len(full_results.equity_curve)} points")
                print(f"   Starting equity: ₹{full_results.equity_curve.iloc[0]:,.0f}")
                print(f"   Ending equity: ₹{full_results.equity_curve.iloc[-1]:,.0f}")
            
        except Exception as e:
            print(f"❌ Performance metrics validation failed: {str(e)}")
            return False
        
        print("\n🎉 ALL TESTS PASSED! Phase 4 Backtesting Engine is working correctly!")
        print("\n📊 Final Summary:")
        print(f"   Strategy: {strategy.name}")
        print(f"   Data Points: {len(data)}")
        print(f"   Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Total Return: {full_results.total_return_pct:.2f}%")
        print(f"   CAGR: {full_results.cagr:.2f}%")
        print(f"   Max Drawdown: {full_results.max_drawdown_pct:.2f}%")
        print(f"   Sharpe Ratio: {full_results.sharpe_ratio:.2f}")
        print(f"   Total Trades: {full_results.total_trades}")
        print(f"   Win Rate: {full_results.win_rate:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    success = test_backtesting_system()
    
    if success:
        print("\n✅ Phase 4 backtesting engine is ready!")
        print("🚀 You can now proceed to Phase 5 (UI development)")
    else:
        print("\n❌ Phase 4 testing failed")
        print("🔧 Please check the errors above and fix them")
    
    return success


if __name__ == "__main__":
    main()