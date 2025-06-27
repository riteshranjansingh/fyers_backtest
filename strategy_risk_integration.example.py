"""
Strategy + Risk Management Integration Example
Demonstrates how Phase 3.2 (Strategies) + Phase 3.3 (Risk Management) work together
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append('.')

# Import Phase 3.2 (Strategies)
from src.strategies.ma_crossover_strategy import create_ma_strategy, MAConfig, MAType
from src.strategies.base_strategy import Signal

# Import Phase 3.3 (Risk Management)
from src.risk.risk_integration import create_risk_manager, RiskManagerConfig
from src.risk.position_sizer import RiskConfig
from src.risk.advanced_trailing import TrailingConfig
from src.risk import get_risk_profile

def create_sample_data(symbol: str = "NSE:RELIANCE-EQ", periods: int = 50) -> pd.DataFrame:
    """Create sample market data for demonstration"""
    dates = pd.date_range('2024-11-01', periods=periods, freq='D')
    np.random.seed(42)
    
    base_price = 2800  # Realistic RELIANCE price
    returns = np.random.randn(periods) * 0.02  # 2% daily volatility
    
    # Add trend for crossover signals
    trend = np.linspace(0, 0.1, periods)  # 10% uptrend
    combined_returns = returns + trend/periods
    
    price_series = base_price * np.exp(combined_returns.cumsum())
    
    data = pd.DataFrame({
        'open': price_series,
        'close': price_series * (1 + np.random.randn(periods) * 0.005),
        'high': 0,
        'low': 0,
        'volume': np.random.randint(1000000, 5000000, periods)
    }, index=dates)
    
    # Calculate realistic high/low
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.rand(periods) * 0.01)
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.rand(periods) * 0.01)
    
    return data

def demonstrate_complete_integration():
    """
    Complete demonstration of strategy + risk management integration
    """
    print("\n🎯 COMPLETE STRATEGY + RISK MANAGEMENT INTEGRATION")
    print("=" * 70)
    
    # Step 1: Setup
    print("\n📋 STEP 1: SETUP")
    print("-" * 30)
    
    # Account setup
    account_balance = 500000  # ₹5 lakh account
    risk_profile = "moderate"
    
    print(f"💰 Account Balance: ₹{account_balance:,}")
    print(f"⚖️  Risk Profile: {risk_profile}")
    
    # Create strategy (Phase 3.2)
    strategy = create_ma_strategy(
        fast_ma_type="EMA", fast_period=9,
        slow_ma_type="SMA", slow_period=21,
        enable_filtering=True,
        signal_strength_threshold=0.4
    )
    print(f"📈 Strategy: {strategy.name}")
    
    # Create risk manager (Phase 3.3)
    risk_manager = create_risk_manager(
        account_balance=account_balance,
        risk_percentage=2.0,  # 2% risk per trade
        risk_profile=risk_profile
    )
    print(f"🛡️  Risk Manager: {risk_profile} profile, 2% risk per trade")
    
    # Step 2: Market Analysis
    print("\n📊 STEP 2: MARKET ANALYSIS & SIGNAL GENERATION")
    print("-" * 50)
    
    # Get market data
    symbol = "NSE:RELIANCE-EQ"
    market_data = create_sample_data(symbol, 50)
    current_price = market_data['close'].iloc[-1]
    
    print(f"📊 Symbol: {symbol}")
    print(f"📅 Data Range: {market_data.index[0].date()} to {market_data.index[-1].date()}")
    print(f"💹 Current Price: ₹{current_price:.2f}")
    print(f"📈 Period Range: ₹{market_data['close'].min():.2f} - ₹{market_data['close'].max():.2f}")
    
    # Generate strategy signals
    print(f"\n🔄 Generating signals with {strategy.name}...")
    strategy_result = strategy.generate_signals(market_data)
    
    print(f"✅ Signal Generation Complete:")
    print(f"   Total Signals: {strategy_result.total_signals}")
    print(f"   Buy Signals: {strategy_result.buy_signals}")
    print(f"   Sell Signals: {strategy_result.sell_signals}")
    print(f"   Avg Confidence: {strategy_result.avg_confidence:.3f}")
    
    if strategy_result.total_signals == 0:
        print("❌ No signals generated. Creating sample signal for demonstration...")
        # Create a sample signal for demonstration
        sample_signal = Signal(
            timestamp=market_data.index[-1],
            signal_type='BUY',
            price=current_price,
            confidence=0.75,
            metadata={
                'strategy': strategy.name,
                'fast_ma_value': current_price * 1.01,
                'slow_ma_value': current_price * 0.99,
                'ma_separation_pct': 2.0
            }
        )
        latest_signal = sample_signal
    else:
        latest_signal = strategy_result.signals[-1]
    
    print(f"🎯 Latest Signal: {latest_signal.signal_type} at ₹{latest_signal.price:.2f} (Confidence: {latest_signal.confidence:.3f})")
    
    # Step 3: Risk Management Processing
    print("\n🛡️  STEP 3: RISK MANAGEMENT PROCESSING")
    print("-" * 45)
    
    print(f"🔄 Processing {latest_signal.signal_type} signal through risk management...")
    
    # Add some market context for realistic risk assessment
    market_context = {
        'volatility': 0.025,  # 2.5% daily volatility
        'atr': current_price * 0.02,  # 2% ATR
        'news_risk': 'normal',
        'volume_context': 'normal'
    }
    
    # Process signal with complete risk management
    recommendation = risk_manager.process_strategy_signal(
        signal=latest_signal,
        market_data=market_data,
        symbol=symbol,
        current_price=current_price,
        market_context=market_context
    )
    
    print(f"✅ Risk Management Analysis Complete:")
    print(f"   Trade Valid: {recommendation.trade_valid}")
    
    if not recommendation.trade_valid:
        print(f"   ❌ Rejection Reason: {recommendation.rejection_reason}")
        return
    
    print(f"   📊 Position Details:")
    print(f"     Recommended Quantity: {recommendation.recommended_quantity} shares")
    print(f"     Position Value: ₹{recommendation.position_value:,.0f}")
    print(f"     Risk Amount: ₹{recommendation.risk_amount:,.0f}")
    print(f"     Risk Percentage: {recommendation.risk_percentage:.2f}%")
    
    print(f"   🛑 Stop Loss:")
    print(f"     Initial Stop Price: ₹{recommendation.initial_stop_price:.2f}")
    print(f"     Stop Loss Method: {recommendation.stop_loss_method}")
    print(f"     Risk per Share: ₹{abs(current_price - recommendation.initial_stop_price):.2f}")
    
    print(f"   📈 Trailing Configuration:")
    print(f"     Trailing Enabled: {recommendation.enable_trailing}")
    if recommendation.enable_trailing:
        print(f"     Breakeven Trigger: {recommendation.trailing_config.breakeven_trigger_rr}:1 R:R")
        print(f"     Partial Book Trigger: {recommendation.trailing_config.partial_book_trigger_rr}:1 R:R")
        print(f"     Partial Book %: {recommendation.trailing_config.partial_book_percentage}%")
    
    print(f"   🔒 Gap Protection:")
    print(f"     Gap Adjusted Size: {recommendation.gap_adjusted_size} shares")
    print(f"     Protection Applied: {recommendation.gap_protection_applied}")
    
    print(f"   ⚠️  Risk Assessment:")
    print(f"     Overall Risk Score: {recommendation.overall_risk_score:.1f}/100")
    if recommendation.risk_warnings:
        print(f"     Warnings: {', '.join(recommendation.risk_warnings)}")
    else:
        print(f"     No warnings")
    
    # Step 4: Trade Execution
    print("\n💼 STEP 4: TRADE EXECUTION & SETUP")
    print("-" * 40)
    
    print(f"🔄 Executing trade recommendation...")
    execution_result = risk_manager.execute_trade_recommendation(
        recommendation=recommendation,
        execution_price=current_price
    )
    
    if execution_result['success']:
        print(f"✅ Trade Executed Successfully:")
        print(f"   Trade ID: {execution_result['trade_id']}")
        print(f"   Stop Loss ID: {execution_result['stop_id']}")
        print(f"   Trailing Stop ID: {execution_result['trailing_id']}")
        
        exec_details = execution_result['execution_details']
        print(f"   📋 Execution Summary:")
        print(f"     Symbol: {exec_details['symbol']}")
        print(f"     Quantity: {exec_details['quantity']} shares")
        print(f"     Entry Price: ₹{exec_details['price']:.2f}")
        print(f"     Stop Price: ₹{exec_details['stop_price']:.2f}")
        print(f"     Risk Amount: ₹{exec_details['risk_amount']:,.0f}")
        print(f"     Trailing Enabled: {exec_details['trailing_enabled']}")
    else:
        print(f"❌ Trade Execution Failed: {execution_result.get('error')}")
        return
    
    # Step 5: Position Management Simulation
    print("\n📊 STEP 5: POSITION MANAGEMENT SIMULATION")
    print("-" * 50)
    
    print(f"🔄 Simulating market movements and position updates...")
    
    # Simulate various price scenarios
    scenarios = [
        (current_price * 1.005, "Small favorable move (+0.5%)"),
        (current_price * 1.015, "1:1 Risk-Reward achieved (+1.5%)"),
        (current_price * 1.03, "2:1 Risk-Reward achieved (+3.0%)"),
        (current_price * 1.045, "Strong move (+4.5%)")
    ]
    
    for i, (new_price, description) in enumerate(scenarios):
        print(f"\n📈 Scenario {i+1}: {description}")
        print(f"   New Price: ₹{new_price:.2f}")
        
        # Update positions
        market_update = {symbol: new_price}
        update_results = risk_manager.update_positions(
            market_data=market_update,
            current_time=datetime.now() + timedelta(hours=i)
        )
        
        # Check trailing stop status
        if execution_result['trailing_id']:
            trailing_status = risk_manager.trailing_stop_manager.get_position_status(
                execution_result['trailing_id']
            )
            
            if trailing_status['exists']:
                print(f"   📊 Trailing Status:")
                print(f"     Stage: {trailing_status['stage']}")
                print(f"     Current Stop: ₹{trailing_status['current_stop']:.2f}")
                print(f"     R:R Ratio: {trailing_status['rr_ratio']:.2f}")
                print(f"     Unrealized P&L: ₹{trailing_status['unrealized_profit']:,.0f}")
                
                if trailing_status['partial_booked'] > 0:
                    print(f"     Partial Booked: {trailing_status['partial_booked']} shares")
                    print(f"     Realized P&L: ₹{trailing_status['realized_profit']:,.0f}")
        
        # Check for triggered actions
        if update_results['partial_bookings']:
            print(f"   🎯 Partial Booking Triggered!")
            for booking in update_results['partial_bookings']:
                print(f"     Booked: {booking['booking_details']['book_quantity']} shares")
                print(f"     Profit: ₹{booking['booking_details']['partial_profit']:,.0f}")
        
        if update_results['stop_hits']:
            print(f"   🛑 Stop Loss Hit!")
            break
    
    # Step 6: Portfolio Summary
    print("\n📈 STEP 6: PORTFOLIO SUMMARY")
    print("-" * 35)
    
    portfolio_summary = risk_manager.get_portfolio_summary()
    
    print(f"💼 Portfolio Overview:")
    print(f"   Active Positions: {portfolio_summary['portfolio_metrics']['active_positions']}")
    print(f"   Total Portfolio Risk: {portfolio_summary['portfolio_metrics']['total_risk']:.2f}%")
    print(f"   Total Position Value: ₹{portfolio_summary['portfolio_metrics'].get('total_position_value', 0):,.0f}")
    
    print(f"\n🛡️  Risk Management Status:")
    print(f"   Active Stop Losses: {portfolio_summary['active_stops']}")
    print(f"   Trailing Positions: {portfolio_summary['trailing_positions']}")
    
    config = portfolio_summary['risk_manager_config']
    print(f"\n⚙️  Configuration:")
    print(f"   Risk Profile: {config['risk_profile']}")
    print(f"   Max Portfolio Risk: {config['max_portfolio_risk']}%")
    print(f"   Components Active: {', '.join([k for k, v in config['components_enabled'].items() if v])}")
    
    print(f"\n🎉 INTEGRATION DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("✅ Strategy generated signals")
    print("✅ Risk management processed and approved trade")
    print("✅ Position sizing calculated (2% account risk)")
    print("✅ Stop losses set with gap protection")
    print("✅ Trailing stops configured (breakeven at 1:1, partial book at 2:1)")
    print("✅ Complete position management lifecycle demonstrated")

if __name__ == "__main__":
    demonstrate_complete_integration()