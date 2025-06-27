"""
Transaction Cost Model - Accurate Indian market transaction cost calculations
Includes brokerage, STT, exchange charges, GST, and regulatory fees
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketSegment(Enum):
    """Market segments with different cost structures"""
    EQUITY = "equity"
    FUTURES = "futures"
    OPTIONS = "options"
    CURRENCY = "currency"
    COMMODITY = "commodity"


class TradeType(Enum):
    """Trade types for cost calculation"""
    BUY = "buy"
    SELL = "sell"
    INTRADAY_BUY = "intraday_buy"
    INTRADAY_SELL = "intraday_sell"


@dataclass
class CostBreakdown:
    """Detailed breakdown of transaction costs"""
    brokerage: float = 0.0
    stt: float = 0.0
    exchange_charges: float = 0.0
    gst: float = 0.0
    sebi_charges: float = 0.0
    stamp_duty: float = 0.0
    total_cost: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'brokerage': self.brokerage,
            'stt': self.stt,
            'exchange_charges': self.exchange_charges,
            'gst': self.gst,
            'sebi_charges': self.sebi_charges,
            'stamp_duty': self.stamp_duty,
            'total_cost': self.total_cost
        }


@dataclass
class CostConfig:
    """Configuration for transaction cost calculation"""
    # Brokerage rates
    equity_brokerage_per_trade: float = 20.0  # ₹20 per trade (discount broker)
    equity_brokerage_pct: float = 0.0  # 0% for discount brokers
    max_brokerage_per_trade: float = 20.0
    
    # STT rates (as per Indian regulations)
    equity_delivery_stt: float = 0.001  # 0.1% on both buy and sell
    equity_intraday_stt: float = 0.00025  # 0.025% on sell side only
    futures_stt: float = 0.0001  # 0.01% on sell side
    options_stt: float = 0.00005  # 0.005% on sell side (premium)
    
    # Exchange charges
    nse_equity_charges: float = 0.0000325  # 0.00325%
    nse_futures_charges: float = 0.0000183  # 0.00183%
    nse_options_charges: float = 0.0003  # 0.03%
    
    # Regulatory charges
    gst_rate: float = 0.18  # 18% on (brokerage + exchange charges)
    sebi_charges: float = 0.000001  # ₹1 per ₹1 crore (0.0001%)
    stamp_duty: float = 0.00003  # 0.003% on buy side (Maharashtra)
    
    # Special rates
    intraday_discount: float = 0.5  # 50% discount on intraday brokerage
    
    def __post_init__(self):
        """Validate configuration"""
        if self.gst_rate < 0 or self.gst_rate > 1:
            raise ValueError(f"Invalid GST rate: {self.gst_rate}")


class TransactionCostModel:
    """
    Comprehensive transaction cost model for Indian markets
    Calculates accurate costs including all charges and taxes
    """
    
    def __init__(self, config: CostConfig = None, **kwargs):
        """
        Initialize cost model
        
        Args:
            config: Cost configuration
            **kwargs: Configuration overrides
        """
        self.config = config or CostConfig()
        
        # Apply any overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.cost_history = []
        
        logger.info(f"Cost model initialized with ₹{self.config.equity_brokerage_per_trade} brokerage per trade")
    
    def calculate_trade_costs(
        self,
        quantity: int,
        price: float,
        trade_type: Union[str, TradeType] = TradeType.BUY,
        market_segment: Union[str, MarketSegment] = MarketSegment.EQUITY,
        is_intraday: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive transaction costs for a trade
        
        Args:
            quantity: Number of shares/contracts
            price: Price per share/contract
            trade_type: Type of trade (BUY/SELL)
            market_segment: Market segment (EQUITY/FUTURES/OPTIONS)
            is_intraday: Whether it's an intraday trade
            
        Returns:
            Dictionary with cost breakdown and total cost
        """
        try:
            # Normalize inputs
            if isinstance(trade_type, str):
                trade_type = TradeType(trade_type.lower())
            if isinstance(market_segment, str):
                market_segment = MarketSegment(market_segment.lower())
            
            # Calculate trade value
            trade_value = quantity * price
            
            # Initialize cost breakdown
            costs = CostBreakdown()
            
            # Calculate each cost component
            costs.brokerage = self._calculate_brokerage(
                trade_value, trade_type, market_segment, is_intraday
            )
            
            costs.stt = self._calculate_stt(
                trade_value, trade_type, market_segment, is_intraday
            )
            
            costs.exchange_charges = self._calculate_exchange_charges(
                trade_value, market_segment
            )
            
            costs.sebi_charges = self._calculate_sebi_charges(trade_value)
            
            costs.stamp_duty = self._calculate_stamp_duty(
                trade_value, trade_type, market_segment
            )
            
            # Calculate GST on (brokerage + exchange charges)
            taxable_amount = costs.brokerage + costs.exchange_charges
            costs.gst = taxable_amount * self.config.gst_rate
            
            # Calculate total cost
            costs.total_cost = (
                costs.brokerage + costs.stt + costs.exchange_charges +
                costs.gst + costs.sebi_charges + costs.stamp_duty
            )
            
            # Round all costs to 2 decimal places
            for field in costs.__dataclass_fields__:
                value = getattr(costs, field)
                setattr(costs, field, round(value, 2))
            
            # Record cost calculation
            cost_record = {
                'timestamp': pd.Timestamp.now(),
                'quantity': quantity,
                'price': price,
                'trade_value': trade_value,
                'trade_type': trade_type.value,
                'market_segment': market_segment.value,
                'is_intraday': is_intraday,
                'costs': costs.to_dict()
            }
            
            self.cost_history.append(cost_record)
            
            return {
                'success': True,
                'total_cost': costs.total_cost,
                'cost_percentage': (costs.total_cost / trade_value) * 100,
                'breakdown': costs.to_dict(),
                'trade_value': trade_value,
                'cost_record': cost_record
            }
            
        except Exception as e:
            logger.error(f"Cost calculation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'total_cost': 0.0
            }
    
    def _calculate_brokerage(
        self,
        trade_value: float,
        trade_type: TradeType,
        market_segment: MarketSegment,
        is_intraday: bool
    ) -> float:
        """Calculate brokerage charges"""
        
        if market_segment == MarketSegment.EQUITY:
            # Flat brokerage for equity (discount broker model)
            brokerage = self.config.equity_brokerage_per_trade
            
            # Apply intraday discount if applicable
            if is_intraday:
                brokerage *= self.config.intraday_discount
            
            # Percentage-based brokerage (if configured)
            if self.config.equity_brokerage_pct > 0:
                pct_brokerage = trade_value * self.config.equity_brokerage_pct
                brokerage = min(brokerage, pct_brokerage)
            
            # Cap at maximum brokerage
            brokerage = min(brokerage, self.config.max_brokerage_per_trade)
            
        elif market_segment == MarketSegment.FUTURES:
            # Futures brokerage (typically flat rate)
            brokerage = self.config.equity_brokerage_per_trade
            
        elif market_segment == MarketSegment.OPTIONS:
            # Options brokerage (typically flat rate)
            brokerage = self.config.equity_brokerage_per_trade
            
        else:
            # Default to equity brokerage
            brokerage = self.config.equity_brokerage_per_trade
        
        return brokerage
    
    def _calculate_stt(
        self,
        trade_value: float,
        trade_type: TradeType,
        market_segment: MarketSegment,
        is_intraday: bool
    ) -> float:
        """Calculate Securities Transaction Tax (STT)"""
        
        stt = 0.0
        
        if market_segment == MarketSegment.EQUITY:
            if is_intraday:
                # Intraday: STT only on sell side
                if trade_type in [TradeType.SELL, TradeType.INTRADAY_SELL]:
                    stt = trade_value * self.config.equity_intraday_stt
            else:
                # Delivery: STT on both buy and sell
                stt = trade_value * self.config.equity_delivery_stt
                
        elif market_segment == MarketSegment.FUTURES:
            # Futures: STT only on sell side
            if trade_type in [TradeType.SELL, TradeType.INTRADAY_SELL]:
                stt = trade_value * self.config.futures_stt
                
        elif market_segment == MarketSegment.OPTIONS:
            # Options: STT only on sell side (on premium)
            if trade_type in [TradeType.SELL, TradeType.INTRADAY_SELL]:
                stt = trade_value * self.config.options_stt
        
        return stt
    
    def _calculate_exchange_charges(
        self,
        trade_value: float,
        market_segment: MarketSegment
    ) -> float:
        """Calculate exchange charges"""
        
        if market_segment == MarketSegment.EQUITY:
            charges = trade_value * self.config.nse_equity_charges
        elif market_segment == MarketSegment.FUTURES:
            charges = trade_value * self.config.nse_futures_charges
        elif market_segment == MarketSegment.OPTIONS:
            charges = trade_value * self.config.nse_options_charges
        else:
            charges = trade_value * self.config.nse_equity_charges
        
        return charges
    
    def _calculate_sebi_charges(self, trade_value: float) -> float:
        """Calculate SEBI regulatory charges"""
        # ₹1 per ₹1 crore turnover
        return trade_value * self.config.sebi_charges
    
    def _calculate_stamp_duty(
        self,
        trade_value: float,
        trade_type: TradeType,
        market_segment: MarketSegment
    ) -> float:
        """Calculate stamp duty"""
        
        # Stamp duty only on buy side for delivery trades
        if (trade_type in [TradeType.BUY] and 
            market_segment == MarketSegment.EQUITY):
            return trade_value * self.config.stamp_duty
        
        return 0.0
    
    def calculate_round_trip_costs(
        self,
        quantity: int,
        entry_price: float,
        exit_price: float,
        market_segment: Union[str, MarketSegment] = MarketSegment.EQUITY,
        is_intraday: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate total costs for a complete round trip (buy + sell)
        
        Args:
            quantity: Number of shares
            entry_price: Entry price per share
            exit_price: Exit price per share
            market_segment: Market segment
            is_intraday: Whether it's intraday trading
            
        Returns:
            Dictionary with complete round trip cost analysis
        """
        # Calculate entry costs
        entry_costs = self.calculate_trade_costs(
            quantity=quantity,
            price=entry_price,
            trade_type=TradeType.BUY,
            market_segment=market_segment,
            is_intraday=is_intraday
        )
        
        # Calculate exit costs
        exit_costs = self.calculate_trade_costs(
            quantity=quantity,
            price=exit_price,
            trade_type=TradeType.SELL,
            market_segment=market_segment,
            is_intraday=is_intraday
        )
        
        if not (entry_costs['success'] and exit_costs['success']):
            return {
                'success': False,
                'error': 'Failed to calculate entry or exit costs'
            }
        
        # Calculate trade P&L
        gross_pnl = (exit_price - entry_price) * quantity
        total_costs = entry_costs['total_cost'] + exit_costs['total_cost']
        net_pnl = gross_pnl - total_costs
        
        return {
            'success': True,
            'gross_pnl': gross_pnl,
            'total_costs': total_costs,
            'net_pnl': net_pnl,
            'cost_percentage': (total_costs / (quantity * entry_price)) * 100,
            'entry_costs': entry_costs,
            'exit_costs': exit_costs,
            'breakeven_price': entry_price + (total_costs / quantity),
            'cost_impact_on_return': (total_costs / abs(gross_pnl)) * 100 if gross_pnl != 0 else 0
        }
    
    def get_cost_statistics(self) -> Dict[str, Any]:
        """Get statistics about historical cost calculations"""
        if not self.cost_history:
            return {'total_calculations': 0}
        
        df = pd.DataFrame([
            {**record, **record['costs']} 
            for record in self.cost_history
        ])
        
        return {
            'total_calculations': len(df),
            'total_cost_sum': df['total_cost'].sum(),
            'avg_cost_per_trade': df['total_cost'].mean(),
            'avg_cost_percentage': (df['total_cost'] / df['trade_value'] * 100).mean(),
            'max_cost': df['total_cost'].max(),
            'min_cost': df['total_cost'].min(),
            'total_brokerage': df['brokerage'].sum(),
            'total_stt': df['stt'].sum(),
            'total_gst': df['gst'].sum(),
            'intraday_trades': len(df[df['is_intraday'] == True]),
            'delivery_trades': len(df[df['is_intraday'] == False])
        }
    
    def optimize_trade_size_for_costs(
        self,
        available_capital: float,
        price: float,
        target_cost_percentage: float = 0.1
    ) -> Dict[str, Any]:
        """
        Optimize trade size to keep costs within target percentage
        
        Args:
            available_capital: Available capital for trade
            price: Price per share
            target_cost_percentage: Target cost as % of trade value
            
        Returns:
            Optimized trade parameters
        """
        max_quantity = int(available_capital / price)
        
        # Binary search for optimal quantity
        left, right = 1, max_quantity
        optimal_quantity = 1
        
        while left <= right:
            mid = (left + right) // 2
            
            # Calculate round trip costs for this quantity
            costs = self.calculate_round_trip_costs(
                quantity=mid,
                entry_price=price,
                exit_price=price,  # Assume same price for cost calculation
                is_intraday=False
            )
            
            if costs['success']:
                cost_pct = costs['cost_percentage']
                
                if cost_pct <= target_cost_percentage:
                    optimal_quantity = mid
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                break
        
        # Calculate final metrics
        final_costs = self.calculate_round_trip_costs(
            quantity=optimal_quantity,
            entry_price=price,
            exit_price=price
        )
        
        return {
            'optimal_quantity': optimal_quantity,
            'trade_value': optimal_quantity * price,
            'cost_percentage': final_costs['cost_percentage'] if final_costs['success'] else 0,
            'total_costs': final_costs['total_costs'] if final_costs['success'] else 0,
            'capital_utilization': (optimal_quantity * price / available_capital) * 100,
            'recommendation': 'OPTIMAL' if final_costs['success'] and final_costs['cost_percentage'] <= target_cost_percentage else 'SUBOPTIMAL'
        }
    
    def reset_history(self):
        """Reset cost calculation history"""
        self.cost_history = []
        logger.info("Cost calculation history reset")


# Predefined cost configurations for different broker types

def create_discount_broker_config() -> CostConfig:
    """Configuration for discount brokers (Zerodha, Upstox, etc.)"""
    return CostConfig(
        equity_brokerage_per_trade=20.0,
        equity_brokerage_pct=0.0,
        max_brokerage_per_trade=20.0,
        intraday_discount=1.0  # No discount, flat ₹20
    )


def create_full_service_broker_config() -> CostConfig:
    """Configuration for full-service brokers"""
    return CostConfig(
        equity_brokerage_per_trade=100.0,
        equity_brokerage_pct=0.005,  # 0.5%
        max_brokerage_per_trade=500.0,
        intraday_discount=0.5
    )


def create_premium_broker_config() -> CostConfig:
    """Configuration for premium/institutional brokers"""
    return CostConfig(
        equity_brokerage_per_trade=10.0,
        equity_brokerage_pct=0.001,  # 0.1%
        max_brokerage_per_trade=50.0,
        intraday_discount=0.8
    )


# Quick calculation functions

def quick_cost_calculation(
    quantity: int,
    price: float,
    trade_type: str = "BUY",
    broker_type: str = "discount"
) -> float:
    """Quick cost calculation with default settings"""
    
    if broker_type == "discount":
        config = create_discount_broker_config()
    elif broker_type == "full_service":
        config = create_full_service_broker_config()
    elif broker_type == "premium":
        config = create_premium_broker_config()
    else:
        config = CostConfig()
    
    cost_model = TransactionCostModel(config)
    result = cost_model.calculate_trade_costs(quantity, price, trade_type)
    
    return result['total_cost'] if result['success'] else 0.0