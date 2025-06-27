"""
Trade Logger - Comprehensive trade logging and history management
Records detailed trade information for analysis and reporting
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import json

from ..strategies.base_strategy import Signal
from ..risk.risk_integration import TradeRecommendation

logger = logging.getLogger(__name__)


@dataclass
class TradeEntry:
    """Complete trade entry record"""
    # Basic trade information (required fields first)
    trade_id: str
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: int
    signal_type: str  # BUY/SELL
    strategy_name: str
    signal_confidence: float
    initial_stop_price: float
    risk_amount: float
    risk_percentage: float
    position_sizing_method: str
    
    # Optional fields with defaults
    signal_metadata: Dict[str, Any] = field(default_factory=dict)
    trailing_enabled: bool = False
    entry_costs: float = 0.0
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    status: str = "OPEN"  # OPEN, CLOSED, PARTIAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'signal_type': self.signal_type,
            'strategy_name': self.strategy_name,
            'signal_confidence': self.signal_confidence,
            'signal_metadata': self.signal_metadata,
            'initial_stop_price': self.initial_stop_price,
            'risk_amount': self.risk_amount,
            'risk_percentage': self.risk_percentage,
            'position_sizing_method': self.position_sizing_method,
            'trailing_enabled': self.trailing_enabled,
            'entry_costs': self.entry_costs,
            'market_conditions': self.market_conditions,
            'status': self.status
        }


@dataclass
class TradeExit:
    """Complete trade exit record"""
    # Basic exit information
    trade_id: str
    exit_date: datetime
    exit_price: float
    exit_quantity: int
    exit_reason: str  # STOP_LOSS, TRAILING_STOP, SIGNAL, MANUAL, etc.
    
    # P&L information
    gross_pnl: float
    net_pnl: float
    return_percentage: float
    
    # Cost information
    exit_costs: float = 0.0
    total_costs: float = 0.0
    
    # Timing information
    hold_duration: timedelta = field(default_factory=lambda: timedelta(0))
    hold_days: float = 0.0
    
    # Additional metrics
    max_favorable_excursion: float = 0.0  # Best price reached
    max_adverse_excursion: float = 0.0    # Worst price reached
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'trade_id': self.trade_id,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'exit_quantity': self.exit_quantity,
            'exit_reason': self.exit_reason,
            'gross_pnl': self.gross_pnl,
            'net_pnl': self.net_pnl,
            'return_percentage': self.return_percentage,
            'exit_costs': self.exit_costs,
            'total_costs': self.total_costs,
            'hold_duration': str(self.hold_duration),
            'hold_days': self.hold_days,
            'max_favorable_excursion': self.max_favorable_excursion,
            'max_adverse_excursion': self.max_adverse_excursion
        }


@dataclass
class CompleteTrade:
    """Combined entry and exit information"""
    entry: TradeEntry
    exit: Optional[TradeExit] = None
    
    # Additional tracking during trade life
    price_updates: List[Dict[str, Any]] = field(default_factory=list)
    stop_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def is_complete(self) -> bool:
        """Check if trade is complete"""
        return self.exit is not None and self.entry.status in ["CLOSED", "PARTIAL"]
    
    @property
    def duration_days(self) -> float:
        """Get trade duration in days"""
        if self.exit:
            return self.exit.hold_days
        return (datetime.now() - self.entry.entry_date).total_seconds() / 86400
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert complete trade to dictionary"""
        trade_dict = self.entry.to_dict()
        
        if self.exit:
            trade_dict.update(self.exit.to_dict())
            trade_dict['trade_complete'] = True
        else:
            trade_dict['trade_complete'] = False
        
        trade_dict['price_updates'] = self.price_updates
        trade_dict['stop_adjustments'] = self.stop_adjustments
        
        return trade_dict


class TradeLogger:
    """
    Comprehensive trade logging system for backtesting and live trading
    Records all trade details for performance analysis and reporting
    """
    
    def __init__(self, log_directory: Optional[str] = None, enable_csv_export: bool = True):
        """
        Initialize trade logger
        
        Args:
            log_directory: Directory to save log files
            enable_csv_export: Whether to enable CSV export functionality
        """
        self.log_directory = log_directory or "logs/trades"
        self.enable_csv_export = enable_csv_export
        
        # Create log directory if it doesn't exist
        if self.enable_csv_export:
            Path(self.log_directory).mkdir(parents=True, exist_ok=True)
        
        # Trade storage
        self.active_trades: Dict[str, TradeEntry] = {}
        self.completed_trades: List[CompleteTrade] = []
        self.trade_counter = 0
        
        # Performance tracking
        self.session_stats = {
            'session_start': datetime.now(),
            'total_entries': 0,
            'total_exits': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        }
        
        logger.info(f"Trade logger initialized with directory: {self.log_directory}")
    
    def log_entry(
        self,
        symbol: str,
        entry_date: datetime,
        entry_price: float,
        quantity: int,
        signal: Signal,
        recommendation: TradeRecommendation,
        costs: float = 0.0,
        market_conditions: Dict[str, Any] = None
    ) -> str:
        """
        Log a trade entry
        
        Args:
            symbol: Trading symbol
            entry_date: Entry timestamp
            entry_price: Entry price per share
            quantity: Number of shares
            signal: Original trading signal
            recommendation: Risk management recommendation
            costs: Entry transaction costs
            market_conditions: Current market conditions
            
        Returns:
            Trade ID for tracking
        """
        try:
            # Generate unique trade ID
            self.trade_counter += 1
            trade_id = f"{symbol}_{entry_date.strftime('%Y%m%d_%H%M%S')}_{self.trade_counter:04d}"
            
            # Create trade entry record
            entry = TradeEntry(
                trade_id=trade_id,
                symbol=symbol,
                entry_date=entry_date,
                entry_price=entry_price,
                quantity=quantity,
                signal_type=signal.signal_type,
                strategy_name=getattr(signal, 'metadata', {}).get('strategy', 'UNKNOWN'),
                signal_confidence=signal.confidence,
                signal_metadata=signal.metadata,
                initial_stop_price=recommendation.initial_stop_price,
                risk_amount=recommendation.risk_amount,
                risk_percentage=recommendation.risk_percentage,
                position_sizing_method=getattr(recommendation, 'position_sizing_method', 'UNKNOWN'),
                trailing_enabled=recommendation.enable_trailing,
                entry_costs=costs,
                market_conditions=market_conditions or {},
                status="OPEN"
            )
            
            # Store active trade
            self.active_trades[trade_id] = entry
            
            # Update session stats
            self.session_stats['total_entries'] += 1
            
            # Log entry
            logger.info(f"Trade entry logged: {trade_id} - {signal.signal_type} {quantity}@₹{entry_price:.2f}")
            
            return trade_id
            
        except Exception as e:
            logger.error(f"Failed to log trade entry: {str(e)}")
            return ""
    
    def log_exit(
        self,
        symbol: str,
        exit_date: datetime,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        trade_record: Dict[str, Any],
        costs: float = 0.0
    ) -> bool:
        """
        Log a trade exit
        
        Args:
            symbol: Trading symbol
            exit_date: Exit timestamp
            exit_price: Exit price per share
            exit_reason: Reason for exit
            pnl: Net P&L for the trade
            trade_record: Complete trade record
            costs: Exit transaction costs
            
        Returns:
            True if logged successfully
        """
        try:
            # Find the corresponding entry
            trade_id = None
            entry = None
            
            # Look for matching entry by symbol and approximate timing
            for tid, trade_entry in self.active_trades.items():
                if (trade_entry.symbol == symbol and 
                    trade_entry.status == "OPEN"):
                    trade_id = tid
                    entry = trade_entry
                    break
            
            if not entry:
                logger.warning(f"No matching entry found for exit: {symbol}")
                return False
            
            # Calculate additional metrics
            hold_duration = exit_date - entry.entry_date
            hold_days = hold_duration.total_seconds() / 86400
            
            quantity = trade_record.get('quantity', entry.quantity)
            gross_pnl = trade_record.get('gross_pnl', 0.0)
            total_costs = entry.entry_costs + costs
            return_pct = (pnl / (entry.entry_price * quantity)) * 100 if quantity > 0 else 0
            
            # Create exit record
            exit_record = TradeExit(
                trade_id=trade_id,
                exit_date=exit_date,
                exit_price=exit_price,
                exit_quantity=quantity,
                exit_reason=exit_reason,
                gross_pnl=gross_pnl,
                net_pnl=pnl,
                return_percentage=return_pct,
                exit_costs=costs,
                total_costs=total_costs,
                hold_duration=hold_duration,
                hold_days=hold_days
            )
            
            # Update entry status
            entry.status = "CLOSED"
            
            # Create complete trade record
            complete_trade = CompleteTrade(entry=entry, exit=exit_record)
            self.completed_trades.append(complete_trade)
            
            # Remove from active trades
            if trade_id in self.active_trades:
                del self.active_trades[trade_id]
            
            # Update session stats
            self.session_stats['total_exits'] += 1
            self.session_stats['total_pnl'] += pnl
            
            if pnl > self.session_stats['best_trade']:
                self.session_stats['best_trade'] = pnl
            if pnl < self.session_stats['worst_trade']:
                self.session_stats['worst_trade'] = pnl
            
            # Log exit
            logger.info(f"Trade exit logged: {trade_id} - {exit_reason} ₹{pnl:.2f} P&L ({hold_days:.1f} days)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log trade exit: {str(e)}")
            return False
    
    def update_trade_price(
        self,
        trade_id: str,
        current_price: float,
        timestamp: datetime,
        additional_info: Dict[str, Any] = None
    ):
        """Update current price for an active trade"""
        if trade_id in self.active_trades:
            entry = self.active_trades[trade_id]
            
            # Find corresponding complete trade record
            complete_trade = None
            for trade in self.completed_trades:
                if trade.entry.trade_id == trade_id and not trade.is_complete:
                    complete_trade = trade
                    break
            
            if not complete_trade:
                # Create complete trade record for active trade
                complete_trade = CompleteTrade(entry=entry)
                self.completed_trades.append(complete_trade)
            
            # Add price update
            price_update = {
                'timestamp': timestamp,
                'price': current_price,
                'unrealized_pnl': (current_price - entry.entry_price) * entry.quantity if entry.signal_type == 'BUY' else (entry.entry_price - current_price) * entry.quantity,
                'additional_info': additional_info or {}
            }
            
            complete_trade.price_updates.append(price_update)
            
            # Update max favorable/adverse excursion tracking
            # This would be implemented in a more sophisticated version
    
    def log_stop_adjustment(
        self,
        trade_id: str,
        old_stop: float,
        new_stop: float,
        timestamp: datetime,
        reason: str
    ):
        """Log stop loss adjustment for a trade"""
        adjustment = {
            'timestamp': timestamp,
            'old_stop': old_stop,
            'new_stop': new_stop,
            'reason': reason
        }
        
        # Find complete trade and add adjustment
        for trade in self.completed_trades:
            if trade.entry.trade_id == trade_id:
                trade.stop_adjustments.append(adjustment)
                break
        
        logger.debug(f"Stop adjustment logged for {trade_id}: ₹{old_stop:.2f} → ₹{new_stop:.2f} ({reason})")
    
    def get_trade_history_df(self, include_active: bool = False) -> pd.DataFrame:
        """
        Get complete trade history as DataFrame
        
        Args:
            include_active: Whether to include active (unclosed) trades
            
        Returns:
            DataFrame with trade history
        """
        trade_data = []
        
        # Add completed trades
        for trade in self.completed_trades:
            if trade.is_complete:
                trade_data.append(trade.to_dict())
        
        # Add active trades if requested
        if include_active:
            for entry in self.active_trades.values():
                trade_dict = entry.to_dict()
                trade_dict['trade_complete'] = False
                trade_data.append(trade_dict)
        
        if not trade_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(trade_data)
        
        # Convert date columns
        date_columns = ['entry_date', 'exit_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Sort by entry date
        if 'entry_date' in df.columns:
            df = df.sort_values('entry_date')
        
        return df
    
    def export_to_csv(self, filename: Optional[str] = None, include_active: bool = False) -> str:
        """
        Export trade history to CSV file
        
        Args:
            filename: Optional filename (auto-generated if None)
            include_active: Whether to include active trades
            
        Returns:
            Path to exported file
        """
        if not self.enable_csv_export:
            raise ValueError("CSV export is disabled")
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_history_{timestamp}.csv"
        
        filepath = os.path.join(self.log_directory, filename)
        
        # Get trade history
        df = self.get_trade_history_df(include_active=include_active)
        
        if df.empty:
            logger.warning("No trades to export")
            return ""
        
        # Export to CSV
        df.to_csv(filepath, index=False)
        
        logger.info(f"Trade history exported to: {filepath}")
        return filepath
    
    def export_to_json(self, filename: Optional[str] = None, include_active: bool = False) -> str:
        """Export trade history to JSON file"""
        if not self.enable_csv_export:
            raise ValueError("Export is disabled")
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_history_{timestamp}.json"
        
        filepath = os.path.join(self.log_directory, filename)
        
        # Prepare data for JSON export
        export_data = {
            'session_info': {
                'session_start': self.session_stats['session_start'].isoformat(),
                'export_time': datetime.now().isoformat(),
                'total_trades': len(self.completed_trades) + (len(self.active_trades) if include_active else 0)
            },
            'session_stats': {**self.session_stats, 'session_start': self.session_stats['session_start'].isoformat()},
            'completed_trades': [],
            'active_trades': []
        }
        
        # Add completed trades
        for trade in self.completed_trades:
            if trade.is_complete:
                trade_dict = trade.to_dict()
                # Convert datetime objects to ISO strings
                for key, value in trade_dict.items():
                    if isinstance(value, datetime):
                        trade_dict[key] = value.isoformat()
                export_data['completed_trades'].append(trade_dict)
        
        # Add active trades if requested
        if include_active:
            for entry in self.active_trades.values():
                trade_dict = entry.to_dict()
                # Convert datetime objects to ISO strings
                for key, value in trade_dict.items():
                    if isinstance(value, datetime):
                        trade_dict[key] = value.isoformat()
                export_data['active_trades'].append(trade_dict)
        
        # Write to JSON file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Trade history exported to JSON: {filepath}")
        return filepath
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current logging session"""
        completed_count = len(self.completed_trades)
        active_count = len(self.active_trades)
        
        # Calculate additional stats from completed trades
        if completed_count > 0:
            completed_df = self.get_trade_history_df(include_active=False)
            
            if not completed_df.empty and 'net_pnl' in completed_df.columns:
                winning_trades = len(completed_df[completed_df['net_pnl'] > 0])
                losing_trades = len(completed_df[completed_df['net_pnl'] < 0])
                win_rate = (winning_trades / completed_count) * 100
                avg_win = completed_df[completed_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
                avg_loss = completed_df[completed_df['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0
            else:
                winning_trades = losing_trades = 0
                win_rate = avg_win = avg_loss = 0
        else:
            winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = 0
        
        session_duration = datetime.now() - self.session_stats['session_start']
        
        return {
            'session_start': self.session_stats['session_start'],
            'session_duration': str(session_duration),
            'total_entries': self.session_stats['total_entries'],
            'total_exits': self.session_stats['total_exits'],
            'completed_trades': completed_count,
            'active_trades': active_count,
            'total_pnl': self.session_stats['total_pnl'],
            'best_trade': self.session_stats['best_trade'],
            'worst_trade': self.session_stats['worst_trade'],
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def reset_session(self):
        """Reset logging session and clear all data"""
        self.active_trades.clear()
        self.completed_trades.clear()
        self.trade_counter = 0
        
        self.session_stats = {
            'session_start': datetime.now(),
            'total_entries': 0,
            'total_exits': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        }
        
        logger.info("Trade logging session reset")


# Utility functions

def create_backtest_logger(results_directory: str = "results/backtests") -> TradeLogger:
    """Create a trade logger configured for backtesting"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(results_directory, f"backtest_{timestamp}")
    
    return TradeLogger(log_directory=log_dir, enable_csv_export=True)


def analyze_trade_performance(trade_logger: TradeLogger) -> Dict[str, Any]:
    """Analyze performance from trade logger data"""
    df = trade_logger.get_trade_history_df()
    
    if df.empty:
        return {'error': 'No trade data available'}
    
    # Basic performance metrics
    total_trades = len(df)
    winning_trades = len(df[df['net_pnl'] > 0]) if 'net_pnl' in df.columns else 0
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # P&L statistics
    if 'net_pnl' in df.columns:
        total_pnl = df['net_pnl'].sum()
        avg_win = df[df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['net_pnl'] < 0]['net_pnl'].mean() if winning_trades < total_trades else 0
        largest_win = df['net_pnl'].max()
        largest_loss = df['net_pnl'].min()
        
        profit_factor = abs(df[df['net_pnl'] > 0]['net_pnl'].sum() / df[df['net_pnl'] < 0]['net_pnl'].sum()) if df[df['net_pnl'] < 0]['net_pnl'].sum() != 0 else 0
    else:
        total_pnl = avg_win = avg_loss = largest_win = largest_loss = profit_factor = 0
    
    # Timing analysis
    if 'hold_days' in df.columns:
        avg_hold_days = df['hold_days'].mean()
        max_hold_days = df['hold_days'].max()
        min_hold_days = df['hold_days'].min()
    else:
        avg_hold_days = max_hold_days = min_hold_days = 0
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': total_trades - winning_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'profit_factor': profit_factor,
        'avg_hold_days': avg_hold_days,
        'max_hold_days': max_hold_days,
        'min_hold_days': min_hold_days
    }