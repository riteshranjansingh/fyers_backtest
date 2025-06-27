"""
Backtesting Engine - Core backtesting logic for strategy evaluation
Processes historical data chronologically with risk management integration
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from ..strategies.base_strategy import BaseStrategy, StrategyResult, Signal
from ..risk.risk_integration import RiskManager, TradeRecommendation
from ..config.integration_bridge import create_risk_manager
from .trade_executor import TradeExecutor
from .cost_model import TransactionCostModel
from .trade_logger import TradeLogger

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting engine"""
    initial_capital: float = 100000.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    risk_profile: str = "moderate"
    cost_profile: str = "zerodha"  # Cost profile to use
    enable_costs: bool = True
    enable_slippage: bool = True
    enable_logging: bool = True
    progress_callback: Optional = None
    
    # Risk management settings
    max_positions: int = 5
    max_daily_trades: int = 10


@dataclass
class BacktestResults:
    """Comprehensive backtesting results"""
    # Basic performance metrics
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    
    # Advanced metrics
    cagr: float
    calmar_ratio: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Time series data
    equity_curve: pd.Series
    drawdown_series: pd.Series
    trade_history: pd.DataFrame
    
    # Performance by period
    monthly_returns: pd.Series
    yearly_returns: pd.Series
    
    # Risk management stats
    max_concurrent_positions: int
    avg_position_size: float
    total_commissions: float
    
    # Metadata
    backtest_duration: timedelta
    data_points: int
    start_date: datetime
    end_date: datetime
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            'performance': {
                'total_return': f"{self.total_return_pct:.2f}%",
                'cagr': f"{self.cagr:.2f}%",
                'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
                'max_drawdown': f"{self.max_drawdown_pct:.2f}%"
            },
            'trades': {
                'total_trades': self.total_trades,
                'win_rate': f"{self.win_rate:.2f}%",
                'profit_factor': f"{self.profit_factor:.2f}",
                'avg_win': f"₹{self.avg_win:.2f}",
                'avg_loss': f"₹{self.avg_loss:.2f}"
            },
            'risk': {
                'max_concurrent_positions': self.max_concurrent_positions,
                'total_commissions': f"₹{self.total_commissions:.2f}",
                'calmar_ratio': f"{self.calmar_ratio:.2f}"
            }
        }


class BacktestEngine:
    """
    Core backtesting engine that processes historical data chronologically
    with comprehensive risk management integration
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig = None,
        risk_manager: RiskManager = None
    ):
        """
        Initialize backtesting engine
        
        Args:
            strategy: Trading strategy to backtest
            config: Backtesting configuration
            risk_manager: Risk management system
        """
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.risk_manager = risk_manager or create_risk_manager(self.config.risk_profile)
        
        # Initialize components
        self.trade_executor = TradeExecutor(
            enable_slippage=self.config.enable_slippage,
            slippage_bps=2.0  # 2 basis points default slippage
        )
        
        if self.config.enable_costs:
            try:
                from ..config.cost_profiles import create_cost_model_with_profile
                self.cost_model = create_cost_model_with_profile(self.config.cost_profile)
            except ImportError:
                # Fallback to default cost model
                self.cost_model = TransactionCostModel()
        else:
            self.cost_model = None
        
        self.trade_logger = TradeLogger() if self.config.enable_logging else None
        
        # Portfolio state
        self.portfolio = {
            'cash': self.config.initial_capital,
            'positions': {},  # symbol -> position info
            'equity_history': [],
            'trade_history': [],
            'daily_pnl': []
        }
        
        # Performance tracking
        self.metrics = {
            'peak_equity': self.config.initial_capital,
            'drawdown': 0.0,
            'max_drawdown': 0.0,
            'consecutive_losses': 0,
            'max_consecutive_losses': 0
        }
        
        logger.info(f"Backtesting engine initialized with {strategy.name} strategy")
    
    def run(
        self,
        data: pd.DataFrame,
        symbol: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> BacktestResults:
        """
        Run complete backtest on historical data
        
        Args:
            data: Historical OHLCV data with DatetimeIndex
            symbol: Symbol being backtested
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)
            
        Returns:
            BacktestResults with comprehensive performance metrics
        """
        try:
            # Validate and prepare data
            data = self._prepare_data(data, start_date, end_date)
            symbol = symbol or "BACKTEST_SYMBOL"
            
            logger.info(f"Starting backtest: {symbol} from {data.index[0]} to {data.index[-1]}")
            logger.info(f"Data points: {len(data)}, Initial capital: ₹{self.config.initial_capital:,.0f}")
            
            # Generate all signals upfront
            strategy_result = self.strategy.generate_signals(data)
            signals_df = strategy_result.get_signals_df()
            
            logger.info(f"Generated {len(strategy_result.signals)} signals with {strategy_result.avg_confidence:.3f} avg confidence")
            
            # Reset portfolio state
            self._reset_portfolio()
            
            # Process data chronologically
            total_points = len(data)
            processed_points = 0
            
            for current_date, row in data.iterrows():
                processed_points += 1
                
                # Update progress
                if self.config.progress_callback:
                    progress = (processed_points / total_points) * 100
                    self.config.progress_callback(progress)
                
                # Process this time period
                self._process_timestamp(
                    current_date=current_date,
                    market_data=row,
                    all_data=data.loc[:current_date],
                    signals_on_date=signals_df.loc[signals_df.index == current_date] if current_date in signals_df.index else pd.DataFrame(),
                    symbol=symbol
                )
                
                # Update portfolio metrics
                self._update_portfolio_metrics(current_date, row['close'])
            
            # Calculate final results
            results = self._calculate_results(data, symbol)
            
            logger.info(f"Backtest completed: {results.total_return_pct:.2f}% return, {results.total_trades} trades")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtesting failed: {str(e)}")
            raise
    
    def _prepare_data(self, data: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Prepare and validate data for backtesting"""
        # Validate data format
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Filter by date range if provided
        if start_date:
            data = data.loc[data.index >= pd.to_datetime(start_date)]
        if end_date:
            data = data.loc[data.index <= pd.to_datetime(end_date)]
        
        if len(data) < 50:
            raise ValueError(f"Insufficient data: {len(data)} points (minimum 50 required)")
        
        # Sort by date to ensure chronological processing
        data = data.sort_index()
        
        # Forward fill any missing values
        data = data.fillna(method='ffill')
        
        return data
    
    def _reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.portfolio = {
            'cash': self.config.initial_capital,
            'positions': {},
            'equity_history': [],
            'trade_history': [],
            'daily_pnl': []
        }
        
        self.metrics = {
            'peak_equity': self.config.initial_capital,
            'drawdown': 0.0,
            'max_drawdown': 0.0,
            'consecutive_losses': 0,
            'max_consecutive_losses': 0
        }
    
    def _process_timestamp(
        self,
        current_date: datetime,
        market_data: pd.Series,
        all_data: pd.DataFrame,
        signals_on_date: pd.DataFrame,
        symbol: str
    ):
        """Process a single timestamp in the backtest"""
        current_price = market_data['close']
        
        # 1. Update existing positions
        self._update_existing_positions(current_date, current_price, market_data, symbol)
        
        # 2. Process new signals
        if not signals_on_date.empty:
            for _, signal_row in signals_on_date.iterrows():
                self._process_signal(
                    signal_row=signal_row,
                    current_date=current_date,
                    market_data=market_data,
                    all_data=all_data,
                    symbol=symbol
                )
        
        # 3. Check position limits and risk controls
        self._enforce_risk_limits(current_date, current_price)
    
    def _process_signal(
        self,
        signal_row: pd.Series,
        current_date: datetime,
        market_data: pd.Series,
        all_data: pd.DataFrame,
        symbol: str
    ):
        """Process a trading signal"""
        try:
            # Create Signal object from row data
            signal = Signal(
                timestamp=current_date,
                signal_type=signal_row['signal_type'],
                price=signal_row['price'],
                confidence=signal_row['confidence'],
                metadata=signal_row.get('metadata', {})
            )
            
            # Check if we can take new positions
            if len(self.portfolio['positions']) >= self.config.max_positions:
                logger.debug(f"Skipping signal - max positions ({self.config.max_positions}) reached")
                return
            
            # Check if already have position in this symbol
            if symbol in self.portfolio['positions']:
                logger.debug(f"Skipping signal - already have position in {symbol}")
                return
            
            # Get trade recommendation from risk manager
            recommendation = self.risk_manager.process_strategy_signal(
                signal=signal,
                market_data=all_data,
                symbol=symbol,
                current_price=market_data['close']
            )
            
            if not recommendation.trade_valid:
                logger.debug(f"Signal rejected: {recommendation.rejection_reason}")
                return
            
            # Check if we have enough cash
            required_capital = recommendation.position_value
            if required_capital > self.portfolio['cash']:
                logger.debug(f"Insufficient cash: need ₹{required_capital:.0f}, have ₹{self.portfolio['cash']:.0f}")
                return
            
            # Execute trade
            execution_result = self.trade_executor.execute_trade(
                signal=signal,
                recommendation=recommendation,
                market_data=market_data
            )
            
            if execution_result['success']:
                self._record_trade_entry(
                    signal=signal,
                    recommendation=recommendation,
                    execution_result=execution_result,
                    current_date=current_date,
                    symbol=symbol
                )
            
        except Exception as e:
            logger.error(f"Error processing signal at {current_date}: {str(e)}")
    
    def _record_trade_entry(
        self,
        signal: Signal,
        recommendation: TradeRecommendation,
        execution_result: Dict,
        current_date: datetime,
        symbol: str
    ):
        """Record a successful trade entry"""
        entry_price = execution_result['execution_price']
        quantity = execution_result['quantity']
        
        # Calculate costs
        costs = 0
        if self.cost_model:
            cost_result = self.cost_model.calculate_trade_costs(
                quantity=quantity,
                price=entry_price,
                trade_type='BUY'
            )
            costs = cost_result['total_cost']
        
        # Update cash
        total_cost = (quantity * entry_price) + costs
        self.portfolio['cash'] -= total_cost
        
        # Create position record
        position = {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price,
            'entry_date': current_date,
            'entry_costs': costs,
            'stop_price': recommendation.initial_stop_price,
            'signal_type': signal.signal_type,
            'recommendation': recommendation,
            'current_price': entry_price,
            'unrealized_pnl': 0.0,
            'trailing_stop': None
        }
        
        # Setup trailing stop if enabled
        if recommendation.enable_trailing:
            position['trailing_stop'] = {
                'current_stop': recommendation.initial_stop_price,
                'highest_price': entry_price if signal.signal_type == 'BUY' else entry_price,
                'lowest_price': entry_price if signal.signal_type == 'SELL' else entry_price,
                'moved_to_breakeven': False,
                'partial_booking_done': False
            }
        
        self.portfolio['positions'][symbol] = position
        
        # Log trade if enabled
        if self.trade_logger:
            self.trade_logger.log_entry(
                symbol=symbol,
                entry_date=current_date,
                entry_price=entry_price,
                quantity=quantity,
                signal=signal,
                recommendation=recommendation,
                costs=costs
            )
        
        logger.debug(f"Trade entered: {symbol} {quantity}@₹{entry_price:.2f}, Stop: ₹{recommendation.initial_stop_price:.2f}")
    
    def _update_existing_positions(
        self,
        current_date: datetime,
        current_price: float,
        market_data: pd.Series,
        symbol: str
    ):
        """Update all existing positions with current market data"""
        positions_to_close = []
        
        for pos_symbol, position in self.portfolio['positions'].items():
            if pos_symbol != symbol:
                continue  # Skip positions in other symbols
                
            # Update current price and unrealized P&L
            position['current_price'] = current_price
            
            if position['signal_type'] == 'BUY':
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
            else:  # SELL/SHORT
                position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['quantity']
            
            # Check stop loss
            stop_hit = self._check_stop_loss(position, current_price, market_data)
            if stop_hit:
                positions_to_close.append((pos_symbol, 'STOP_LOSS', stop_hit['exit_price']))
                continue
            
            # Update trailing stop
            if position.get('trailing_stop'):
                self._update_trailing_stop(position, current_price, current_date)
                
                # Check if trailing stop hit
                trailing_hit = self._check_trailing_stop(position, current_price)
                if trailing_hit:
                    positions_to_close.append((pos_symbol, 'TRAILING_STOP', trailing_hit['exit_price']))
        
        # Close positions that hit stops
        for pos_symbol, exit_reason, exit_price in positions_to_close:
            self._close_position(pos_symbol, current_date, exit_price, exit_reason)
    
    def _check_stop_loss(self, position: Dict, current_price: float, market_data: pd.Series) -> Optional[Dict]:
        """Check if position should be stopped out"""
        stop_price = position['stop_price']
        signal_type = position['signal_type']
        
        if signal_type == 'BUY' and current_price <= stop_price:
            return {'exit_price': stop_price, 'reason': 'STOP_LOSS'}
        elif signal_type == 'SELL' and current_price >= stop_price:
            return {'exit_price': stop_price, 'reason': 'STOP_LOSS'}
        
        return None
    
    def _update_trailing_stop(self, position: Dict, current_price: float, current_date: datetime):
        """Update trailing stop for position"""
        trailing = position['trailing_stop']
        if not trailing:
            return
        
        signal_type = position['signal_type']
        entry_price = position['entry_price']
        
        if signal_type == 'BUY':
            # Update highest price seen
            if current_price > trailing['highest_price']:
                trailing['highest_price'] = current_price
            
            # Check for breakeven move (1:1 R:R)
            if not trailing['moved_to_breakeven']:
                risk = entry_price - position['stop_price']
                if current_price >= entry_price + risk:  # 1:1 R:R achieved
                    trailing['current_stop'] = entry_price + 2  # Breakeven + 2 points
                    trailing['moved_to_breakeven'] = True
                    logger.debug(f"Trailing stop moved to breakeven+2: ₹{trailing['current_stop']:.2f}")
            
            # Check for partial booking (2:1 R:R)
            if not trailing['partial_booking_done']:
                risk = entry_price - position['stop_price']
                if current_price >= entry_price + (2 * risk):  # 2:1 R:R achieved
                    # In real implementation, would book 50% here
                    trailing['partial_booking_done'] = True
                    logger.debug(f"2:1 R:R achieved - partial booking trigger")
        
        else:  # SELL position
            # Update lowest price seen
            if current_price < trailing['lowest_price']:
                trailing['lowest_price'] = current_price
            
            # Similar logic for short positions (inverted)
            if not trailing['moved_to_breakeven']:
                risk = position['stop_price'] - entry_price
                if current_price <= entry_price - risk:  # 1:1 R:R achieved
                    trailing['current_stop'] = entry_price - 2  # Breakeven - 2 points
                    trailing['moved_to_breakeven'] = True
    
    def _check_trailing_stop(self, position: Dict, current_price: float) -> Optional[Dict]:
        """Check if trailing stop should trigger"""
        trailing = position.get('trailing_stop')
        if not trailing:
            return None
        
        signal_type = position['signal_type']
        current_stop = trailing['current_stop']
        
        if signal_type == 'BUY' and current_price <= current_stop:
            return {'exit_price': current_stop, 'reason': 'TRAILING_STOP'}
        elif signal_type == 'SELL' and current_price >= current_stop:
            return {'exit_price': current_stop, 'reason': 'TRAILING_STOP'}
        
        return None
    
    def _close_position(self, symbol: str, exit_date: datetime, exit_price: float, exit_reason: str):
        """Close a position and record the trade"""
        position = self.portfolio['positions'][symbol]
        
        quantity = position['quantity']
        entry_price = position['entry_price']
        entry_costs = position['entry_costs']
        
        # Calculate exit costs
        exit_costs = 0
        if self.cost_model:
            cost_result = self.cost_model.calculate_trade_costs(
                quantity=quantity,
                price=exit_price,
                trade_type='SELL'
            )
            exit_costs = cost_result['total_cost']
        
        # Calculate P&L
        if position['signal_type'] == 'BUY':
            gross_pnl = (exit_price - entry_price) * quantity
        else:  # SELL
            gross_pnl = (entry_price - exit_price) * quantity
        
        net_pnl = gross_pnl - entry_costs - exit_costs
        
        # Update cash
        if position['signal_type'] == 'BUY':
            self.portfolio['cash'] += (quantity * exit_price) - exit_costs
        else:  # SELL
            self.portfolio['cash'] += (quantity * entry_price) + gross_pnl - exit_costs
        
        # Record trade
        trade_record = {
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'signal_type': position['signal_type'],
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'gross_pnl': gross_pnl,
            'entry_costs': entry_costs,
            'exit_costs': exit_costs,
            'net_pnl': net_pnl,
            'return_pct': (net_pnl / (quantity * entry_price)) * 100,
            'exit_reason': exit_reason,
            'hold_days': (exit_date - position['entry_date']).days,
            'recommendation': position['recommendation']
        }
        
        self.portfolio['trade_history'].append(trade_record)
        
        # Log trade if enabled
        if self.trade_logger:
            self.trade_logger.log_exit(
                symbol=symbol,
                exit_date=exit_date,
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl=net_pnl,
                trade_record=trade_record
            )
        
        # Remove position
        del self.portfolio['positions'][symbol]
        
        logger.debug(f"Position closed: {symbol} P&L: ₹{net_pnl:.2f} ({exit_reason})")
    
    def _enforce_risk_limits(self, current_date: datetime, current_price: float):
        """Enforce portfolio-level risk limits"""
        # Emergency stop loss (5% account loss)
        current_equity = self._calculate_current_equity(current_price)
        drawdown_pct = (self.metrics['peak_equity'] - current_equity) / self.metrics['peak_equity'] * 100
        
        if drawdown_pct > 5.0:  # Emergency stop
            logger.warning(f"Emergency stop triggered at {drawdown_pct:.1f}% drawdown")
            # Close all positions
            for symbol in list(self.portfolio['positions'].keys()):
                self._close_position(symbol, current_date, current_price, 'EMERGENCY_STOP')
    
    def _update_portfolio_metrics(self, current_date: datetime, current_price: float):
        """Update portfolio metrics and equity curve"""
        current_equity = self._calculate_current_equity(current_price)
        
        # Update peak equity and drawdown
        if current_equity > self.metrics['peak_equity']:
            self.metrics['peak_equity'] = current_equity
            self.metrics['drawdown'] = 0.0
        else:
            self.metrics['drawdown'] = self.metrics['peak_equity'] - current_equity
            if self.metrics['drawdown'] > self.metrics['max_drawdown']:
                self.metrics['max_drawdown'] = self.metrics['drawdown']
        
        # Record equity point
        self.portfolio['equity_history'].append({
            'date': current_date,
            'equity': current_equity,
            'cash': self.portfolio['cash'],
            'positions_value': current_equity - self.portfolio['cash'],
            'drawdown': self.metrics['drawdown']
        })
    
    def _calculate_current_equity(self, current_price: float) -> float:
        """Calculate current portfolio equity"""
        equity = self.portfolio['cash']
        
        # Add value of open positions
        for position in self.portfolio['positions'].values():
            position_value = position['quantity'] * position['current_price']
            equity += position_value
        
        return equity
    
    def _calculate_results(self, data: pd.DataFrame, symbol: str) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        # Convert equity history to DataFrame
        equity_df = pd.DataFrame(self.portfolio['equity_history'])
        equity_df.set_index('date', inplace=True)
        
        # Convert trade history to DataFrame
        trades_df = pd.DataFrame(self.portfolio['trade_history'])
        
        # Basic performance
        initial_capital = self.config.initial_capital
        final_equity = equity_df['equity'].iloc[-1] if not equity_df.empty else initial_capital
        total_return = final_equity - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # Time metrics
        start_date = data.index[0]
        end_date = data.index[-1]
        duration = end_date - start_date
        years = duration.days / 365.25
        
        # Calculate CAGR
        cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Trade statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0]) if total_trades > 0 else 0
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # P&L statistics
        if total_trades > 0:
            wins = trades_df[trades_df['net_pnl'] > 0]['net_pnl']
            losses = trades_df[trades_df['net_pnl'] < 0]['net_pnl']
            
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
            largest_win = wins.max() if len(wins) > 0 else 0
            largest_loss = abs(losses.min()) if len(losses) > 0 else 0
            
            gross_profit = wins.sum() if len(wins) > 0 else 0
            gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            avg_win = avg_loss = largest_win = largest_loss = profit_factor = 0
        
        # Risk metrics
        max_drawdown = self.metrics['max_drawdown']
        max_drawdown_pct = (max_drawdown / self.metrics['peak_equity'] * 100) if self.metrics['peak_equity'] > 0 else 0
        
        # Calculate Sharpe and Sortino ratios
        if not equity_df.empty:
            returns = equity_df['equity'].pct_change().dropna()
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            negative_returns = returns[returns < 0]
            sortino_ratio = (returns.mean() * 252) / (negative_returns.std() * np.sqrt(252)) if len(negative_returns) > 0 and negative_returns.std() > 0 else 0
        else:
            sharpe_ratio = sortino_ratio = 0
        
        # Calmar ratio
        calmar_ratio = cagr / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        # Create series for visualization
        equity_curve = equity_df['equity'] if not equity_df.empty else pd.Series([initial_capital], name='equity')
        drawdown_series = equity_df['drawdown'] if not equity_df.empty else pd.Series([0], name='drawdown')
        
        # Monthly and yearly returns
        if not equity_df.empty:
            monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
            yearly_returns = equity_curve.resample('Y').last().pct_change().dropna()
        else:
            monthly_returns = pd.Series(dtype=float)
            yearly_returns = pd.Series(dtype=float)
        
        # Additional metrics
        total_commissions = trades_df[['entry_costs', 'exit_costs']].sum().sum() if total_trades > 0 else 0
        max_concurrent_positions = max([len(self.portfolio['positions']) for _ in range(1)], default=0)
        avg_position_size = trades_df['quantity'].mean() if total_trades > 0 else 0
        
        return BacktestResults(
            initial_capital=initial_capital,
            final_capital=final_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            cagr=cagr,
            calmar_ratio=calmar_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            equity_curve=equity_curve,
            drawdown_series=drawdown_series,
            trade_history=trades_df,
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns,
            max_concurrent_positions=max_concurrent_positions,
            avg_position_size=avg_position_size,
            total_commissions=total_commissions,
            backtest_duration=duration,
            data_points=len(data),
            start_date=start_date,
            end_date=end_date
        )


def quick_backtest(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    initial_capital: float = 100000,
    risk_profile: str = "moderate",
    cost_profile: str = "zerodha"
) -> BacktestResults:
    """
    Quick backtest with default settings
    
    Args:
        strategy: Trading strategy
        data: Historical data
        initial_capital: Starting capital
        risk_profile: Risk management profile
        cost_profile: Cost profile to use
        
    Returns:
        BacktestResults
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        risk_profile=risk_profile,
        cost_profile=cost_profile
    )
    
    engine = BacktestEngine(strategy, config)
    return engine.run(data)