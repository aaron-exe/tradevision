"""
Backtesting and Strategy Simulator Module
Test trading strategies on historical data with comprehensive performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from src.currency import format_currency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """Backtest trading strategies based on model predictions"""
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001, 
                 slippage: float = 0.0005):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital in USD
            commission: Commission rate (0.001 = 0.1%)
            slippage: Slippage rate (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
    def run_strategy(self, 
                     actual_prices: np.ndarray,
                     predicted_prices: np.ndarray,
                     dates: pd.DatetimeIndex,
                     strategy: str = 'threshold',
                     threshold: float = 0.01,
                     hold_days: int = 1) -> Dict:
        """
        Run backtest with specified strategy
        
        Args:
            actual_prices: Actual historical prices
            predicted_prices: Model predicted prices
            dates: Date index
            strategy: Strategy type ('threshold', 'long_only', 'long_short')
            threshold: Minimum predicted change % to trigger trade (0.01 = 1%)
            hold_days: Number of days to hold position
            
        Returns:
            Dictionary with backtest results
        """
        if len(actual_prices) != len(predicted_prices):
            raise ValueError("Actual and predicted prices must have same length")
        
        # Calculate predicted returns
        predicted_returns = np.zeros(len(predicted_prices))
        for i in range(1, len(predicted_prices)):
            if actual_prices[i-1] != 0:
                predicted_returns[i] = (predicted_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        shares = 0
        entry_price = 0
        trades = []
        equity_curve = [capital]
        positions_held = []
        hold_counter = 0
        
        # Run backtest
        for i in range(1, len(actual_prices) - 1):
            current_price = actual_prices[i]
            next_price = actual_prices[i + 1]
            
            # Close position if hold period expired
            if position != 0 and hold_counter >= hold_days:
                # Calculate return
                if position == 1:  # Long position
                    pnl = shares * (current_price - entry_price)
                    commission_cost = shares * current_price * self.commission
                else:  # Short position
                    pnl = shares * (entry_price - current_price)
                    commission_cost = shares * current_price * self.commission
                
                capital += pnl - commission_cost
                
                trades.append({
                    'entry_date': dates[i - hold_counter],
                    'exit_date': dates[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': 'Long' if position == 1 else 'Short',
                    'shares': shares,
                    'pnl': pnl - commission_cost,
                    'return': (pnl - commission_cost) / (shares * entry_price) * 100
                })
                
                position = 0
                shares = 0
                hold_counter = 0
            
            # Update hold counter
            if position != 0:
                hold_counter += 1
            
            # Generate signals based on strategy
            if position == 0:  # No current position
                signal = 0
                
                if strategy == 'threshold':
                    if predicted_returns[i] > threshold:
                        signal = 1  # Buy
                    elif predicted_returns[i] < -threshold:
                        signal = -1  # Sell short
                        
                elif strategy == 'long_only':
                    if predicted_returns[i] > threshold:
                        signal = 1  # Buy only
                        
                elif strategy == 'long_short':
                    if predicted_returns[i] > 0:
                        signal = 1  # Buy
                    else:
                        signal = -1  # Sell short
                
                # Execute trade
                if signal != 0:
                    # Apply slippage
                    execution_price = current_price * (1 + self.slippage * signal)
                    
                    # Calculate shares to buy/sell
                    available_capital = capital * 0.95  # Use 95% of capital
                    shares = int(available_capital / execution_price)
                    
                    if shares > 0:
                        commission_cost = shares * execution_price * self.commission
                        capital -= commission_cost
                        
                        position = signal
                        entry_price = execution_price
                        hold_counter = 0
            
            # Update equity curve
            if position == 1:
                equity = capital + shares * current_price
            elif position == -1:
                equity = capital + shares * (2 * entry_price - current_price)
            else:
                equity = capital
            
            equity_curve.append(equity)
            positions_held.append(position)
        
        # Close any remaining position
        if position != 0:
            final_price = actual_prices[-1]
            if position == 1:
                pnl = shares * (final_price - entry_price)
                commission_cost = shares * final_price * self.commission
            else:
                pnl = shares * (entry_price - final_price)
                commission_cost = shares * final_price * self.commission
            
            capital += pnl - commission_cost
            equity_curve[-1] = capital
            
            trades.append({
                'entry_date': dates[len(equity_curve) - hold_counter - 2],
                'exit_date': dates[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'position': 'Long' if position == 1 else 'Short',
                'shares': shares,
                'pnl': pnl - commission_cost,
                'return': (pnl - commission_cost) / (shares * entry_price) * 100
            })
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(equity_curve, trades, dates)
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'positions': positions_held,
            'metrics': metrics,
            'final_capital': capital
        }
    
    def _calculate_metrics(self, equity_curve: List[float], trades: List[Dict], 
                          dates: pd.DatetimeIndex) -> Dict:
        """Calculate performance metrics"""
        
        if len(trades) == 0:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0
            }
        
        # Total return
        total_return = equity_curve[-1] - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Trade statistics
        num_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = (len(winning_trades) / num_trades * 100) if num_trades > 0 else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        total_wins = sum([t['pnl'] for t in winning_trades])
        total_losses = abs(sum([t['pnl'] for t in losing_trades]))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Sharpe ratio (annualized)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = equity_curve[0]
        max_dd = 0
        max_dd_pct = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = peak - value
            dd_pct = (dd / peak * 100) if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'num_winning_trades': len(winning_trades),
            'num_losing_trades': len(losing_trades)
        }
    
    def compare_strategies(self, actual_prices: np.ndarray, predicted_prices: np.ndarray,
                          dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Compare multiple strategy configurations"""
        
        strategies = [
            {'name': 'Conservative (2% threshold, long only)', 'strategy': 'long_only', 'threshold': 0.02, 'hold': 1},
            {'name': 'Moderate (1% threshold)', 'strategy': 'threshold', 'threshold': 0.01, 'hold': 1},
            {'name': 'Aggressive (0.5% threshold)', 'strategy': 'threshold', 'threshold': 0.005, 'hold': 1},
            {'name': 'Long-Short', 'strategy': 'long_short', 'threshold': 0, 'hold': 1},
            {'name': 'Hold 3 Days', 'strategy': 'threshold', 'threshold': 0.01, 'hold': 3},
            {'name': 'Hold 5 Days', 'strategy': 'threshold', 'threshold': 0.01, 'hold': 5},
        ]
        
        results = []
        for config in strategies:
            try:
                result = self.run_strategy(
                    actual_prices, predicted_prices, dates,
                    strategy=config['strategy'],
                    threshold=config['threshold'],
                    hold_days=config['hold']
                )
                
                results.append({
                    'Strategy': config['name'],
                    'Total Return': format_currency(result['metrics']['total_return']),
                    'Return %': f"{result['metrics']['total_return_pct']:.2f}%",
                    'Trades': result['metrics']['num_trades'],
                    'Win Rate': f"{result['metrics']['win_rate']:.1f}%",
                    'Sharpe': f"{result['metrics']['sharpe_ratio']:.2f}",
                    'Max DD %': f"{result['metrics']['max_drawdown_pct']:.2f}%"
                })
            except Exception as e:
                logger.error(f"Error running strategy {config['name']}: {e}")
        
        return pd.DataFrame(results)
