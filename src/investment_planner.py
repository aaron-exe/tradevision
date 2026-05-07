"""
Investment Planner Module
AI-powered investment planning and portfolio allocation engine.
Reuses existing TradingSignals, RiskAnalyzer, StockDataFetcher, and FeatureEngineer
to generate intelligent stock recommendations and capital allocation strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.data_fetcher import StockDataFetcher
from src.feature_engineering import FeatureEngineer
from src.trading_signals import TradingSignals
from src.risk_metrics import RiskAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Curated stock universe with sector classification
STOCK_UNIVERSE = {
    'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology'},
    'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology'},
    'MSFT': {'name': 'Microsoft Corp.', 'sector': 'Technology'},
    'TSLA': {'name': 'Tesla Inc.', 'sector': 'Technology'},
    'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Consumer Cyclical'},
    'META': {'name': 'Meta Platforms Inc.', 'sector': 'Technology'},
    'NFLX': {'name': 'Netflix Inc.', 'sector': 'Communication Services'},
    'NVDA': {'name': 'NVIDIA Corp.', 'sector': 'Technology'},
    'KO': {'name': 'Coca-Cola Co.', 'sector': 'Consumer Defensive'},
    'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Healthcare'},
    'WMT': {'name': 'Walmart Inc.', 'sector': 'Consumer Defensive'},
    'PG': {'name': 'Procter & Gamble Co.', 'sector': 'Consumer Defensive'},
    'JPM': {'name': 'JPMorgan Chase & Co.', 'sector': 'Financial Services'},
    'BAC': {'name': 'Bank of America Corp.', 'sector': 'Financial Services'},
    'V': {'name': 'Visa Inc.', 'sector': 'Financial Services'},
    'MA': {'name': 'Mastercard Inc.', 'sector': 'Financial Services'},
    'DIS': {'name': 'Walt Disney Co.', 'sector': 'Communication Services'},
    'MCD': {'name': "McDonald's Corp.", 'sector': 'Consumer Cyclical'},
    'NKE': {'name': 'Nike Inc.', 'sector': 'Consumer Cyclical'},
    'XOM': {'name': 'Exxon Mobil Corp.', 'sector': 'Energy'},
    'CVX': {'name': 'Chevron Corp.', 'sector': 'Energy'},
    'PFE': {'name': 'Pfizer Inc.', 'sector': 'Healthcare'},
    'UNH': {'name': 'UnitedHealth Group Inc.', 'sector': 'Healthcare'},
    'HD': {'name': 'Home Depot Inc.', 'sector': 'Consumer Cyclical'},
}

# Available sectors derived from universe
AVAILABLE_SECTORS = sorted(set(info['sector'] for info in STOCK_UNIVERSE.values()))

# Timeframe mappings
TIMEFRAME_MAP = {
    '1 Month': {'months': 1, 'trading_days': 21},
    '3 Months': {'months': 3, 'trading_days': 63},
    '6 Months': {'months': 6, 'trading_days': 126},
    '1 Year': {'months': 12, 'trading_days': 252},
    '2 Years': {'months': 24, 'trading_days': 504},
}


class InvestmentPlanner:
    """AI-powered investment planner using existing analysis systems."""

    def __init__(self):
        self.fetcher = StockDataFetcher()
        self.engineer = FeatureEngineer()

    def generate_plan(self, investment_amount, target_return_pct,
                      timeframe='1 Year', risk_tolerance='Medium',
                      sector_preferences=None, strategy='AI-selected',
                      num_stocks=5, progress_callback=None):
        """
        Generate a complete investment plan.

        Args:
            investment_amount (float): Total capital to invest (USD).
            target_return_pct (float): Desired return percentage.
            timeframe (str): Investment horizon key from TIMEFRAME_MAP.
            risk_tolerance (str): 'Low', 'Medium', or 'High'.
            sector_preferences (list|None): Sectors to focus on, or None for all.
            strategy (str): Investment strategy preference.
            num_stocks (int): Desired number of stocks in allocation.
            progress_callback (callable|None): fn(pct, msg) for progress updates.

        Returns:
            dict: Complete investment plan with allocations, projections, and analysis.
        """
        if progress_callback is None:
            progress_callback = lambda pct, msg: None

        tf = TIMEFRAME_MAP.get(timeframe, TIMEFRAME_MAP['1 Year'])
        target_return_decimal = target_return_pct / 100.0

        # Step 1 — Determine candidate tickers
        progress_callback(0.05, "Identifying candidate stocks...")
        candidates = self._get_candidates(sector_preferences)

        # Step 2 — Fetch data and score every candidate
        progress_callback(0.10, "Fetching market data & scoring stocks...")
        scored = self._score_candidates(candidates, risk_tolerance, strategy,
                                        tf, progress_callback)

        if not scored:
            return {'success': False,
                    'message': 'Unable to fetch data for any candidate stocks. Please try again.'}

        # Step 3 — Select top N and allocate
        progress_callback(0.80, "Generating optimal allocation...")
        selected = scored[:min(num_stocks, len(scored))]
        allocations = self._allocate_capital(selected, investment_amount,
                                             target_return_decimal, tf,
                                             risk_tolerance)

        # Step 4 — Portfolio-level analytics
        progress_callback(0.90, "Computing portfolio analytics...")
        portfolio_analytics = self._compute_portfolio_analytics(
            allocations, investment_amount, target_return_pct, tf, risk_tolerance)

        progress_callback(1.0, "Investment plan ready!")

        return {
            'success': True,
            'allocations': allocations,
            'analytics': portfolio_analytics,
            'parameters': {
                'investment_amount': investment_amount,
                'target_return_pct': target_return_pct,
                'timeframe': timeframe,
                'timeframe_info': tf,
                'risk_tolerance': risk_tolerance,
                'sector_preferences': sector_preferences,
                'strategy': strategy,
                'num_stocks': num_stocks,
                'generated_at': datetime.now().isoformat(),
            }
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_candidates(self, sector_preferences):
        """Filter stock universe by sector preferences."""
        if sector_preferences and len(sector_preferences) > 0:
            return {t: info for t, info in STOCK_UNIVERSE.items()
                    if info['sector'] in sector_preferences}
        return dict(STOCK_UNIVERSE)

    def _score_candidates(self, candidates, risk_tolerance, strategy, tf,
                          progress_callback):
        """
        Fetch 1-year data for each candidate, compute signals & risk,
        and return a sorted list of scored stock dicts.
        """
        scored = []
        total = len(candidates)

        for idx, (ticker, info) in enumerate(candidates.items()):
            pct = 0.10 + 0.65 * ((idx + 1) / total)
            progress_callback(pct, f"Analyzing {info['name']} ({ticker})...")
            try:
                df = self.fetcher.fetch_stock_data(ticker, period='1y')
                if df is None or df.empty or len(df) < 60:
                    continue

                df = self.engineer.add_all_indicators(df)

                # --- Trading Signals ---
                signals = TradingSignals.get_comprehensive_signals(df)
                overall = signals['overall']
                buy_score = overall['buy_score']
                sell_score = overall['sell_score']
                confidence = overall['confidence']
                recommendation = overall['recommendation']

                # --- Risk Metrics ---
                risk_metrics = RiskAnalyzer.get_comprehensive_risk_metrics(df)
                risk_rating = RiskAnalyzer.get_risk_rating(risk_metrics)
                volatility = risk_metrics['annualized_volatility_pct']
                sharpe = risk_metrics['sharpe_ratio']
                max_dd = abs(risk_metrics['max_drawdown_pct'])
                mean_return_pct = risk_metrics['mean_return_pct']

                current_price = float(df['Close'].iloc[-1])

                # --- Composite score ---
                composite = self._compute_composite_score(
                    buy_score, sell_score, confidence, sharpe, volatility,
                    max_dd, mean_return_pct, risk_tolerance, strategy)

                # --- Projected upside (annualized mean return scaled to timeframe) ---
                trading_days = tf['trading_days']
                projected_return_pct = (risk_metrics['mean_return'] * trading_days) * 100
                # Clamp to reasonable bounds
                projected_return_pct = max(-50, min(projected_return_pct, 200))

                # --- Reasoning text ---
                reasoning = self._generate_reasoning(
                    ticker, info['name'], recommendation, confidence,
                    volatility, sharpe, mean_return_pct, risk_rating, signals)

                scored.append({
                    'ticker': ticker,
                    'name': info['name'],
                    'sector': info['sector'],
                    'current_price': current_price,
                    'composite_score': composite,
                    'buy_score': buy_score,
                    'sell_score': sell_score,
                    'confidence': confidence,
                    'recommendation': recommendation,
                    'volatility': volatility,
                    'sharpe': sharpe,
                    'max_drawdown': max_dd,
                    'mean_return_pct': mean_return_pct,
                    'projected_return_pct': projected_return_pct,
                    'risk_rating': risk_rating,
                    'reasoning': reasoning,
                })

            except Exception as e:
                logger.warning(f"Skipping {ticker}: {e}")
                continue

        # Sort by composite score descending
        scored.sort(key=lambda s: s['composite_score'], reverse=True)
        return scored

    def _compute_composite_score(self, buy_score, sell_score, confidence,
                                  sharpe, volatility, max_dd, mean_return_pct,
                                  risk_tolerance, strategy):
        """
        Weighted composite score combining signal strength, risk-adjusted
        return quality, and user-preference modifiers.
        """
        # Base signal score (0-100 range)
        signal_score = max(0, buy_score - sell_score) + (confidence / 2)

        # Risk-adjusted return score
        sharpe_score = max(0, min(sharpe * 20, 50))  # 0-50
        return_score = max(0, min(mean_return_pct, 50))  # 0-50

        # Volatility penalty (lower vol = higher score)
        vol_penalty = max(0, min(volatility / 2, 25))

        # Drawdown penalty
        dd_penalty = max(0, min(max_dd / 3, 20))

        # Risk tolerance modifier
        risk_mult = {'Low': 0.6, 'Medium': 1.0, 'High': 1.4}.get(risk_tolerance, 1.0)

        # Strategy modifiers
        strat_bonus = 0
        if strategy == 'Growth':
            strat_bonus = return_score * 0.3
        elif strategy == 'Value':
            strat_bonus = sharpe_score * 0.3 - vol_penalty * 0.2
        elif strategy == 'Dividend':
            strat_bonus = -vol_penalty * 0.3  # favor low-vol
        elif strategy == 'Momentum':
            strat_bonus = signal_score * 0.3

        composite = (
            signal_score * 0.35
            + sharpe_score * 0.20
            + return_score * 0.20
            - vol_penalty * 0.10 * (2 - risk_mult)
            - dd_penalty * 0.10
            + strat_bonus * 0.05
        ) * risk_mult

        return round(max(0, composite), 2)

    def _allocate_capital(self, selected, investment_amount,
                          target_return_decimal, tf, risk_tolerance):
        """
        Distribute capital across selected stocks proportionally
        to their composite scores, then compute per-stock metrics.
        """
        total_score = sum(s['composite_score'] for s in selected)
        if total_score == 0:
            total_score = 1  # prevent div-by-zero

        allocations = []
        for stock in selected:
            raw_pct = (stock['composite_score'] / total_score) * 100
            alloc_pct = round(raw_pct, 1)
            alloc_amount = round(investment_amount * (alloc_pct / 100), 2)

            # Estimated shares (fractional)
            est_shares = round(alloc_amount / stock['current_price'], 4) if stock['current_price'] > 0 else 0

            # Expected return contribution (proportion of this stock's projected return)
            stock_return = alloc_amount * (stock['projected_return_pct'] / 100)

            allocations.append({
                **stock,
                'allocation_pct': alloc_pct,
                'allocation_amount': alloc_amount,
                'estimated_shares': est_shares,
                'expected_return_contribution': round(stock_return, 2),
            })

        # Normalize percentages to exactly 100%
        total_pct = sum(a['allocation_pct'] for a in allocations)
        if total_pct != 100 and len(allocations) > 0:
            diff = 100 - total_pct
            allocations[0]['allocation_pct'] = round(allocations[0]['allocation_pct'] + diff, 1)
            allocations[0]['allocation_amount'] = round(
                investment_amount * (allocations[0]['allocation_pct'] / 100), 2)

        return allocations

    def _compute_portfolio_analytics(self, allocations, investment_amount,
                                      target_return_pct, tf, risk_tolerance):
        """Compute portfolio-level summary analytics."""
        if not allocations:
            return {}

        # Weighted metrics
        total_alloc = sum(a['allocation_pct'] for a in allocations)
        w_volatility = sum(a['volatility'] * a['allocation_pct'] / total_alloc
                           for a in allocations)
        w_sharpe = sum(a['sharpe'] * a['allocation_pct'] / total_alloc
                       for a in allocations)
        w_projected_return = sum(a['projected_return_pct'] * a['allocation_pct'] / total_alloc
                                 for a in allocations)
        w_confidence = sum(a['confidence'] * a['allocation_pct'] / total_alloc
                           for a in allocations)
        total_expected_return = sum(a['expected_return_contribution'] for a in allocations)

        # Diversification score (based on sector spread + count)
        sectors = set(a['sector'] for a in allocations)
        max_single_pct = max(a['allocation_pct'] for a in allocations)
        diversification_score = min(100, (
            len(sectors) / len(AVAILABLE_SECTORS) * 40
            + len(allocations) / 10 * 30
            + (100 - max_single_pct) / 100 * 30
        ))

        # Portfolio risk score (0-100, lower = less risky)
        risk_score = min(100, max(0,
            w_volatility * 1.5
            + (100 - w_confidence) * 0.3
            + max_single_pct * 0.2
        ))

        # Risk label
        if risk_score < 30:
            risk_label = 'Low Risk'
            risk_color = '#10b981'
        elif risk_score < 55:
            risk_label = 'Moderate Risk'
            risk_color = '#f59e0b'
        elif risk_score < 75:
            risk_label = 'High Risk'
            risk_color = '#ef4444'
        else:
            risk_label = 'Very High Risk'
            risk_color = '#dc2626'

        # Projected portfolio value over time (monthly)
        months = tf['months']
        monthly_return = w_projected_return / (months if months > 0 else 12) / 100
        trajectory = []
        value = investment_amount
        for m in range(months + 1):
            trajectory.append({
                'month': m,
                'value': round(value, 2),
            })
            value *= (1 + monthly_return)

        # Target feasibility
        needed_return = target_return_pct
        feasibility = 'Likely Achievable' if w_projected_return >= needed_return * 0.8 else (
            'Stretch Goal' if w_projected_return >= needed_return * 0.5 else 'Ambitious Target')
        feasibility_color = '#10b981' if feasibility == 'Likely Achievable' else (
            '#f59e0b' if feasibility == 'Stretch Goal' else '#ef4444')

        return {
            'weighted_volatility': round(w_volatility, 2),
            'weighted_sharpe': round(w_sharpe, 2),
            'weighted_projected_return': round(w_projected_return, 2),
            'weighted_confidence': round(w_confidence, 1),
            'total_expected_return': round(total_expected_return, 2),
            'total_expected_return_pct': round(
                (total_expected_return / investment_amount) * 100, 2) if investment_amount > 0 else 0,
            'diversification_score': round(diversification_score, 1),
            'risk_score': round(risk_score, 1),
            'risk_label': risk_label,
            'risk_color': risk_color,
            'sectors': list(sectors),
            'num_sectors': len(sectors),
            'num_stocks': len(allocations),
            'trajectory': trajectory,
            'target_feasibility': feasibility,
            'target_feasibility_color': feasibility_color,
            'estimated_final_value': round(trajectory[-1]['value'], 2) if trajectory else investment_amount,
        }

    @staticmethod
    def _generate_reasoning(ticker, name, recommendation, confidence,
                            volatility, sharpe, mean_return_pct,
                            risk_rating, signals):
        """Build a human-readable reasoning summary for a stock pick."""
        parts = []

        # Recommendation
        rec_map = {
            'STRONG BUY': f'{name} shows strong bullish signals across multiple indicators.',
            'BUY': f'{name} displays bullish technical momentum.',
            'HOLD': f'{name} is in a neutral consolidation phase.',
            'SELL': f'{name} shows bearish signals — included for diversification.',
            'STRONG SELL': f'{name} is under selling pressure — minimal allocation.',
        }
        parts.append(rec_map.get(recommendation,
                                 f'{name} has a {recommendation} signal.'))

        # Risk profile
        parts.append(
            f"Risk profile: {risk_rating['rating']} "
            f"(volatility {volatility:.1f}%, Sharpe {sharpe:.2f})."
        )

        # Signal detail
        sig_details = signals.get('signals', {})
        bullish = [k for k, v in sig_details.items()
                   if v.get('signal') in ('BUY',)]
        bearish = [k for k, v in sig_details.items()
                   if v.get('signal') in ('SELL',)]
        if bullish:
            parts.append(f"Bullish on: {', '.join(bullish)}.")
        if bearish:
            parts.append(f"Caution from: {', '.join(bearish)}.")

        # Return
        if mean_return_pct > 0:
            parts.append(
                f"Annualized historical return: +{mean_return_pct:.1f}%.")
        else:
            parts.append(
                f"Annualized historical return: {mean_return_pct:.1f}%.")

        return ' '.join(parts)
