"""
Investment Planner Module — Production-Grade Portfolio Construction Engine.
Features: multi-market universes, feasibility engine, strategy-differentiated scoring,
goal-based risk derivation, allocation constraints, bull/base/bear projections.
"""
import math
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import streamlit as st

from src.data_fetcher import StockDataFetcher
from src.feature_engineering import FeatureEngineer
from src.trading_signals import TradingSignals
from src.risk_metrics import RiskAnalyzer
from src.stock_universes import (
    MARKET_CHOICES, STRATEGY_CHOICES, GOAL_MODES,
    get_sectors_for_market, get_universe, get_universe_size,
    US_STOCKS, INDIAN_STOCKS, INTERNATIONAL_STOCKS, MARKET_UNIVERSES,
)

logger = logging.getLogger(__name__)

# Re-export for app.py imports
__all__ = [
    'InvestmentPlanner', 'FeasibilityEngine',
    'MARKET_CHOICES', 'STRATEGY_CHOICES', 'GOAL_MODES', 'TIMEFRAME_MAP',
    'get_sectors_for_market', 'get_universe_size',
]

TIMEFRAME_MAP = {
    '3 Months': {'months': 3, 'years': 0.25, 'trading_days': 63},
    '6 Months': {'months': 6, 'years': 0.5, 'trading_days': 126},
    '1 Year':   {'months': 12, 'years': 1.0, 'trading_days': 252},
    '2 Years':  {'months': 24, 'years': 2.0, 'trading_days': 504},
    '3 Years':  {'months': 36, 'years': 3.0, 'trading_days': 756},
    '5 Years':  {'months': 60, 'years': 5.0, 'trading_days': 1260},
}

MAX_CANDIDATES = 30  # performance cap on data fetches

# ── Strategy configuration ────────────────────────────────────────────────
STRATEGY_WEIGHTS = {
    'Growth': {
        'signal': 0.20, 'momentum': 0.30, 'sharpe': 0.10,
        'vol_pen': 0.05, 'dd_pen': 0.10, 'sector': 0.20, 'div': 0.00, 'volume': 0.05,
        'preferred': ['Technology', 'Consumer Cyclical', 'Healthcare', 'Communication Services'],
        'sector_boost': 1.5, 'vol_tolerance': 1.3,
    },
    'Value': {
        'signal': 0.15, 'momentum': 0.10, 'sharpe': 0.30,
        'vol_pen': 0.20, 'dd_pen': 0.15, 'sector': 0.10, 'div': 0.00, 'volume': 0.00,
        'preferred': ['Financial Services', 'Consumer Defensive', 'Industrials', 'Healthcare'],
        'sector_boost': 1.4, 'vol_tolerance': 0.7,
    },
    'Dividend': {
        'signal': 0.10, 'momentum': 0.05, 'sharpe': 0.15,
        'vol_pen': 0.20, 'dd_pen': 0.10, 'sector': 0.15, 'div': 0.25, 'volume': 0.00,
        'preferred': ['Utilities', 'Consumer Defensive', 'Real Estate', 'Energy'],
        'sector_boost': 1.6, 'vol_tolerance': 0.5,
    },
    'Momentum': {
        'signal': 0.40, 'momentum': 0.15, 'sharpe': 0.05,
        'vol_pen': 0.05, 'dd_pen': 0.10, 'sector': 0.05, 'div': 0.00, 'volume': 0.20,
        'preferred': [],
        'sector_boost': 1.0, 'vol_tolerance': 1.2,
    },
    'AI Optimized': {
        'signal': 0.20, 'momentum': 0.20, 'sharpe': 0.20,
        'vol_pen': 0.15, 'dd_pen': 0.10, 'sector': 0.05, 'div': 0.00, 'volume': 0.10,
        'preferred': [],
        'sector_boost': 1.0, 'vol_tolerance': 1.0,
    },
}

# ── Feasibility Engine ────────────────────────────────────────────────────
class FeasibilityEngine:
    """Analyzes goal realism and derives required risk exposure."""

    @staticmethod
    def normalize_goal(amount, goal_mode, goal_value, years):
        """Convert any goal mode to required CAGR and target value."""
        if goal_mode == 'Target Return %':
            target_value = amount * (1 + goal_value / 100.0)
        elif goal_mode == 'Target Final Value':
            target_value = goal_value
        elif goal_mode == 'Target Profit':
            target_value = amount + goal_value
        else:
            target_value = amount * (1 + goal_value / 100.0)

        if target_value <= amount:
            return 0.0, target_value, 0.0

        if years <= 0:
            years = 1.0
        required_cagr = (target_value / amount) ** (1.0 / years) - 1.0
        annualized_return_pct = required_cagr * 100.0
        return required_cagr, target_value, annualized_return_pct

    @staticmethod
    def classify(required_cagr):
        """5-tier feasibility classification."""
        pct = required_cagr * 100
        if pct <= 8:
            return 'Conservative', '#10b981', 'Target appears achievable with diversified market exposure. Historical S&P 500 average is ~10% CAGR.'
        elif pct <= 18:
            return 'Realistic', '#3b82f6', 'Target requires above-average stock picking but is historically plausible with a growth-oriented portfolio.'
        elif pct <= 45:
            return 'Aggressive', '#f59e0b', 'Target requires concentrated positions and elevated volatility. Above-market performance needed.'
        elif pct <= 80:
            return 'Speculative', '#ef4444', 'Target exceeds typical equity returns. Very high risk of underperformance. Consider a longer timeframe or lower target.'
        else:
            return 'Unrealistic', '#dc2626', 'Requested return exceeds historically sustainable market returns. Capital loss is highly probable at this risk level.'

    @staticmethod
    def estimate_probability(required_cagr):
        """Rough probability estimate using historical equity return distribution."""
        # Historical US equity: mean ~10%, std ~16%
        mu, sigma = 0.10, 0.16
        if sigma == 0:
            return 50.0
        z = (required_cagr - mu) / sigma
        # Approximate CDF using error function
        prob_achieving = 0.5 * (1 - math.erf(z / math.sqrt(2)))
        return round(max(1, min(99, prob_achieving * 100)), 0)

    @staticmethod
    def derive_risk_profile(required_cagr, user_preference):
        """Derive actual risk profile from target. User preference is advisory."""
        pct = required_cagr * 100
        if pct <= 8:
            derived = 'Low'
        elif pct <= 18:
            derived = 'Medium'
        else:
            derived = 'High'

        # Check for mismatch
        risk_levels = {'Low': 0, 'Medium': 1, 'High': 2}
        mismatch = False
        mismatch_msg = ''
        if risk_levels.get(derived, 1) > risk_levels.get(user_preference, 1):
            mismatch = True
            mismatch_msg = (
                f'Your target requires {derived.lower()}-risk exposure, '
                f'but you selected {user_preference.lower()} risk preference. '
                f'The portfolio will be adjusted toward {derived.lower()} risk to pursue your target.'
            )

        return {
            'derived': derived,
            'user_preference': user_preference,
            'effective': derived if mismatch else user_preference,
            'mismatch': mismatch,
            'mismatch_message': mismatch_msg,
        }


# ── Main Planner ──────────────────────────────────────────────────────────
class InvestmentPlanner:
    """Production-grade investment planner with multi-market support."""

    def __init__(self):
        self.fetcher = StockDataFetcher()
        self.engineer = FeatureEngineer()

    def generate_plan(self, investment_amount, goal_mode, goal_value,
                      timeframe='1 Year', risk_preference='Medium',
                      market='US Stocks', sector_preferences=None,
                      strategy='AI Optimized', num_stocks=5,
                      progress_callback=None):
        if progress_callback is None:
            progress_callback = lambda p, m: None

        tf = TIMEFRAME_MAP.get(timeframe, TIMEFRAME_MAP['1 Year'])
        years = tf['years']

        # ── Feasibility Analysis ──
        progress_callback(0.02, 'Analyzing goal feasibility...')
        req_cagr, target_value, ann_return_pct = FeasibilityEngine.normalize_goal(
            investment_amount, goal_mode, goal_value, years)
        feas_label, feas_color, feas_text = FeasibilityEngine.classify(req_cagr)
        probability = FeasibilityEngine.estimate_probability(req_cagr)
        risk_profile = FeasibilityEngine.derive_risk_profile(req_cagr, risk_preference)
        effective_risk = risk_profile['effective']

        feasibility = {
            'required_cagr': round(req_cagr * 100, 2),
            'target_value': round(target_value, 2),
            'annualized_return_pct': round(ann_return_pct, 2),
            'label': feas_label, 'color': feas_color, 'text': feas_text,
            'probability': probability,
            'risk_profile': risk_profile,
        }

        # ── Candidate Selection ──
        progress_callback(0.05, 'Selecting candidate universe...')
        candidates = self._get_candidates(market, sector_preferences, strategy)

        # ── Score Candidates ──
        progress_callback(0.10, 'Fetching market data & scoring...')
        scored = self._score_candidates(candidates, effective_risk, strategy, tf, progress_callback)

        if not scored:
            return {'success': False, 'message': 'Unable to fetch data for any candidates. Try again.'}

        # ── Allocate ──
        progress_callback(0.85, 'Constructing portfolio...')
        selected = scored[:min(num_stocks, len(scored))]
        allocations = self._allocate(selected, investment_amount, effective_risk, strategy)

        # ── Analytics ──
        progress_callback(0.92, 'Computing analytics...')
        analytics = self._analytics(allocations, investment_amount, feasibility, tf, market)

        progress_callback(1.0, 'Investment plan ready!')
        return {
            'success': True,
            'allocations': allocations,
            'analytics': analytics,
            'feasibility': feasibility,
            'parameters': {
                'investment_amount': investment_amount,
                'goal_mode': goal_mode,
                'goal_value': goal_value,
                'target_value': target_value,
                'timeframe': timeframe,
                'timeframe_info': tf,
                'risk_preference': risk_preference,
                'effective_risk': effective_risk,
                'market': market,
                'sector_preferences': sector_preferences,
                'strategy': strategy,
                'num_stocks': num_stocks,
                'generated_at': datetime.now().isoformat(),
            },
        }

    # ── Candidate selection ───────────────────────────────────────────────
    def _get_candidates(self, market, sector_prefs, strategy):
        universe = get_universe(market)
        # Sector filter
        if sector_prefs:
            universe = {t: v for t, v in universe.items() if v['sector'] in sector_prefs}

        # If too many, prioritize strategy-aligned sectors, then cap
        sw = STRATEGY_WEIGHTS.get(strategy, STRATEGY_WEIGHTS['AI Optimized'])
        preferred = sw['preferred']

        if len(universe) > MAX_CANDIDATES:
            preferred_stocks = {t: v for t, v in universe.items() if v['sector'] in preferred}
            other_stocks = {t: v for t, v in universe.items() if v['sector'] not in preferred}
            # Take all preferred, fill remainder from others
            n_other = MAX_CANDIDATES - len(preferred_stocks)
            if n_other > 0:
                other_list = list(other_stocks.items())[:n_other]
                result = dict(preferred_stocks)
                result.update(dict(other_list))
                return result
            else:
                return dict(list(preferred_stocks.items())[:MAX_CANDIDATES])

        return universe

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch_and_prepare_data(_fetcher, _engineer, ticker):
        df = _fetcher.fetch_stock_data(ticker, period='1y')
        if df is None or df.empty or len(df) < 60:
            return None
        return _engineer.add_all_indicators(df)

    def _score_candidates(self, candidates, risk_level, strategy, tf, progress_cb):
        scored = []
        total = len(candidates)
        sw = STRATEGY_WEIGHTS.get(strategy, STRATEGY_WEIGHTS['AI Optimized'])

        for idx, (ticker, info) in enumerate(candidates.items()):
            pct = 0.10 + 0.70 * ((idx + 1) / total)
            progress_cb(pct, f'Analyzing {info["name"]} ({ticker})...')
            try:
                df = self._fetch_and_prepare_data(self.fetcher, self.engineer, ticker)
                if df is None:
                    continue

                signals = TradingSignals.get_comprehensive_signals(df)
                overall = signals['overall']
                risk_m = RiskAnalyzer.get_comprehensive_risk_metrics(df)
                risk_rating = RiskAnalyzer.get_risk_rating(risk_m)

                buy_sc = overall['buy_score']
                sell_sc = overall['sell_score']
                confidence = overall['confidence']
                vol = risk_m['annualized_volatility_pct']
                sharpe = risk_m['sharpe_ratio']
                max_dd = abs(risk_m['max_drawdown_pct'])
                mean_ret = risk_m['mean_return_pct']
                price = float(df['Close'].iloc[-1])

                # Volume confirmation
                vol_ratio = 1.0
                if 'Volume_Ratio' in df.columns:
                    vol_ratio = float(df['Volume_Ratio'].iloc[-1])

                composite = self._composite(
                    buy_sc, sell_sc, confidence, sharpe, vol, max_dd, mean_ret,
                    info, sw, risk_level, vol_ratio)

                # ── Financial Realism Calibration ──
                # 1. Baseline Priors per Strategy
                baseline_cagrs = {
                    'Dividend': 0.07,
                    'Value': 0.085,
                    'AI Optimized': 0.11,
                    'Growth': 0.14,
                    'Momentum': 0.17
                }
                base_cagr = baseline_cagrs.get(strategy, 0.11)
                
                # Adjust baseline based on risk preference (advisory nudge)
                risk_nudge = {'Low': -0.02, 'Medium': 0.0, 'High': 0.03}.get(risk_level, 0.0)
                base_cagr += risk_nudge

                # 2. Historical Bounding & Confidence Shrinkage
                # risk_m['mean_return'] is already annualized return.
                hist_cagr = max(-0.15, min(risk_m['mean_return'], 0.25))
                conf_weight = min(confidence / 100.0, 0.85) # Never trust ML 100%
                shrunk_cagr = (hist_cagr * conf_weight) + (base_cagr * (1 - conf_weight))

                # 3. ML Alpha Bounding
                ml_signal = (buy_sc - sell_sc) / 100.0  # -1.0 to 1.0
                max_alpha = 0.06 if strategy in ('Growth', 'Momentum') else 0.04
                ml_alpha = ml_signal * max_alpha

                # 4. Volatility Drag Integration (Geometric compounding penalty)
                annual_vol = risk_m['volatility'] # typically 0.15 to 0.60
                vol_drag = (annual_vol ** 2) / 2.0

                expected_cagr = shrunk_cagr + ml_alpha - vol_drag
                
                # Hard bounds for institutional realism
                expected_cagr = max(-0.05, min(expected_cagr, 0.35))
                
                # 5. Timeframe Projection (Geometric)
                years = tf['years']
                proj_ret = ((1 + expected_cagr) ** years - 1) * 100
                daily_std = annual_vol / math.sqrt(252) if annual_vol > 0 else 0.01

                reasoning = self._reasoning(
                    info['name'], overall['recommendation'], confidence,
                    vol, sharpe, mean_ret, risk_rating, signals)

                scored.append({
                    'ticker': ticker, 'name': info['name'], 'sector': info['sector'],
                    'div_tier': info.get('div_tier', 'low'),
                    'current_price': price, 'composite_score': composite,
                    'buy_score': buy_sc, 'sell_score': sell_sc,
                    'confidence': confidence, 'recommendation': overall['recommendation'],
                    'volatility': vol, 'sharpe': sharpe, 'max_drawdown': max_dd,
                    'mean_return_pct': mean_ret, 'projected_return_pct': proj_ret,
                    'expected_cagr': expected_cagr,
                    'daily_std': daily_std, 'risk_rating': risk_rating, 'reasoning': reasoning,
                })
            except Exception as e:
                logger.warning(f'Skipping {ticker}: {e}')
                continue

        scored.sort(key=lambda s: s['composite_score'], reverse=True)
        return scored

    def _composite(self, buy_sc, sell_sc, conf, sharpe, vol, max_dd, mean_ret,
                   info, sw, risk_level, vol_ratio):
        """Strategy-differentiated composite scoring."""
        # Normalized sub-scores (0-100 scale)
        signal = max(0, buy_sc - sell_sc) + conf * 0.5
        momentum = max(0, min(mean_ret * 2, 100))
        sharpe_s = max(0, min(sharpe * 25, 100))
        vol_pen = max(0, min(vol * 1.5, 100))
        dd_pen = max(0, min(max_dd, 100))
        vol_confirm = max(0, min(vol_ratio * 30, 100))

        # Dividend quality score
        div_map = {'high': 90, 'medium': 55, 'low': 20, 'none': 0}
        div_score = div_map.get(info.get('div_tier', 'low'), 20)

        # Sector alignment score
        sector_s = 0
        if info['sector'] in sw['preferred']:
            sector_s = 80 * sw['sector_boost']
        else:
            sector_s = 30

        # Weighted composite
        raw = (
            signal * sw['signal']
            + momentum * sw['momentum']
            + sharpe_s * sw['sharpe']
            - vol_pen * sw['vol_pen'] * (1.5 - {'Low': 0, 'Medium': 0.5, 'High': 1.0}.get(risk_level, 0.5))
            - dd_pen * sw['dd_pen']
            + sector_s * sw['sector']
            + div_score * sw['div']
            + vol_confirm * sw['volume']
        )

        # Risk-level multiplier: aggressive goals boost high-vol stocks
        risk_mult = {'Low': 0.8, 'Medium': 1.0, 'High': 1.2}.get(risk_level, 1.0)
        if risk_level == 'High':
            raw += momentum * 0.15  # extra momentum bonus for aggressive

        return round(max(0, raw * risk_mult), 2)

    # ── Allocation with constraints ───────────────────────────────────────
    def _allocate(self, selected, amount, risk_level, strategy):
        if not selected:
            return []

        total_score = sum(s['composite_score'] for s in selected) or 1
        MAX_SINGLE = 25.0
        MAX_SECTOR = 40.0
        MIN_ALLOC = 3.0

        # Initial score-proportional allocation
        allocs = []
        for s in selected:
            pct = (s['composite_score'] / total_score) * 100
            allocs.append({**s, 'allocation_pct': pct})

        # Clamp single-stock max
        for a in allocs:
            if a['allocation_pct'] > MAX_SINGLE:
                a['allocation_pct'] = MAX_SINGLE

        # Clamp sector max
        sector_totals = {}
        for a in allocs:
            sector_totals[a['sector']] = sector_totals.get(a['sector'], 0) + a['allocation_pct']
        for sector, total in sector_totals.items():
            if total > MAX_SECTOR:
                stocks_in_sector = [a for a in allocs if a['sector'] == sector]
                scale = MAX_SECTOR / total
                for a in stocks_in_sector:
                    a['allocation_pct'] *= scale

        # Enforce minimum
        for a in allocs:
            if a['allocation_pct'] < MIN_ALLOC:
                a['allocation_pct'] = MIN_ALLOC

        # Normalize to 100%
        total_pct = sum(a['allocation_pct'] for a in allocs)
        if total_pct > 0:
            for a in allocs:
                a['allocation_pct'] = round(a['allocation_pct'] / total_pct * 100, 1)

        # Fix rounding to exactly 100
        diff = 100.0 - sum(a['allocation_pct'] for a in allocs)
        if abs(diff) > 0 and allocs:
            allocs[0]['allocation_pct'] = round(allocs[0]['allocation_pct'] + diff, 1)

        # Compute dollar amounts
        for a in allocs:
            a['allocation_amount'] = round(amount * a['allocation_pct'] / 100, 2)
            a['estimated_shares'] = round(a['allocation_amount'] / a['current_price'], 4) if a['current_price'] > 0 else 0
            a['expected_return_contribution'] = round(a['allocation_amount'] * a['projected_return_pct'] / 100, 2)

        return allocs

    # ── Portfolio analytics ───────────────────────────────────────────────
    def _analytics(self, allocs, amount, feasibility, tf, market):
        if not allocs:
            return {}
        total_pct = sum(a['allocation_pct'] for a in allocs) or 1
        w = lambda key: sum(a[key] * a['allocation_pct'] / total_pct for a in allocs)

        w_vol = w('volatility')
        w_sharpe = w('sharpe')
        w_proj = w('projected_return_pct')
        w_cagr = w('expected_cagr')
        w_conf = w('confidence')
        total_exp_ret = sum(a['expected_return_contribution'] for a in allocs)

        sectors = set(a['sector'] for a in allocs)
        max_single = max(a['allocation_pct'] for a in allocs)
        all_sectors = get_sectors_for_market(market)
        div_score = min(100, len(sectors) / max(len(all_sectors), 1) * 40
                        + len(allocs) / 10 * 30 + (100 - max_single) / 100 * 30)

        risk_score = min(100, max(0, w_vol * 1.5 + (100 - w_conf) * 0.3 + max_single * 0.2))
        if risk_score < 30:
            rl, rc = 'Low Risk', '#10b981'
        elif risk_score < 55:
            rl, rc = 'Moderate Risk', '#f59e0b'
        elif risk_score < 75:
            rl, rc = 'High Risk', '#ef4444'
        else:
            rl, rc = 'Very High Risk', '#dc2626'

        # Bull / Base / Bear trajectories (Probabilistic geometric random walk approximation)
        months = tf['months']
        # Convert expected annual CAGR to monthly for compounding
        monthly_cagr = (1 + w_cagr) ** (1 / 12) - 1
        
        # Scenario variations (based on volatility)
        annual_vol_dec = w_vol / 100.0
        # For scenario trajectory, adjust the monthly growth rate
        monthly_bull = (1 + w_cagr + annual_vol_dec * 0.5) ** (1 / 12) - 1
        monthly_bear = (1 + max(w_cagr - annual_vol_dec * 0.7, -0.15)) ** (1 / 12) - 1

        trajectories = {'base': [], 'bull': [], 'bear': []}
        vals = {'base': amount, 'bull': amount, 'bear': amount}
        for m in range(months + 1):
            for scenario in trajectories:
                trajectories[scenario].append({'month': m, 'value': round(vals[scenario], 2)})
            vals['base'] *= (1 + monthly_cagr)
            vals['bull'] *= (1 + monthly_bull)
            vals['bear'] *= (1 + monthly_bear)

        return {
            'weighted_volatility': round(w_vol, 2),
            'weighted_sharpe': round(w_sharpe, 2),
            'weighted_projected_return': round(w_proj, 2),
            'weighted_expected_cagr': round(w_cagr * 100, 2),
            'weighted_confidence': round(w_conf, 1),
            'total_expected_return': round(total_exp_ret, 2),
            'total_expected_return_pct': round(total_exp_ret / amount * 100, 2) if amount else 0,
            'diversification_score': round(div_score, 1),
            'risk_score': round(risk_score, 1),
            'risk_label': rl, 'risk_color': rc,
            'sectors': list(sectors), 'num_sectors': len(sectors),
            'num_stocks': len(allocs),
            'trajectories': trajectories,
            'estimated_final_value': round(trajectories['base'][-1]['value'], 2),
            'estimated_final_bull': round(trajectories['bull'][-1]['value'], 2),
            'estimated_final_bear': round(trajectories['bear'][-1]['value'], 2),
        }

    # ── Reasoning ─────────────────────────────────────────────────────────
    @staticmethod
    def _reasoning(name, rec, conf, vol, sharpe, mean_ret, risk_rating, signals):
        parts = []
        rec_map = {
            'STRONG BUY': f'{name} shows strong bullish signals across multiple indicators.',
            'BUY': f'{name} displays bullish technical momentum.',
            'HOLD': f'{name} is in a neutral consolidation phase.',
            'SELL': f'{name} shows bearish signals — included for diversification.',
            'STRONG SELL': f'{name} is under selling pressure — minimal allocation.',
        }
        parts.append(rec_map.get(rec, f'{name} has a {rec} signal.'))
        parts.append(f'Risk: {risk_rating["rating"]} (vol {vol:.1f}%, Sharpe {sharpe:.2f}).')

        sig_d = signals.get('signals', {})
        bulls = [k for k, v in sig_d.items() if v.get('signal') == 'BUY']
        bears = [k for k, v in sig_d.items() if v.get('signal') == 'SELL']
        if bulls:
            parts.append(f'Bullish: {", ".join(bulls)}.')
        if bears:
            parts.append(f'Caution: {", ".join(bears)}.')
        sign = '+' if mean_ret > 0 else ''
        parts.append(f'Annualized return: {sign}{mean_ret:.1f}%.')
        return ' '.join(parts)
