"""
Alert System Module
Create and manage price alerts, prediction alerts, and pattern alerts
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSystem:
    """Manage stock alerts and notifications"""
    
    def __init__(self, alerts_file: str = 'data/alerts.json'):
        """Initialize alert system"""
        self.alerts_file = alerts_file
        self.alerts = self._load_alerts()
        
    def _load_alerts(self) -> List[Dict]:
        """Load alerts from file"""
        if os.path.exists(self.alerts_file):
            try:
                with open(self.alerts_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading alerts: {e}")
                return []
        return []
    
    def _save_alerts(self):
        """Save alerts to file"""
        try:
            os.makedirs(os.path.dirname(self.alerts_file), exist_ok=True)
            with open(self.alerts_file, 'w') as f:
                json.dump(self.alerts, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving alerts: {e}")
    
    def create_price_alert(self, ticker: str, alert_type: str, price: float, 
                          condition: str) -> str:
        """
        Create a price alert
        
        Args:
            ticker: Stock ticker symbol
            alert_type: 'price' or 'price_change'
            price: Target price or change percentage
            condition: 'above', 'below', 'crosses_above', 'crosses_below'
            
        Returns:
            Alert ID
        """
        alert_id = f"{ticker}_{alert_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        alert = {
            'id': alert_id,
            'ticker': ticker,
            'type': alert_type,
            'condition': condition,
            'target_value': price,
            'created_at': datetime.now().isoformat(),
            'triggered': False,
            'active': True
        }
        
        self.alerts.append(alert)
        self._save_alerts()
        return alert_id
    
    def create_prediction_alert(self, ticker: str, predicted_change: float, 
                                condition: str) -> str:
        """
        Create a prediction-based alert
        
        Args:
            ticker: Stock ticker
            predicted_change: Predicted % change threshold
            condition: 'above' or 'below'
            
        Returns:
            Alert ID
        """
        alert_id = f"{ticker}_prediction_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        alert = {
            'id': alert_id,
            'ticker': ticker,
            'type': 'prediction',
            'condition': condition,
            'target_value': predicted_change,
            'created_at': datetime.now().isoformat(),
            'triggered': False,
            'active': True
        }
        
        self.alerts.append(alert)
        self._save_alerts()
        return alert_id
    
    def create_pattern_alert(self, ticker: str, pattern_name: str) -> str:
        """
        Create a pattern detection alert
        
        Args:
            ticker: Stock ticker
            pattern_name: Name of candlestick or chart pattern
            
        Returns:
            Alert ID
        """
        alert_id = f"{ticker}_pattern_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        alert = {
            'id': alert_id,
            'ticker': ticker,
            'type': 'pattern',
            'pattern': pattern_name,
            'created_at': datetime.now().isoformat(),
            'triggered': False,
            'active': True
        }
        
        self.alerts.append(alert)
        self._save_alerts()
        return alert_id
    
    def check_price_alerts(self, ticker: str, current_price: float, 
                          previous_price: Optional[float] = None) -> List[Dict]:
        """Check if any price alerts should trigger"""
        triggered = []
        
        for alert in self.alerts:
            if (alert['ticker'] == ticker and 
                alert['type'] == 'price' and 
                alert['active'] and 
                not alert['triggered']):
                
                should_trigger = False
                
                if alert['condition'] == 'above':
                    should_trigger = current_price > alert['target_value']
                elif alert['condition'] == 'below':
                    should_trigger = current_price < alert['target_value']
                elif alert['condition'] == 'crosses_above' and previous_price:
                    should_trigger = (previous_price <= alert['target_value'] and 
                                    current_price > alert['target_value'])
                elif alert['condition'] == 'crosses_below' and previous_price:
                    should_trigger = (previous_price >= alert['target_value'] and 
                                    current_price < alert['target_value'])
                
                if should_trigger:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now().isoformat()
                    alert['triggered_price'] = current_price
                    triggered.append(alert)
        
        if triggered:
            self._save_alerts()
        
        return triggered
    
    def check_prediction_alerts(self, ticker: str, predicted_change_pct: float) -> List[Dict]:
        """Check if any prediction alerts should trigger"""
        triggered = []
        
        for alert in self.alerts:
            if (alert['ticker'] == ticker and 
                alert['type'] == 'prediction' and 
                alert['active'] and 
                not alert['triggered']):
                
                should_trigger = False
                
                if alert['condition'] == 'above':
                    should_trigger = predicted_change_pct > alert['target_value']
                elif alert['condition'] == 'below':
                    should_trigger = predicted_change_pct < alert['target_value']
                
                if should_trigger:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now().isoformat()
                    alert['predicted_change'] = predicted_change_pct
                    triggered.append(alert)
        
        if triggered:
            self._save_alerts()
        
        return triggered
    
    def check_pattern_alerts(self, ticker: str, detected_patterns: List[str]) -> List[Dict]:
        """Check if any pattern alerts should trigger"""
        triggered = []
        
        for alert in self.alerts:
            if (alert['ticker'] == ticker and 
                alert['type'] == 'pattern' and 
                alert['active'] and 
                not alert['triggered']):
                
                if alert['pattern'] in detected_patterns:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now().isoformat()
                    triggered.append(alert)
        
        if triggered:
            self._save_alerts()
        
        return triggered
    
    def get_active_alerts(self, ticker: Optional[str] = None) -> List[Dict]:
        """Get all active alerts, optionally filtered by ticker"""
        if ticker:
            return [a for a in self.alerts if a['ticker'] == ticker and a['active']]
        return [a for a in self.alerts if a['active']]
    
    def get_triggered_alerts(self, ticker: Optional[str] = None) -> List[Dict]:
        """Get triggered alerts"""
        if ticker:
            return [a for a in self.alerts if a['ticker'] == ticker and a['triggered']]
        return [a for a in self.alerts if a['triggered']]
    
    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert"""
        initial_len = len(self.alerts)
        self.alerts = [a for a in self.alerts if a['id'] != alert_id]
        
        if len(self.alerts) < initial_len:
            self._save_alerts()
            return True
        return False
    
    def deactivate_alert(self, alert_id: str) -> bool:
        """Deactivate an alert"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['active'] = False
                self._save_alerts()
                return True
        return False
    
    def reset_alert(self, alert_id: str) -> bool:
        """Reset a triggered alert"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['triggered'] = False
                alert['active'] = True
                if 'triggered_at' in alert:
                    del alert['triggered_at']
                self._save_alerts()
                return True
        return False
    
    def clear_all_alerts(self, ticker: Optional[str] = None):
        """Clear all alerts or alerts for specific ticker"""
        if ticker:
            self.alerts = [a for a in self.alerts if a['ticker'] != ticker]
        else:
            self.alerts = []
        self._save_alerts()
