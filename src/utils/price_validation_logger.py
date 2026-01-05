"""
Thread-safe Price Validation Logger
Handles concurrent writes to price validation log
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import threading

# Thread lock for file operations
_log_lock = threading.Lock()

class PriceValidationLogger:
    """Thread-safe logger for price validation warnings"""
    
    def __init__(self, log_file: str = "logs/price_validation_log.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file if doesn't exist
        if not self.log_file.exists():
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump({"warnings": []}, f, ensure_ascii=False, indent=2)
    
    def log_warnings(self, warnings: List[Dict], video_id: str, video_title: str = "") -> None:
        """
        Thread-safe logging of price warnings
        
        Args:
            warnings: List of warning dictionaries from validator
            video_id: YouTube video ID
            video_title: Optional video title
        """
        if not warnings:
            return
        
        timestamp = datetime.now().isoformat()
        
        # Prepare log entries
        entries = []
        for warning in warnings:
            entries.append({
                "timestamp": timestamp,
                "video_id": video_id,
                "video_title": video_title,
                "ticker": warning['ticker'],
                "stated_price": warning['stated_price'],
                "52w_high": warning['52w_high'],
                "52w_low": warning['52w_low'],
                "current_price": warning.get('current'),
                "severity": warning['severity'],
                "deviation_pct": warning['deviation_pct'],
                "message": warning['message'],
                "context": warning['context'],
                "learned": False  # For future auto-learning
            })
        
        # Thread-safe write
        with _log_lock:
            try:
                # Read existing logs
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Append new warnings
                data['warnings'].extend(entries)
                
                # Write back
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print(f"   ✅ Logged {len(entries)} price warnings to {self.log_file}")
                
            except Exception as e:
                print(f"   ❌ Failed to log price warnings: {e}")
    
    def get_unlearned_warnings(self) -> List[Dict]:
        """Get warnings that haven't been learned yet"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return [w for w in data.get('warnings', []) if not w.get('learned', False)]
        except Exception as e:
            print(f"   ⚠️ Failed to read unlearned warnings: {e}")
            return []
    
    def mark_as_learned(self, warning_ids: List[int]) -> None:
        """Mark warnings as learned (for future auto-learning integration)"""
        with _log_lock:
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for idx in warning_ids:
                    if idx < len(data['warnings']):
                        data['warnings'][idx]['learned'] = True
                
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"   ❌ Failed to mark warnings as learned: {e}")
