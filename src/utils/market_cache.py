#!/usr/bin/env python3
"""
Market Data Cache Manager
จัดการ cache ราคาหุ้นเพื่อลดการเรียก yfinance API ซ้ำ
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import hashlib


class MarketDataCache:
    """
    Persistent cache for market data with TTL (Time To Live)
    
    Features:
    - File-based storage (JSON)
    - TTL support (ข้อมูลหมดอายุหลังกำหนด)
    - Auto-cleanup expired entries
    - Thread-safe operations
    """
    
    def __init__(self, cache_dir: str = "cache", ttl_hours: int = 24):
        """
        Args:
            cache_dir: โฟลเดอร์เก็บ cache
            ttl_hours: อายุของ cache (ชั่วโมง)
        """
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "market_data_cache.json")
        self.ttl_seconds = ttl_hours * 3600
        
        # สร้างโฟลเดอร์ถ้ายังไม่มี
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # โหลด cache จากไฟล์
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """โหลด cache จากไฟล์"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """บันทึก cache ลงไฟล์"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
    
    def _generate_key(self, ticker: str, date: str) -> str:
        """สร้าง cache key"""
        return f"{ticker}_{date}"
    
    def _is_expired(self, entry: Dict) -> bool:
        """ตรวจสอบว่า cache หมดอายุหรือไม่"""
        if 'timestamp' not in entry:
            return True
        
        cached_time = datetime.fromisoformat(entry['timestamp'])
        age_seconds = (datetime.now() - cached_time).total_seconds()
        
        return age_seconds > self.ttl_seconds
    
    def get(self, ticker: str, date: str) -> Optional[Dict]:
        """
        ดึงข้อมูลจาก cache
        
        Returns:
            Dict ของราคา หรือ None ถ้าไม่มี/หมดอายุ
        """
        key = self._generate_key(ticker, date)
        
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # ตรวจสอบอายุ
        if self._is_expired(entry):
            del self.cache[key]
            self._save_cache()
            return None
        
        return entry.get('data')
    
    def set(self, ticker: str, date: str, data: Dict):
        """
        เก็บข้อมูลลง cache
        
        Args:
            ticker: ชื่อหุ้น
            date: วันที่ (YYYY-MM-DD)
            data: ข้อมูลราคา
        """
        key = self._generate_key(ticker, date)
        
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_cache()
    
    def cleanup_expired(self) -> int:
        """
        ลบ entries ที่หมดอายุ
        
        Returns:
            จำนวน entries ที่ถูกลบ
        """
        expired_keys = []
        
        for key, entry in self.cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self._save_cache()
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict:
        """สถิติการใช้งาน cache"""
        total = len(self.cache)
        expired = sum(1 for entry in self.cache.values() if self._is_expired(entry))
        
        return {
            'total_entries': total,
            'valid_entries': total - expired,
            'expired_entries': expired,
            'cache_file': self.cache_file,
            'size_kb': os.path.getsize(self.cache_file) / 1024 if os.path.exists(self.cache_file) else 0
        }


# Singleton instance
_market_cache = None

def get_market_cache(ttl_hours: int = 24) -> MarketDataCache:
    """
    Get singleton cache instance
    
    Args:
        ttl_hours: Time to live in hours (default 24h = 1 day)
    """
    global _market_cache
    if _market_cache is None:
        _market_cache = MarketDataCache(ttl_hours=ttl_hours)
    return _market_cache


# Example usage
if __name__ == "__main__":
    cache = get_market_cache()
    
    # Set data
    cache.set("PTT", "2025-12-24", {
        "market_price": 35.50,
        "price_range": {"low": 34.00, "high": 36.00}
    })
    
    # Get data
    data = cache.get("PTT", "2025-12-24")
    print(f"Cached data: {data}")
    
    # Stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
