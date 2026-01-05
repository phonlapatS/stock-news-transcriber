#!/usr/bin/env python3
"""
Smart Pattern Cache
Multi-layer defense: Layer 4

เก็บ patterns ที่แก้ไขแล้วเพื่อใช้ซ้ำ (ไม่ต้องแก้ซ้ำ)
"""

import json
import os
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict


class PatternCache:
    """
    Smart pattern cache สำหรับเก็บการแก้ไขที่ทำแล้ว
    
    Features:
    1. Entity corrections (THAI → TISCO, etc.)
    2. Number word conversions
    3. Proper noun fixes
    4. ASR error patterns
    """
    
    def __init__(self, cache_file: str = "pattern_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
        # Cache TTL (Time To Live) - 30 days
        self.ttl_days = 30
    
    def _load_cache(self) -> Dict:
        """Load cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save pattern cache: {e}")
    
    def _is_expired(self, timestamp: str) -> bool:
        """Check if cache entry is expired"""
        try:
            entry_time = datetime.fromisoformat(timestamp)
            expiry_time = entry_time + timedelta(days=self.ttl_days)
            return datetime.now() > expiry_time
        except:
            return True
    
    def get(self, pattern_type: str, key: str) -> Optional[str]:
        """
        Get cached pattern
        
        Args:
            pattern_type: 'entity', 'number', 'proper_noun', 'asr_error'
            key: Pattern key (e.g., 'THAI ESG', 'สามบาท')
            
        Returns:
            Cached value or None
        """
        if pattern_type not in self.cache:
            return None
        
        if key not in self.cache[pattern_type]:
            return None
        
        entry = self.cache[pattern_type][key]
        
        # Check expiry
        if self._is_expired(entry.get('timestamp', '')):
            # Remove expired entry
            del self.cache[pattern_type][key]
            self._save_cache()
            return None
        
        return entry.get('value')
    
    def set(self, pattern_type: str, key: str, value: str, confidence: float = 1.0):
        """
        Cache a pattern
        
        Args:
            pattern_type: 'entity', 'number', 'proper_noun', 'asr_error'
            key: Pattern key
            value: Corrected value
            confidence: Confidence score (0.0-1.0)
        """
        if pattern_type not in self.cache:
            self.cache[pattern_type] = {}
        
        self.cache[pattern_type][key] = {
            'value': value,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'count': self.cache[pattern_type].get(key, {}).get('count', 0) + 1
        }
        
        self._save_cache()
    
    def get_entity_correction(self, entity: str) -> Optional[str]:
        """Get cached entity correction"""
        return self.get('entity', entity.lower())
    
    def cache_entity_correction(self, original: str, corrected: str, confidence: float = 1.0):
        """Cache entity correction"""
        self.set('entity', original.lower(), corrected, confidence)
    
    def get_number_conversion(self, number_word: str) -> Optional[str]:
        """Get cached number word conversion"""
        return self.get('number', number_word.lower())
    
    def cache_number_conversion(self, word: str, number: str):
        """Cache number word conversion"""
        self.set('number', word.lower(), number, confidence=1.0)
    
    def get_proper_noun_fix(self, noun: str) -> Optional[str]:
        """Get cached proper noun fix"""
        return self.get('proper_noun', noun.lower())
    
    def cache_proper_noun_fix(self, original: str, fixed: str):
        """Cache proper noun fix"""
        self.set('proper_noun', original.lower(), fixed, confidence=1.0)
    
    def get_asr_error_fix(self, error: str) -> Optional[str]:
        """Get cached ASR error fix"""
        return self.get('asr_error', error.lower())
    
    def cache_asr_error_fix(self, error: str, fix: str):
        """Cache ASR error fix"""
        self.set('asr_error', error.lower(), fix, confidence=0.9)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        stats = {
            'total_entries': 0,
            'by_type': {},
            'expired_entries': 0
        }
        
        for pattern_type, entries in self.cache.items():
            count = len(entries)
            stats['by_type'][pattern_type] = count
            stats['total_entries'] += count
            
            # Count expired
            for key, entry in entries.items():
                if self._is_expired(entry.get('timestamp', '')):
                    stats['expired_entries'] += 1
        
        return stats
    
    def cleanup_expired(self):
        """Remove expired entries"""
        cleaned = False
        
        for pattern_type in list(self.cache.keys()):
            for key in list(self.cache[pattern_type].keys()):
                entry = self.cache[pattern_type][key]
                if self._is_expired(entry.get('timestamp', '')):
                    del self.cache[pattern_type][key]
                    cleaned = True
        
        if cleaned:
            self._save_cache()
            print(f"   🧹 Cleaned up expired cache entries")


# Global cache instance
_pattern_cache = None

def get_pattern_cache() -> PatternCache:
    """Get global pattern cache instance"""
    global _pattern_cache
    if _pattern_cache is None:
        _pattern_cache = PatternCache()
    return _pattern_cache

