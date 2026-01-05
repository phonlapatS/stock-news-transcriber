# smart_cache.py
"""
Smart Pattern Cache - Pre-LLM text corrections
Replaces known ASR errors instantly without API calls
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

class SmartPatternCache:
    """Pattern-based replacement for known ASR errors"""
    
    def __init__(self, cache_file: str = "pattern_cache.json"):
        self.cache_file = Path(cache_file)
        self.patterns = {}  # {pattern: replacement}
        self.term_cache = {}  # {exact_term: corrected_term}
        self.stats = {"hits": 0, "misses": 0}
        self.load()
        self._init_default_patterns()
    
    def _init_default_patterns(self):
        """Initialize with known financial term patterns"""
        default_patterns = {
            # USO Phase variations
            r'ยูโซ\s*เฟส\s*หนึ่ง': 'USO Phase 1',
            r'ยูโซ\s*เฟส\s*สอง': 'USO Phase 2',
            r'ยูโซ\s*เฟส\s*สาม': 'USO Phase 3',
            r'ยูโซเฟสหนึ่ง': 'USO Phase 1',
            r'ยูโซเฟสสอง': 'USO Phase 2',
            r'ยูโซเฟสสาม': 'USO Phase 3',
            r'User\s*Fest\s*(\d+)': r'USO Phase \1',
            
            # Quarter/Period
            r'คิวสาม': 'Q3',
            r'คิวสี่': 'Q4',
            r'คิวหนึ่ง': 'Q1',
            r'คิวสอง': 'Q2',
            r'ไตรมาสที่\s*สาม': 'Q3',
            r'ไตรมาสที่\s*สี่': 'Q4',
            
            # Common ASR errors
            r'กิจการโซ': 'Aquaculture',
            r'อะลอก\s*คู\s*ลี': 'Aquaculture',
            r'Allockculy': 'Aquaculture',
            
            # Financial terms
            r'คิวออนคิว': 'QoQ',
            r'เออาร์พีดี': 'RPD',
            r'ทีพีดี': 'TPD',
        }
        
        for pattern, replacement in default_patterns.items():
            if pattern not in self.patterns:
                self.patterns[pattern] = replacement
    
    def load(self):
        """Load cache from file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.patterns = data.get("patterns", {})
                    self.term_cache = data.get("term_cache", {})
                print(f"✅ Loaded {len(self.patterns)} patterns + {len(self.term_cache)} terms from cache")
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}")
    
    def save(self):
        """Save cache to file"""
        try:
            data = {
                "patterns": self.patterns,
                "term_cache": self.term_cache,
                "stats": self.stats
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Error saving cache: {e}")
    
    def apply_patterns(self, text: str) -> Tuple[str, int]:
        """Apply pattern-based replacements
        
        Returns:
            (corrected_text, num_replacements)
        """
        corrected = text
        total_replacements = 0
        
        for pattern, replacement in self.patterns.items():
            try:
                new_text, count = re.subn(pattern, replacement, corrected, flags=re.IGNORECASE)
                if count > 0:
                    corrected = new_text
                    total_replacements += count
                    self.stats["hits"] += count
            except re.error:
                continue  # Skip invalid regex
        
        return corrected, total_replacements
    
    def apply_term_cache(self, text: str) -> Tuple[str, int]:
        """Apply exact term replacements
        
        Returns:
            (corrected_text, num_replacements)
        """
        corrected = text
        total_replacements = 0
        
        for raw, corrected_term in self.term_cache.items():
            if raw in corrected:
                count = corrected.count(raw)
                corrected = corrected.replace(raw, corrected_term)
                total_replacements += count
                self.stats["hits"] += count
        
        return corrected, total_replacements
    
    def process(self, text: str) -> Tuple[str, Dict]:
        """Full pre-processing with both patterns and terms
        
        Returns:
            (corrected_text, stats_dict)
        """
        # Stage 1: Pattern-based
        text, pattern_hits = self.apply_patterns(text)
        
        # Stage 2: Term cache
        text, term_hits = self.apply_term_cache(text)
        
        total_hits = pattern_hits + term_hits
        
        return text, {
            "pattern_replacements": pattern_hits,
            "term_replacements": term_hits,
            "total_replacements": total_hits,
            "cache_hit_rate": total_hits / max(1, len(text.split()))
        }
    
    def learn_from_correction(self, raw: str, corrected: str):
        """Learn new patterns from LLM corrections"""
        # Extract differences between raw and corrected
        # Simple implementation: find exact phrase replacements
        from difflib import SequenceMatcher
        
        matcher = SequenceMatcher(None, raw, corrected)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace' and (i2 - i1) > 3:  # Only learn significant changes
                raw_phrase = raw[i1:i2].strip()
                corrected_phrase = corrected[j1:j2].strip()
                
                if raw_phrase and corrected_phrase:
                    # Add to term cache
                    self.term_cache[raw_phrase] = corrected_phrase
    
    def add_pattern(self, pattern: str, replacement: str):
        """Manually add a new pattern"""
        self.patterns[pattern] = replacement
        self.save()
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "total_patterns": len(self.patterns),
            "total_terms": len(self.term_cache),
            "cache_hits": self.stats["hits"],
            "cache_misses": self.stats["misses"]
        }


# Global instance
_cache = None

def get_smart_cache() -> SmartPatternCache:
    """Get global smart cache instance"""
    global _cache
    if _cache is None:
        _cache = SmartPatternCache()
    return _cache


if __name__ == "__main__":
    # Test
    cache = SmartPatternCache()
    
    test_text = "บริษัท TFM ในไตรมาสที่สาม มียูโซเฟสหนึ่งและคิวสาม เติบโตดี"
    corrected, stats = cache.process(test_text)
    
    print(f"Original: {test_text}")
    print(f"Corrected: {corrected}")
    print(f"Stats: {stats}")
    
    # Expected: USO Phase 1, Q3 replacements
