#!/usr/bin/env python3
"""
Pre-processing Layer - แก้ไขปัญหาก่อนส่ง LLM
Multi-layer defense: Layer 1

Features:
1. Filler word removal (regex-based, ไม่พึ่ง LLM)
2. Number word conversion (Thai → Arabic)
3. Proper noun pattern matching (Dow Jones, NASDAQ, etc.)
4. ASR error pattern correction
"""

import re
from typing import Tuple, Dict, List
from collections import OrderedDict


class TextPreprocessor:
    """
    Pre-processing layer สำหรับแก้ไขปัญหาที่แก้ได้ด้วย pattern matching
    ไม่พึ่ง LLM → เร็ว, แม่นยำ, ไม่เปลือง quota
    """
    
    def __init__(self):
        # Filler words ที่ต้องลบทิ้ง (comprehensive list)
        self.filler_words = [
            # Common fillers
            r'\bนะฮะ\b', r'\bเนาะ\b', r'\bอ่า\b', r'\bเอ่อ\b', r'\bอืม\b',
            r'\bจ้า\b', r'\bจ๊ะ\b', r'\bอ๋อ\b', r'\bเออ\b',
            
            # Polite particles (ลบเมื่อไม่จำเป็น)
            r'\bครับ\b', r'\bค่ะ\b', r'\bนะครับ\b', r'\bนะคะ\b',
            r'\bครับผม\b', r'\bค่ะคุณ\b',
            
            # Hesitation
            r'\bเอ่อ\s+', r'\bอืม\s+', r'\bอ่า\s+',
        ]
        
        # Number word mappings (Thai → Arabic)
        self.number_words = {
            'ศูนย์': '0', 'หนึ่ง': '1', 'สอง': '2', 'สาม': '3', 'สี่': '4',
            'ห้า': '5', 'หก': '6', 'เจ็ด': '7', 'แปด': '8', 'เก้า': '9',
            'สิบ': '10', 'ยี่สิบ': '20', 'สามสิบ': '30', 'สี่สิบ': '40',
            'ห้าสิบ': '50', 'หกสิบ': '60', 'เจ็ดสิบ': '70', 'แปดสิบ': '80',
            'เก้าสิบ': '90',
            'ร้อย': '100', 'พัน': '1000', 'หมื่น': '10000', 'แสน': '100000',
            'ล้าน': '1000000',
            'จุด': '.', 'ครึ่ง': '0.5'
        }
        
        # Proper noun patterns (regex-based)
        self.proper_noun_patterns = [
            # Dow Jones
            (r'\b(ดาว\s*โจนส์|ดาว\s*โจรน์|ดาเทา|Datao)\b', 'Dow Jones'),
            # NASDAQ
            (r'\b(แนสแด็ก|นัสแด็ก|นาสแด็ก)\b', 'NASDAQ'),
            # S&P 500
            (r'\b(เอส\s*แอนด์\s*พี|เอส\s*แอนด์\s*พี\s*ห้าร้อย|S\s*&\s*P\s*500)\b', 'S&P 500'),
            # SET Index
            (r'\b(เซ็ต\s*เด็ก|เซท\s*เด็ก|เซน\s*เด็ก|เซ็น\s*เด็ก|เซ็ด\s*เด็ก)\b', 'SET Index'),
        ]
        
        # [IMPROVED] Load ASR error patterns from asr_errors.json
        self.asr_error_patterns = self._load_asr_error_patterns()
        
        # Compile regex patterns for performance
        self.filler_pattern = re.compile('|'.join(self.filler_words), re.IGNORECASE)
        
    def remove_filler_words(self, text: str) -> Tuple[str, int]:
        """
        ลบ filler words ออกจากข้อความ
        
        Args:
            text: ข้อความต้นฉบับ
            
        Returns:
            Tuple of (cleaned_text, num_removed)
        """
        original = text
        cleaned = self.filler_pattern.sub('', text)
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        # Count removed (approximate)
        num_removed = len(re.findall('|'.join(self.filler_words), original, re.IGNORECASE))
        
        return cleaned, num_removed
    
    def convert_number_words(self, text: str) -> Tuple[str, int]:
        """
        แปลงคำตัวเลขภาษาไทยเป็นตัวเลขอารบิก
        
        Examples:
            "สามบาท" → "3 บาท"
            "สองพันบาท" → "2,000 บาท"
            "หนึ่งจุดห้า" → "1.5"
        
        Args:
            text: ข้อความต้นฉบับ
            
        Returns:
            Tuple of (converted_text, num_conversions)
        """
        converted = text
        num_conversions = 0
        
        # Pattern: number word + "บาท" or number word + "จุด" + number
        patterns = [
            # Simple numbers: "สามบาท" → "3 บาท"
            (r'\b(หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ)\s*บาท\b', 
             lambda m: f"{self._word_to_number(m.group(1))} บาท"),
            
            # Complex numbers: "สองพันบาท" → "2,000 บาท"
            (r'\b(หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ)\s*(ร้อย|พัน|หมื่น|แสน|ล้าน)\s*บาท\b',
             lambda m: f"{self._word_to_number(m.group(1))}{self._word_to_number(m.group(2))} บาท"),
            
            # Decimals: "หนึ่งจุดห้า" → "1.5"
            (r'\b(หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ)\s*จุด\s*(หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ)\b',
             lambda m: f"{self._word_to_number(m.group(1))}.{self._word_to_number(m.group(2))}"),
        ]
        
        for pattern, replacement in patterns:
            matches = list(re.finditer(pattern, converted))
            if matches:
                # Replace from end to start to preserve positions
                for match in reversed(matches):
                    converted = converted[:match.start()] + replacement(match) + converted[match.end():]
                    num_conversions += 1
        
        return converted, num_conversions
    
    def _word_to_number(self, word: str) -> str:
        """Convert Thai number word to Arabic digit"""
        return self.number_words.get(word, word)
    
    def _load_asr_error_patterns(self) -> List[Tuple[str, str]]:
        """Load ASR error patterns from asr_errors.json"""
        patterns = []
        
        try:
            import json
            import os
            
            errors_file = "asr_errors.json"
            if os.path.exists(errors_file):
                with open(errors_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Extract common patterns (if structured)
                if isinstance(data, dict) and 'errors' in data:
                    error_list = data['errors']
                elif isinstance(data, list):
                    error_list = data
                else:
                    error_list = []
                
                # Get last 50 entries (most recent)
                for entry in error_list[-50:]:
                    if isinstance(entry, dict):
                        raw = entry.get('raw', '')
                        corrected = entry.get('corrected', '')
                        
                        if raw and corrected and raw != corrected:
                            # Create regex pattern (escape special chars)
                            pattern = re.escape(raw)
                            patterns.append((pattern, corrected))
        except Exception as e:
            # Silently fail and use fallback
            pass
        
        # Fallback to minimal hardcoded patterns
        if not patterns:
            patterns = [
                (r'\bทำงานสัตว์\b', 'ทำงานสิบ'),
            ]
        
        return patterns
    
    def fix_proper_nouns(self, text: str) -> Tuple[str, int]:
        """
        แก้ไข proper nouns ที่รู้จักแล้ว
        
        Examples:
            "ดาวโจนส์" → "Dow Jones"
            "แนสแด็ก" → "NASDAQ"
        
        Args:
            text: ข้อความต้นฉบับ
            
        Returns:
            Tuple of (fixed_text, num_fixes)
        """
        fixed = text
        num_fixes = 0
        
        for pattern, replacement in self.proper_noun_patterns:
            matches = list(re.finditer(pattern, fixed, re.IGNORECASE))
            if matches:
                # Replace from end to start
                for match in reversed(matches):
                    fixed = fixed[:match.start()] + replacement + fixed[match.end():]
                    num_fixes += 1
        
        return fixed, num_fixes
    
    def fix_asr_errors(self, text: str) -> Tuple[str, int]:
        """
        แก้ไข ASR errors ที่รู้จักแล้ว
        
        Args:
            text: ข้อความต้นฉบับ
            
        Returns:
            Tuple of (fixed_text, num_fixes)
        """
        fixed = text
        num_fixes = 0
        
        for pattern, replacement in self.asr_error_patterns:
            matches = list(re.finditer(pattern, fixed, re.IGNORECASE))
            if matches:
                for match in reversed(matches):
                    fixed = fixed[:match.start()] + replacement + fixed[match.end():]
                    num_fixes += 1
        
        # Clean up multiple spaces
        fixed = re.sub(r'\s+', ' ', fixed)
        
        return fixed, num_fixes
    
    def preprocess(self, text: str, verbose: bool = False) -> Tuple[str, Dict[str, int]]:
        """
        Pre-process ข้อความทั้งหมด
        
        Args:
            text: ข้อความต้นฉบับ
            verbose: แสดงสถิติหรือไม่
            
        Returns:
            Tuple of (preprocessed_text, stats_dict)
        """
        stats = {
            'filler_removed': 0,
            'numbers_converted': 0,
            'proper_nouns_fixed': 0,
            'asr_errors_fixed': 0
        }
        
        processed = text
        
        # Step 1: Remove filler words
        processed, stats['filler_removed'] = self.remove_filler_words(processed)
        
        # Step 2: Convert number words
        processed, stats['numbers_converted'] = self.convert_number_words(processed)
        
        # Step 3: Fix proper nouns
        processed, stats['proper_nouns_fixed'] = self.fix_proper_nouns(processed)
        
        # Step 4: Fix ASR errors
        processed, stats['asr_errors_fixed'] = self.fix_asr_errors(processed)
        
        if verbose:
            total_fixes = sum(stats.values())
            if total_fixes > 0:
                print(f"   🔧 Pre-processing: {total_fixes} fixes applied")
                if stats['filler_removed'] > 0:
                    print(f"      - Removed {stats['filler_removed']} filler words")
                if stats['numbers_converted'] > 0:
                    print(f"      - Converted {stats['numbers_converted']} number words")
                if stats['proper_nouns_fixed'] > 0:
                    print(f"      - Fixed {stats['proper_nouns_fixed']} proper nouns")
                if stats['asr_errors_fixed'] > 0:
                    print(f"      - Fixed {stats['asr_errors_fixed']} ASR errors")
        
        return processed, stats


# Convenience function
def preprocess_text(text: str, verbose: bool = False) -> str:
    """
    Pre-process ข้อความ (convenience function)
    
    Args:
        text: ข้อความต้นฉบับ
        verbose: แสดงสถิติหรือไม่
        
    Returns:
        Pre-processed text
    """
    preprocessor = TextPreprocessor()
    processed, _ = preprocessor.preprocess(text, verbose=verbose)
    return processed

