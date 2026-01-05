#!/usr/bin/env python3
"""
Post-processing Validator
Multi-layer defense: Layer 3

ตรวจสอบผลลัพธ์หลัง LLM เพื่อจับ hallucinations และ inconsistencies
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class PostProcessingValidator:
    """
    Post-processing validator สำหรับตรวจสอบผลลัพธ์หลัง LLM
    
    Features:
    1. Hallucination detection (entities ที่ไม่มีในต้นฉบับ)
    2. Format consistency check
    3. Entity correctness verification
    4. Context preservation check
    """
    
    def __init__(self, stock_context_manager=None, finance_term_manager=None):
        self.stock_ctx = stock_context_manager
        self.term_mgr = finance_term_manager
        
        # Patterns for entity extraction
        self.ticker_pattern = re.compile(r'\b([A-Z]{2,6})\b')
        self.price_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*บาท')
        
    def detect_hallucinations(self, raw_text: str, clean_text: str) -> List[Dict]:
        """
        ตรวจจับ hallucinations (entities ที่ไม่มีในต้นฉบับ)
        
        Args:
            raw_text: ข้อความต้นฉบับ (RAW)
            clean_text: ข้อความที่แก้ไขแล้ว (CLEAN)
            
        Returns:
            List of hallucination issues
        """
        issues = []
        
        # Extract entities from both texts
        raw_entities = set(self._extract_entities(raw_text))
        clean_entities = set(self._extract_entities(clean_text))
        
        # Find entities in CLEAN but not in RAW
        new_entities = clean_entities - raw_entities
        
        for entity in new_entities:
            # Check if it's a known entity (might be correction)
            if self._is_known_entity(entity):
                # Might be a correction, not hallucination
                continue
            
            # Check if it's a proper noun correction (Dow Jones, etc.)
            if self._is_proper_noun_correction(entity, raw_text):
                continue
            
            # Potential hallucination
            issues.append({
                'type': 'hallucination',
                'entity': entity,
                'severity': 'high',
                'description': f'Entity "{entity}" appears in CLEAN but not in RAW'
            })
        
        return issues
    
    def check_format_consistency(self, text: str) -> List[Dict]:
        """
        ตรวจสอบความสม่ำเสมอของ format
        
        Args:
            text: ข้อความที่ต้องการตรวจสอบ
            
        Returns:
            List of format issues
        """
        issues = []
        
        # Check for inconsistent number formatting
        # Pattern: บางที่ใช้ "1,000 บาท" บางที่ใช้ "1000 บาท"
        prices = self.price_pattern.findall(text)
        if prices:
            has_comma = any(',' in p for p in prices)
            no_comma = any(',' not in p and len(p) > 3 for p in prices)
            
            if has_comma and no_comma:
                issues.append({
                    'type': 'format_inconsistency',
                    'severity': 'low',
                    'description': 'Inconsistent number formatting (some with commas, some without)'
                })
        
        # Check for inconsistent ticker formatting
        # Pattern: บางที่ใช้ "PTT" บางที่ใช้ "ปตท"
        tickers = self.ticker_pattern.findall(text)
        if tickers:
            # Check if there are Thai words that might be tickers
            thai_ticker_pattern = re.compile(r'\b[ก-๙]{2,10}\b')
            thai_words = thai_ticker_pattern.findall(text)
            
            # If we have both English tickers and Thai words that might be tickers
            if tickers and len(thai_words) > len(tickers) * 2:
                issues.append({
                    'type': 'format_inconsistency',
                    'severity': 'medium',
                    'description': 'Mixed ticker formats (English and Thai)'
                })
        
        return issues
    
    def verify_entity_correctness(self, text: str) -> List[Dict]:
        """
        ตรวจสอบความถูกต้องของ entities
        
        Args:
            text: ข้อความที่ต้องการตรวจสอบ
            
        Returns:
            List of correctness issues
        """
        issues = []
        
        # Extract all tickers
        tickers = set(self.ticker_pattern.findall(text))
        
        for ticker in tickers:
            # Check if ticker is in knowledge base
            if self.stock_ctx:
                if ticker not in self.stock_ctx.all_tickers:
                    # Might be a fund or unknown entity
                    # Check context
                    context = self._get_entity_context(ticker, text)
                    if 'กองทุน' in context.lower():
                        # Might be a fund, not an error
                        continue
                    
                    issues.append({
                        'type': 'unknown_entity',
                        'entity': ticker,
                        'severity': 'medium',
                        'description': f'Ticker "{ticker}" not found in knowledge base'
                    })
        
        return issues
    
    def check_context_preservation(self, raw_text: str, clean_text: str) -> List[Dict]:
        """
        ตรวจสอบว่าบริบทถูกรักษาไว้หรือไม่
        
        Args:
            raw_text: ข้อความต้นฉบับ
            clean_text: ข้อความที่แก้ไขแล้ว
            
        Returns:
            List of context preservation issues
        """
        issues = []
        
        # Extract key phrases from RAW
        raw_phrases = self._extract_key_phrases(raw_text)
        clean_phrases = self._extract_key_phrases(clean_text)
        
        # Check if important phrases are missing
        missing_phrases = []
        for phrase in raw_phrases:
            if phrase not in clean_phrases:
                # Check if it's a significant phrase (has numbers or entities)
                if self._is_significant_phrase(phrase):
                    missing_phrases.append(phrase)
        
        if missing_phrases:
            issues.append({
                'type': 'context_loss',
                'severity': 'high',
                'description': f'Missing {len(missing_phrases)} significant phrases from RAW',
                'missing_phrases': missing_phrases[:5]  # Show first 5
            })
        
        return issues
    
    def validate(self, raw_text: str, clean_text: str) -> Dict:
        """
        Validate CLEAN output กับ RAW
        
        Args:
            raw_text: ข้อความต้นฉบับ (RAW)
            clean_text: ข้อความที่แก้ไขแล้ว (CLEAN)
            
        Returns:
            Validation result dict
        """
        all_issues = []
        
        # 1. Detect hallucinations
        hallucinations = self.detect_hallucinations(raw_text, clean_text)
        all_issues.extend(hallucinations)
        
        # 2. Check format consistency
        format_issues = self.check_format_consistency(clean_text)
        all_issues.extend(format_issues)
        
        # 3. Verify entity correctness
        entity_issues = self.verify_entity_correctness(clean_text)
        all_issues.extend(entity_issues)
        
        # 4. Check context preservation
        context_issues = self.check_context_preservation(raw_text, clean_text)
        all_issues.extend(context_issues)
        
        # Categorize by severity
        high_severity = [i for i in all_issues if i.get('severity') == 'high']
        medium_severity = [i for i in all_issues if i.get('severity') == 'medium']
        low_severity = [i for i in all_issues if i.get('severity') == 'low']
        
        return {
            'is_valid': len(high_severity) == 0,
            'total_issues': len(all_issues),
            'high_severity': len(high_severity),
            'medium_severity': len(medium_severity),
            'low_severity': len(low_severity),
            'issues': all_issues,
            'summary': {
                'hallucinations': len(hallucinations),
                'format_issues': len(format_issues),
                'entity_issues': len(entity_issues),
                'context_issues': len(context_issues)
            }
        }
    
    # Helper methods
    def _extract_entities(self, text: str) -> List[str]:
        """Extract all potential entities from text"""
        entities = self.ticker_pattern.findall(text)
        return entities
    
    def _is_known_entity(self, entity: str) -> bool:
        """Check if entity is in knowledge base"""
        if self.stock_ctx:
            return entity in self.stock_ctx.all_tickers
        return False
    
    def _is_proper_noun_correction(self, entity: str, raw_text: str) -> bool:
        """Check if entity might be a proper noun correction"""
        proper_nouns = ['Dow Jones', 'NASDAQ', 'S&P 500', 'SET Index']
        return entity in proper_nouns
    
    def _get_entity_context(self, entity: str, text: str, window: int = 50) -> str:
        """Get context around entity"""
        pattern = re.escape(entity)
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            return text[start:end]
        return ''
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases (sentences with numbers or entities)"""
        sentences = re.split(r'[.!?]\s+|\n+', text)
        key_phrases = []
        
        for sentence in sentences:
            # Check if sentence has numbers or entities
            if self.price_pattern.search(sentence) or self.ticker_pattern.search(sentence):
                key_phrases.append(sentence.strip())
        
        return key_phrases
    
    def _is_significant_phrase(self, phrase: str) -> bool:
        """Check if phrase is significant (has numbers or entities)"""
        return bool(self.price_pattern.search(phrase) or self.ticker_pattern.search(phrase))


# Convenience function
def validate_clean_output(raw_text: str, clean_text: str, stock_ctx=None, term_mgr=None) -> Dict:
    """
    Validate CLEAN output (convenience function)
    
    Args:
        raw_text: RAW transcript
        clean_text: CLEAN transcript
        stock_ctx: StockContextManager instance
        term_mgr: FinanceTermManager instance
        
    Returns:
        Validation result dict
    """
    validator = PostProcessingValidator(stock_ctx, term_mgr)
    return validator.validate(raw_text, clean_text)

