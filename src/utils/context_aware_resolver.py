#!/usr/bin/env python3
"""
Context-Aware Entity Resolver
Multi-layer defense: Layer 2

แก้ปัญหา Context Blindness:
- แยกกองทุน (Mutual Funds) vs หุ้น (Stocks)
- Context-aware matching (ไม่แปลง THAI ESG → TISCO)
- Entity type detection (Fund/Stock/Index/Crypto)
"""

import re
from typing import Optional, Dict, Tuple, List
from collections import defaultdict


class ContextAwareResolver:
    """
    Context-aware entity resolver
    แยกประเภท entity และแก้ไขตามบริบท
    """
    
    def __init__(self, stock_context_manager=None, finance_term_manager=None):
        self.stock_ctx = stock_context_manager
        self.term_mgr = finance_term_manager
        
        # Fund indicators (คำที่บ่งชี้ว่าเป็นกองทุน)
        self.fund_indicators = [
            r'\bกองทุน\b',
            r'\bmutual\s*fund\b',
            r'\bfund\b',
            r'\bกองทุนรวม\b',
            r'\bกองทุนเปิด\b',
            r'\bกองทุนปิด\b',
        ]
        
        # Stock indicators (คำที่บ่งชี้ว่าเป็นหุ้น)
        self.stock_indicators = [
            r'\bหุ้น\b',
            r'\bstock\b',
            r'\bshare\b',
            r'\bหุ้นสามัญ\b',
            r'\bหุ้นบุริมสิทธิ\b',
        ]
        
        # Index indicators
        self.index_indicators = [
            r'\bindex\b',
            r'\bดัชนี\b',
            r'\bSET\s*Index\b',
            r'\bSET\s*50\b',
            r'\bSET\s*100\b',
        ]
        
        # Compile patterns
        self.fund_pattern = re.compile('|'.join(self.fund_indicators), re.IGNORECASE)
        self.stock_pattern = re.compile('|'.join(self.stock_indicators), re.IGNORECASE)
        self.index_pattern = re.compile('|'.join(self.index_indicators), re.IGNORECASE)
        
        # [IMPROVED] Load known funds from KB instead of hardcoding
        self.known_funds = self._load_known_funds_from_kb()
        
        # [IMPROVED] Increase context window for better accuracy
        self.context_window = 200  # Increased from 100
    
    def detect_entity_type(self, entity: str, context: str) -> str:
        """
        ตรวจจับประเภท entity จากบริบท
        
        Returns:
            'fund', 'stock', 'index', 'crypto', 'unknown'
        """
        context_lower = context.lower()
        entity_lower = entity.lower()
        
        # Check for fund indicators
        if self.fund_pattern.search(context):
            # Double-check: ถ้ามี "กองทุน" ในบริบท → เป็น fund
            if 'กองทุน' in context_lower:
                return 'fund'
        
        # Check for stock indicators
        if self.stock_pattern.search(context):
            return 'stock'
        
        # Check for index indicators
        if self.index_pattern.search(context) or entity_lower in ['set', 'set index', 'dow jones', 'nasdaq', 's&p 500']:
            return 'index'
        
        # Check if entity is in known funds
        if entity.upper() in self.known_funds:
            # Need to check context to be sure
            if 'กองทุน' in context_lower:
                return 'fund'
            else:
                return 'stock'  # Default to stock if ambiguous
        
        # Default: unknown (let LLM decide)
        return 'unknown'
    
    def _load_known_funds_from_kb(self) -> Dict[str, str]:
        """Load known funds from KB or finance_terms.json"""
        funds = {}
        
        # Try to load from finance_terms.json
        try:
            from config import FINANCE_TERM_FILE
            import json
            import os
            
            if os.path.exists(FINANCE_TERM_FILE):
                with open(FINANCE_TERM_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Look for Mutual Funds section
                if 'Mutual Funds' in data:
                    for canonical, aliases in data['Mutual Funds'].items():
                        funds[canonical.upper()] = canonical
                        for alias in aliases:
                            funds[alias.upper()] = canonical
        except:
            pass
        
        # Fallback to minimal hardcoded list (only if KB not available)
        if not funds:
            funds = {
                'THAI ESG': 'THAI ESG',
                'KKP': 'KKP',
            }
        
        return funds
    
    def resolve_with_context(self, entity: str, text: str, position: int = None) -> Optional[Dict]:
        """
        Resolve entity พร้อมตรวจสอบบริบท
        
        Args:
            entity: Entity ที่ต้องการ resolve
            text: ข้อความทั้งหมด
            position: ตำแหน่งของ entity ในข้อความ (optional)
            
        Returns:
            {
                'original': str,
                'resolved': str,
                'type': 'fund'|'stock'|'index'|'unknown',
                'confidence': float,
                'context': str
            } or None
        """
        if not entity or not entity.strip():
            return None
        
        # Find entity in text if position not provided
        if position is None:
            # Search for entity (case-insensitive)
            pattern = re.escape(entity)
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                return None
            position = match.start()
        
        # Extract context window
        start = max(0, position - self.context_window)
        end = min(len(text), position + len(entity) + self.context_window)
        context = text[start:end]
        
        # Detect entity type
        entity_type = self.detect_entity_type(entity, context)
        
        # Resolve based on type
        resolved = None
        confidence = 0.0
        
        if entity_type == 'fund':
            # For funds: เก็บชื่อเดิม (ไม่แปลงเป็นหุ้น!)
            resolved = entity
            confidence = 0.9
            
            # Check if it's a known fund
            entity_upper = entity.upper()
            if entity_upper in self.known_funds:
                resolved = self.known_funds[entity_upper]
                confidence = 1.0
        
        elif entity_type == 'stock':
            # For stocks: ใช้ resolver ปกติ
            if self.stock_ctx:
                from src.core.data_managers import SmartMarketResolver
                resolver = SmartMarketResolver(self.stock_ctx, self.term_mgr)
                ticker = resolver.resolve(entity)
                if ticker:
                    resolved = ticker.replace('.BK', '')
                    confidence = 0.9
                else:
                    resolved = entity  # Keep original if not found
                    confidence = 0.5
            else:
                resolved = entity
                confidence = 0.5
        
        elif entity_type == 'index':
            # For indices: ใช้ finance_terms
            if self.term_mgr and hasattr(self.term_mgr, 'data'):
                # Check in finance_terms
                for category, mapping in self.term_mgr.data.items():
                    for canonical, aliases in mapping.items():
                        if entity.lower() in [a.lower() for a in aliases] or entity.lower() == canonical.lower():
                            resolved = canonical
                            confidence = 0.9
                            break
                    if resolved:
                        break
            
            if not resolved:
                resolved = entity
                confidence = 0.7
        
        else:
            # Unknown: keep original
            resolved = entity
            confidence = 0.3
        
        return {
            'original': entity,
            'resolved': resolved,
            'type': entity_type,
            'confidence': confidence,
            'context': context
        }
    
    def fix_context_blindness(self, text: str) -> Tuple[str, List[Dict]]:
        """
        แก้ไขปัญหา Context Blindness (กองทุน → หุ้น)
        
        Examples:
            "กองทุน THAI ESG" → ไม่แปลงเป็น "กองทุน TISCO"
            "หุ้น TISCO" → แปลงเป็น "TISCO" (ถูกต้อง)
        
        Args:
            text: ข้อความที่ต้องการแก้ไข
            
        Returns:
            Tuple of (fixed_text, fixes_applied)
        """
        fixes = []
        fixed_text = text
        
        # Find all potential entities (uppercase words, 2-6 chars)
        entity_pattern = r'\b([A-Z]{2,6})\b'
        entities = list(re.finditer(entity_pattern, text))
        
        # Process from end to start to preserve positions
        for match in reversed(entities):
            entity = match.group(1)
            position = match.start()
            
            # Resolve with context
            resolution = self.resolve_with_context(entity, text, position)
            
            if resolution and resolution['resolved'] != entity:
                # Apply fix
                fixed_text = fixed_text[:match.start()] + resolution['resolved'] + fixed_text[match.end():]
                fixes.append(resolution)
        
        return fixed_text, fixes


# Convenience function
def resolve_entities_with_context(text: str, stock_ctx=None, term_mgr=None) -> str:
    """
    Resolve entities พร้อมตรวจสอบบริบท (convenience function)
    
    Args:
        text: ข้อความที่ต้องการแก้ไข
        stock_ctx: StockContextManager instance
        term_mgr: FinanceTermManager instance
        
    Returns:
        Fixed text
    """
    resolver = ContextAwareResolver(stock_ctx, term_mgr)
    fixed, _ = resolver.fix_context_blindness(text)
    return fixed

