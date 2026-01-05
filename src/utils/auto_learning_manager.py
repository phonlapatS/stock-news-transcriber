#!/usr/bin/env python3
"""
Auto-Learning Manager
Automatically updates knowledge bases when corrections are made

This creates a sustainable, self-improving system that learns from actual usage
instead of requiring manual pattern maintenance.
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher, Differ
from datetime import datetime


class AutoLearningManager:
    """
    Manages automatic knowledge base updates from LLM corrections
    
   Learns from comparing RAW vs CORRECTED text and auto-updates:
    - knowledge_base.json (stock tickers)
    - finance_terms.json (financial terminology)
    - asr_errors.json (error patterns)
    """
    
    def __init__(self):
        self.kb_path = "knowledge_base.json"
        self.finance_path = "finance_terms.json"
        self.errors_path = "asr_errors.json"
        
        # Thresholds for validation (STRICTER!)
        self.min_confidence = 0.85  # Must be 85%+ confident (was 0.75)
        self.min_frequency = 1      # Learn from first occurrence
        
        # Validation thresholds
        self.similarity_thresholds = {
            "stock_ticker": 0.4,
            "technical_term": 0.5,
            "number": 0.3,
            "general": 0.6
        }
        
        # Preferred terminology (user preference: English technical terms)
        self.preferred_terms = {
            "โล": "low",
            "ไฮ": "high",
            "แนวรับ": "support",
            "แนวต้าน": "resistance",
            "เบรกเอาท์": "breakout",
            "เบรกดาวน์": "breakdown",
        }
    
    def extract_corrections(self, raw_text: str, corrected_text: str, video_id: str = None) -> List[Dict]:
        """
        Extract what was corrected by comparing raw vs corrected text
        
        Args:
            raw_text: Original text
            corrected_text: LLM-corrected text
            video_id: Video identifier for tracking
            
        Returns:
            List of corrections with metadata:
            [{
                "raw": "low",
                "corrected": "low",  # (normalized to English)
                "context": "ทดสอบ low ที่ 105",
                "confidence": 0.95,
                "category": "technical_term",
                "video_id": "xyz"
            }, ...]
        """
        if not raw_text or not corrected_text:
            return []
        
        corrections = []
        
        # แบ่งข้อความออกเป็น tokens (คำๆ) เพื่อนำไปเปรียบเทียบ
        raw_tokens = self._tokenize(raw_text)
        corrected_tokens = self._tokenize(corrected_text)
        
        # หาความแตกต่างระหว่าง raw กับ corrected ด้วย Differ
        differ = Differ()
        diff = list(differ.compare(raw_tokens, corrected_tokens))
        # ผลลัพธ์จะเป็น list ของ string ที่ขึ้นต้นด้วย:
        # '- ' = คำที่ถูกลบ (มีใน raw แต่ไม่มีใน corrected)
        # '+ ' = คำที่ถูกเพิ่ม (ไม่มีใน raw แต่มีใน corrected)
        # '  ' = คำที่เหมือนกัน
        
        i = 0  # ตัวนับสำหรับวนลูป
        while i < len(diff):
            line = diff[i]
            
            # มองหาการแทนที่ (replacement) คือ "ลบ" ตามด้วย "เพิ่ม"
            if line.startswith('- '):
                raw_word = line[2:].strip()  # เอาคำที่ถูกลบ (ข้ามเครื่องหมาย '- ')
                
                # เช็คว่าบรรทัดถัดไปเป็นการเพิ่มหรือไม่ → แปลว่ามีการแทนที่
                if i + 1 < len(diff) and diff[i + 1].startswith('+ '):
                    corrected_word = diff[i + 1][2:].strip()  # เอาคำที่ถูกแก้
                    
                    # ดึงบริบทรอบๆ คำที่ถูกแก้ (เพื่อใช้ในการจำแนกประเภทและคำนวณความมั่นใจ)
                    context = self._get_context(raw_text, raw_word)
                    
                    # จำแนกประเภทของการแก้ไข (stock_ticker, technical_term, number, general)
                    category = self._classify_correction(raw_word, corrected_word, context)
                    
                    # คำนวณความมั่นใจว่าการแก้ไขนี้ถูกต้อง (0.0 - 1.0)
                    confidence = self._calculate_confidence(raw_word, corrected_word, context)
                    
                    # แปลงเป็นรูปแบบที่ต้องการ (เช่น แปลง "โล" → "low" เป็น English)
                    corrected_word = self._normalize_to_preferred(corrected_word)
                    
                    # ===== 5-LAYER VALIDATION =====
                    is_valid, final_confidence, reason = self._validate_correction(
                        raw_word, corrected_word, category, context, raw_text
                    )
                    
                    if is_valid:
                        corrections.append({
                            "raw": raw_word,
                            "corrected": corrected_word,
                            "context": context,
                            "confidence": final_confidence,
                            "category": category,
                            "video_id": video_id or "unknown",
                            "timestamp": datetime.now().isoformat()
                        })
                        # print(f"   ✅ Learned: '{raw_word}' → '{corrected_word}' (conf: {final_confidence:.2f})")
                    else:
                        pass  # Silently reject bad corrections
                        # print(f"   ❌ Rejected: '{raw_word}' → '{corrected_word}' - {reason}")
                    
                    i += 2  # Skip both lines
                    continue
            
            i += 1
        
        return corrections
    
    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens/words"""
        # Simple word-based tokenization
        return text.split()
    
    def _get_context(self, text: str, word: str, window: int = 50) -> str:
        """Get surrounding context for a word"""
        idx = text.find(word)
        if idx == -1:
            return ""
        
        start = max(0, idx - window)
        end = min(len(text), idx + len(word) + window)
        return text[start:end]
    
    def _classify_correction(self, raw: str, corrected: str, context: str) -> str:
        """
        Classify correction type
        
        Categories:
        - stock_ticker: Stock symbols (e.g., "เออีที" → "AOT")
        - technical_term: Technical terms (e.g., "low", "high")
        - number: Numbers
        - general: Other corrections
        """
        # Stock ticker: All caps, 2-5 letters
        if corrected.isupper() and 2 <= len(corrected) <= 5 and corrected.isalpha():
            return "stock_ticker"
        
        # Technical term: Contains financial keywords in context
        financial_keywords = ["บาท", "จุด", "แนว", "ราคา", "หุ้น", "ทดสอบ", "ทะลุ", "หลุด"]
        if any(kw in context for kw in financial_keywords):
            # Check if it's a known technical term
            known_terms = ["low", "high", "support", "resistance", "breakout", "breakdown"]
            if corrected.lower() in known_terms or raw in self.preferred_terms:
                return "technical_term"
        
        # Number
        if corrected.replace('.', '').replace(',', '').isdigit():
            return "number"
        
        return "general"
    
    def _calculate_confidence(self, raw: str, corrected: str, context: str) -> float:
        """
        Calculate confidence that this correction is valid
        
        Factors:
        - Length similarity
        - Context relevance (financial terms)
        - Proper formatting
        """
        score = 0.5  # Base score
        
        # Length similarity
        if raw and corrected:
            len_ratio = min(len(raw), len(corrected)) / max(len(raw), len(corrected))
            score += len_ratio * 0.2
        
        # Financial context
        financial_keywords = ["บาท", "จุด", "แนว", "ราคา", "หุ้น"]
        if any(kw in context for kw in financial_keywords):
            score += 0.2
        
        # Proper formatting (caps, consistent style)
        if corrected.isupper() or corrected.isdigit():
            score += 0.1
        
        return min(1.0, score)
    
    def _normalize_to_preferred(self, term: str) -> str:
        """
        Normalize term to preferred form (English for technical terms)
        
       User preference: Use English technical terms (low, high, etc.)
        """
        # Check if it's a Thai term that should be English
        if term in self.preferred_terms:
            return self.preferred_terms[term]
        
        return term
    
    # ==================== VALIDATION METHODS ====================
    
    def _validate_correction(
        self, 
        raw: str, 
        corrected: str, 
        category: str,
        context: str,
        raw_text: str
    ) -> tuple:
        """
        5-Layer validation pipeline
        
        Returns: (is_valid, confidence, rejection_reason)
        """
        # Layer 1: Length validation
        valid, reason = self._validate_length(raw, corrected, category)
        if not valid:
            return False, 0.0, f"Length: {reason}"
        
        # Layer 2: Similarity check
        valid, similarity = self._validate_similarity(raw, corrected, category)
        if not valid:
            return False, 0.0, f"Similarity: {similarity:.2f} too low"
        
        # Layer 3: Hallucination detection
        valid, reason = self._validate_not_hallucination(raw_text, corrected, category)
        if not valid:
            return False, 0.0, f"Hallucination: {reason}"
        
        # Layer 4: Semantic validation
        valid, reason = self._validate_semantic(raw, corrected, category, context)
        if not valid:
            return False, 0.0, f"Semantic: {reason}"
        
        # Layer 5: Enhanced confidence
        confidence = self._calculate_enhanced_confidence(
            raw, corrected, context, similarity, category
        )
        
        if confidence < self.min_confidence:
            return False, confidence, f"Confidence {confidence:.2f} < {self.min_confidence}"
        
        return True, confidence, ""
    
    def _validate_length(self, raw: str, corrected: str, category: str) -> tuple:
        """Layer 1: Length validation"""
        # Too short
        if len(raw) < 2 or len(corrected) < 1:
            return False, "Too short"
        
        # Category-specific rules
        if category == "stock_ticker":
            # Ticker: 2-5 uppercase letters
            if not (2 <= len(corrected) <= 5 and corrected.isupper() and corrected.isalpha()):
                return False, f"Invalid ticker format: {corrected}"
            
            # RAW length should be similar (±50%)
            len_ratio = len(corrected) / len(raw) if len(raw) > 0 else 0
            if not (0.5 <= len_ratio <= 2.0):
                return False, f"Length mismatch: {len(raw)} → {len(corrected)}"
        
        elif category == "number":
            # Number from long text is suspicious
            if len(raw) > 30:
                return False, f"Number from sentence: '{raw[:20]}...' → '{corrected}'"
        
        elif category == "general":
            # Reject very long corrections (likely sentences)
            if len(corrected) > 30:
                if len(raw) < len(corrected) * 0.8:
                    return False, f"Learning sentence: '{corrected[:20]}...'"
        
        return True, ""
    
    def _validate_similarity(self, raw: str, corrected: str, category: str) -> tuple:
        """Layer 2: Similarity validation"""
        from difflib import SequenceMatcher
        
        # Calculate similarity
        similarity = SequenceMatcher(None, raw.lower(), corrected.lower()).ratio()
        
        # Special case: Thai number → digit
        thai_nums = "หนึ่งสองสามสี่ห้าหกเจ็ดแปดเก้าสิบร้อยพันหมื่นแสนล้าน"
        if corrected.replace('.', '').replace(',', '').isdigit() and any(c in raw for c in thai_nums):
            return True, similarity
        
        # Get threshold for category
        threshold = self.similarity_thresholds.get(category, 0.5)
        
        if similarity < threshold:
            return False, similarity
        
        return True, similarity
    
    def _validate_not_hallucination(self, raw_text: str, corrected: str, category: str) -> tuple:
        """Layer 3: Hallucination detection"""
        corrected_lower = corrected.lower()
        raw_lower = raw_text.lower()
        
        # Exact match OK
        if corrected_lower in raw_lower:
            return True, ""
        
        # Fuzzy match for tickers
        if category == "stock_ticker":
            from difflib import SequenceMatcher
            words = raw_text.split()
            for word in words:
                if SequenceMatcher(None, word.lower(), corrected_lower).ratio() > 0.7:
                    return True, ""
            
            # Validate against known tickers
            valid_tickers = self._get_valid_tickers()
            if corrected.upper() not in valid_tickers:
                return False, f"Unknown ticker: {corrected}"
        
        # Numbers can be spelled out
        if category == "number":
            return True, ""
        
        # Long phrases must have words in source
        corrected_words = corrected.split()
        if len(corrected_words) > 3:
            matches = sum(1 for w in corrected_words if len(w) > 2 and w.lower() in raw_lower)
            if matches / len(corrected_words) < 0.5:
                return False, f"Phrase not in source"
        
        return True, ""
    
    def _validate_semantic(self, raw: str, corrected: str, category: str, context: str) -> tuple:
        """Layer 4: Semantic validation"""
        # Ticker → non-ticker is wrong
        if category == "stock_ticker":
            if not (corrected.isupper() and corrected.isalpha()):
                return False, f"Ticker became non-ticker: '{raw}' → '{corrected}'"
            
            # Must have stock context
            ticker_keywords = ["บาท", "หุ้น", "ราคา", "จุด"]
            if not any(kw in context for kw in ticker_keywords):
                return False, "No stock context"
        
        # Suspicious phrase → number
        if corrected.replace('.', '').replace(',', '').isdigit():
            thai_digits = "หนึ่งสองสามสี่ห้าหกเจ็ดแปดเก้าสิบร้อยพันหมื่นแสนล้าน"
            if not any(c in raw for c in thai_digits) and not raw.replace('.', '').isdigit():
                if len(raw) > 10:
                    return False, f"Phrase→Number: '{raw[:15]}...' → '{corrected}'"
        
        # Long Thai → short English (likely wrong)
        if category == "general" and corrected.isalpha() and corrected.islower():
            known_terms = ["low", "high", "support", "resistance", "breakout", "breakdown"]
            if corrected.lower() not in known_terms and len(raw) > 15:
                return False, f"Long→Short: '{raw[:15]}...' → '{corrected}'"
        
        return True, ""
    
    def _calculate_enhanced_confidence(
        self, 
        raw: str, 
        corrected: str, 
        context: str,
        similarity: float,
        category: str
    ) -> float:
        """Layer 5: Enhanced confidence calculation"""
        score = 0.0  # Start from 0 (was 0.5!)
        
        # Factor 1: Similarity (30%)
        score += similarity * 0.3
        
        # Factor 2: Length compatibility (20%)
        if len(raw) > 0 and len(corrected) > 0:
            len_ratio = min(len(raw), len(corrected)) / max(len(raw), len(corrected))
            score += len_ratio * 0.2
        
        # Factor 3: Category confidence (25%)
        if category == "stock_ticker":
            score += 0.25 if self._is_valid_ticker_format(corrected) else 0.0
        elif category == "number":
            score += 0.25 if corrected.replace('.', '').replace(',', '').isdigit() else 0.0
        elif category == "technical_term":
            score += 0.25 if corrected.lower() in self.preferred_terms.values() else 0.1
        else:
            score += 0.15
        
        # Factor 4: Context relevance (15%)
        context_keywords = {
            "stock_ticker": ["บาท", "หุ้น", "ราคา"],
            "number": ["บาท", "จุด", "ราคา"],
            "technical_term": ["แนว", "ทดสอบ", "ทะลุ", "หลุด"],
            "general": []
        }
        keywords = context_keywords.get(category, [])
        if keywords:
            matches = sum(1 for kw in keywords if kw in context)
            score += (matches / len(keywords)) * 0.15
        else:
            score += 0.05
        
        # Factor 5: Known pattern bonus (10%)
        if self._is_known_pattern(raw, corrected):
            score += 0.10
        
        return min(1.0, score)
    
    def _is_valid_ticker_format(self, text: str) -> bool:
        """Check if text matches ticker format"""
        return text.isupper() and 2 <= len(text) <= 5 and text.isalpha()
    
    def _get_valid_tickers(self) -> set:
        """Load SET ticker list from knowledge_base.json"""
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                kb = json.load(f)
            
            tickers = set()
            for category, stocks in kb.items():
                for ticker in stocks.keys():
                    tickers.add(ticker.replace('.BK', ''))
            return tickers
        except:
            return set()
    
    def _is_known_pattern(self, raw: str, corrected: str) -> bool:
        """Check if pattern exists in asr_errors.json"""
        try:
            with open(self.errors_path, 'r', encoding='utf-8') as f:
                errors = json.load(f)
            return raw in errors and errors[raw].get("correction") == corrected
        except:
            return False
    
    def update_knowledge_bases(self, corrections: List[Dict]) -> Dict[str, int]:
        """
        Update all knowledge bases with validated corrections
        
        Args:
            corrections: List from extract_corrections()
            
        Returns:
            Statistics: {"kb_updates": N, "finance_updates": M, "error_updates": K}
        """
        if not corrections:
            return {"kb_updates": 0, "finance_updates": 0, "error_updates": 0}
        
        # Group by category
        by_category = {}
        for correction in corrections:
            cat = correction['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(correction)
        
        # Update each knowledge base
        stats = {
            "kb_updates": 0,
            "finance_updates": 0,
            "error_updates": 0
        }
        
        if 'stock_ticker' in by_category:
            stats['kb_updates'] = self._update_stock_tickers(by_category['stock_ticker'])
        
        if 'technical_term' in by_category:
            stats['finance_updates'] = self._update_finance_terms(by_category['technical_term'])
        
        # Always update error patterns
        stats['error_updates'] = self._update_error_patterns(corrections)
        
        # === CACHE INVALIDATION ===
        # สำคัญมาก! เมื่อ Knowledge Base ถูก update แล้ว → cache เก่าจะไม่ตรงกับข้อมูลใหม่
        # ต้อง invalidate cache เพื่อให้วิดีโอถัดไปได้ใช้ context ใหม่ที่อัปเดตแล้ว
        total_updates = sum(stats.values())  # นับจำนวน updates ทั้งหมด
        if total_updates > 0:
            try:
                from src.agents.llm_factory import invalidate_cache
                invalidate_cache()  # ล้าง cache ที่เก็บ KB context
                print(f"   🔄 Cache invalidated due to KB updates")
            except Exception as e:
                print(f"   ⚠️ Failed to invalidate cache: {e}")
        
        return stats
    
    def _update_stock_tickers(self, corrections: List[Dict]) -> int:
        """Update knowledge_base.json with new stock ticker variations"""
        if not os.path.exists(self.kb_path):
            print(f"   ⚠️ {self.kb_path} not found, skipping stock ticker update")
            return 0
        
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                kb = json.load(f)
        except Exception as e:
            print(f"   ⚠️ Error loading {self.kb_path}: {e}")
            return 0
        
        updates = 0
        
        # วนลูปแต่ละ correction ที่เป็นประเภท stock_ticker
        for correction in corrections:
            raw = correction['raw'].lower()  # คำที่ ASR ถอดผิด (เช่น "เออีที")
            ticker = correction['corrected'].upper()  # ชื่อหุ้นที่ถูกต้อง (เช่น "AOT")
            
            # หาว่า ticker นี้มีอยู่ใน KB แล้วหรือยัง
            added = False  # flag เช็คว่าเพิ่มแล้วหรือยัง
            for category, stocks in kb.items():  # วนแต่ละหมวดหมู่ (Bank, Energy, etc.)
                for stock_ticker, variations in stocks.items():  # วนแต่ละหุ้นในหมวดนั้น
                    if stock_ticker == f"{ticker}.BK":  # เจอ ticker ที่ต้องการ
                        # เพิ่ม variation ใหม่ถ้ายังไม่มี
                        if raw not in [v.lower() for v in variations]:
                            variations.append(raw)
                            updates += 1
                            print(f"   ✅ KB: Added '{raw}' → {ticker}")
                        added = True
                        break  # ออกจาก loop ชั้นใน
                if added:
                    break  # ออกจาก loop ชั้นนอก
            
            # If not found, add to "Others" category
            if not added:
                if "Others" not in kb:
                    kb["Others"] = {}
                
                kb_key = f"{ticker}.BK"
                if kb_key not in kb["Others"]:
                    kb["Others"][kb_key] = [ticker.lower(), raw]
                    print(f"   ✅ KB: New ticker '{raw}' → {ticker}")
                else:
                    if raw not in kb["Others"][kb_key]:
                        kb["Others"][kb_key].append(raw)
                        print(f"   ✅ KB: Added variation '{raw}' to {ticker}")
                updates += 1
        
        # Save if updated
        if updates > 0:
            try:
                with open(self.kb_path, 'w', encoding='utf-8') as f:
                    json.dump(kb, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"   ⚠️ Error saving {self.kb_path}: {e}")
                return 0
        
        return updates
    
    def _update_finance_terms(self, corrections: List[Dict]) -> int:
        """Update finance_terms.json with new terminology"""
        if not os.path.exists(self.finance_path):
            print(f"   ⚠️ {self.finance_path} not found, skipping finance terms update")
            return 0
        
        try:
            with open(self.finance_path, 'r', encoding='utf-8') as f:
                terms = json.load(f)
        except Exception as e:
            print(f"   ⚠️ Error loading {self.finance_path}: {e}")
            return 0
        
        updates = 0
        category = "Technical Terms"  # Default category
        
        # Ensure category exists
        if category not in terms:
            terms[category] = {}
        
        for correction in corrections:
            raw = correction['raw']
            corrected = correction['corrected']
            
            # Add or update term
            if corrected not in terms[category]:
                terms[category][corrected] = [corrected, raw]
                updates += 1
                print(f"   ✅ Finance: New term '{raw}' → '{corrected}'")
            else:
                # Add variation if not exists
                if raw not in terms[category][corrected]:
                    terms[category][corrected].append(raw)
                    updates += 1
                    print(f"   ✅ Finance: Added '{raw}' to '{corrected}'")
        
        # Save if updated
        if updates > 0:
            try:
                with open(self.finance_path, 'w', encoding='utf-8') as f:
                    json.dump(terms, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"   ⚠️ Error saving {self.finance_path}: {e}")
                return 0
        
        return updates
    
    def _update_error_patterns(self, corrections: List[Dict]) -> int:
        """Update asr_errors.json with error patterns"""
        try:
            from asr_error_logger import ASRErrorLogger
            
            logger = ASRErrorLogger(self.errors_path)
            
            for correction in corrections:
                logger.log_error(
                    raw=correction['raw'],
                    corrected=correction['corrected'],
                    context=correction['context'],
                    video_id=correction.get('video_id', 'unknown')
                )
            
            logger.save()
            return len(corrections)
            
        except Exception as e:
            print(f"   ⚠️ Error updating {self.errors_path}: {e}")
            return 0
