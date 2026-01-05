# src/core/data_managers.py
import os
import json
from typing import Optional
import re
from config import MASTER_KB_FILE, CACHE_FILE, FINANCE_TERM_FILE

# Import optional libs
try:
    from thefuzz import process
except ImportError:
    process = None
try:
    import yfinance as yf
except ImportError:
    yf = None

# --- KNOWLEDGE BASE ---
class StockContextManager:
    """โหลดรายชื่อหุ้นทั้งหมดเข้าสู่ Memory"""
    def __init__(self, kb_file: str = MASTER_KB_FILE):
        self.kb_file = kb_file
        self.sector_data = {}
        self.all_tickers = set()
        self.search_corpus = {} 
        self.fuzzy_corpus = {}  
        self.load_kb()

    def load_kb(self):
        if not os.path.exists(self.kb_file): return
        try:
            with open(self.kb_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.sector_data = data
                for sector, stocks in data.items():
                    if not isinstance(stocks, dict): continue
                    for ticker, aliases in stocks.items():
                        clean_ticker = ticker.replace(".BK", "")
                        self.all_tickers.add(clean_ticker)
                        # สร้าง Index สำหรับค้นหา
                        self.search_corpus[clean_ticker.lower()] = ticker
                        self.fuzzy_corpus[clean_ticker.lower()] = ticker
                        for alias in aliases:
                            self.search_corpus[alias.lower()] = ticker
                            self.fuzzy_corpus[alias.lower()] = ticker
        except Exception: pass

    def get_sector_prompt_str(self) -> str:
        lines = ["--- รายชื่อหุ้นแยกตามกลุ่มอุตสาหกรรม ---"]
        for sector, stocks in self.sector_data.items():
            if isinstance(stocks, dict):
                tickers = [t.replace(".BK", "") for t in stocks.keys()]
                lines.append(f"## กลุ่ม {sector}:\n    - {', '.join(tickers)}")
        return "\n".join(lines)

class FinanceTermManager:
    """โหลดศัพท์การเงินและคำแก้ผิด"""
    def __init__(self, term_file: str = FINANCE_TERM_FILE):
        self.term_file = term_file
        self.data = {}
        self._load_terms()

    def _load_terms(self):
        if not os.path.exists(self.term_file): return
        try:
            with open(self.term_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception: pass

    def get_prompt_str(self) -> str:
        if not self.data: return ""
        lines = ["--- Domain Finance Terms ---"]
        for group, mapping in self.data.items():
            lines.append(f"### {group}")
            for canonical, aliases in mapping.items():
                lines.append(f"- {canonical}: {', '.join(aliases)}")
        return "\n".join(lines)

# --- RESOLVER (VALIDATOR) ---
class SmartMarketResolver:
    """ตัวตรวจสอบความถูกต้องของชื่อหุ้น (Logic หลักของ Verifier Node)"""
    def __init__(self, context_mgr: StockContextManager, term_mgr: Optional[FinanceTermManager] = None, cache_file: str = CACHE_FILE):
        self.ctx = context_mgr
        self.term_mgr = term_mgr
        self.memory = context_mgr.search_corpus.copy()
        
        # 0. Load Finance Terms into Resolver Memory (Unify SIlos)
        if term_mgr and hasattr(term_mgr, 'data'):
            for category, mapping in term_mgr.data.items():
                for canonical, aliases in mapping.items():
                    self.memory[canonical.lower()] = canonical
                    for alias in aliases:
                        self.memory[alias.lower()] = canonical
        
        self.cache_file = cache_file
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.memory.update(json.load(f))
            except: pass

    def save_cache(self, mention: str, ticker: str):
        new_data = {}
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    new_data = json.load(f)
            except: pass
        new_data[mention.lower()] = ticker
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        self.memory[mention.lower()] = ticker

    def resolve(self, mention: str) -> Optional[str]:
        mention_key = mention.lower().strip()
        
        # 1. Exact Match (เร็วสุด)
        if mention_key in self.memory: return self.memory[mention_key]
        
        # 2. Fuzzy Match (ฉลาดกว่า)
        if process:
            # หาคำที่คล้ายที่สุดใน KB
            best_match, score = process.extractOne(mention_key, self.ctx.fuzzy_corpus.keys())
            ticker = self.ctx.fuzzy_corpus[best_match]
            ticker_clean = ticker.replace(".BK", "")

            # Safety Rule A: หุ้นชื่อสั้น (<=3 ตัว) ต้องเหมือน 100% เท่านั้น
            if len(ticker_clean) <= 3 and score < 100: return None
            
            # Safety Rule B: ภาษาอังกฤษต้องมั่นใจสูง (กัน Bitcoin -> SSET)
            is_english = re.match(r"^[A-Za-z]+$", mention)
            if is_english and score < 95: return None

            # ถ้ามั่นใจเกิน 90% ให้ผ่าน
            if score >= 90:
                self.save_cache(mention_key, ticker)
                return ticker
        return None