#!/usr/bin/env python3
"""
DuckDuckGo Search Tool - ให้ LLM search ข้อมูลหุ้นได้เอง
"""

from duckduckgo_search import DDGS
from typing import List, Dict, Optional
import re


class StockSearchTool:
    """
    เครื่องมือค้นหาข้อมูลหุ้นด้วย DuckDuckGo
    ให้ LLM เรียกใช้เมื่อไม่แน่ใจชื่อหุ้นหรือต้องการข้อมูลเพิ่มเติม
    """
    
    def __init__(self):
        self.cache = {}
    
    def search_ticker(self, query: str, max_results: int = 3) -> Optional[Dict]:
        """
        ค้นหาข้อมูลหุ้นจาก DuckDuckGo
        
        Args:
            query: คำค้นหา (เช่น "RCL หุ้น", "บริษัท อาร์ ซี แอล")
            max_results: จำนวนผลลัพธ์สูงสุด
            
        Returns:
            {
                'ticker': str,
                'company_name': str,
                'description': str,
                'source': str
            }
        """
        # Check cache
        if query in self.cache:
            return self.cache[query]
        
        try:
            with DDGS() as ddgs:
                # ค้นหาเฉพาะหุ้นไทย
                search_query = f"{query} หุ้น SET ตลาดหลักทรัพย์"
                results = list(ddgs.text(search_query, max_results=max_results))
                
                if not results:
                    return None
                
                # Parse ผลลัพธ์แรก
                first_result = results[0]
                
                # พยายามดึงชื่อ ticker จากข้อความ
                ticker = self._extract_ticker_from_text(first_result['body'])
                
                result = {
                    'ticker': ticker,
                    'company_name': self._extract_company_name(first_result['title']),
                    'description': first_result['body'][:200],
                    'source': first_result['href']
                }
                
                # Cache result
                self.cache[query] = result
                return result
                
        except Exception as e:
            print(f"⚠️ DuckDuckGo search failed: {e}")
            return None
    
    def _extract_ticker_from_text(self, text: str) -> Optional[str]:
        """ดึงชื่อ ticker จากข้อความ"""
        # Pattern: ตัวย่อหุ้น 2-6 ตัวอักษร
        tickers = re.findall(r'\b([A-Z]{2,6})\b', text)
        # Return ตัวแรกที่พบ
        return tickers[0] if tickers else None
    
    def _extract_company_name(self, title: str) -> str:
        """ดึงชื่อบริษัทจาก title"""
        # ลบคำที่ไม่เกี่ยวข้อง
        cleaned = re.sub(r'\s*[-|:]\s*.*', '', title)
        return cleaned.strip()
    
    def verify_ticker(self, ticker: str, expected_company: str = None) -> Dict:
        """
        ยืนยันว่า ticker ถูกต้องหรือไม่
        
        Args:
            ticker: ชื่อ ticker ที่ต้องการตรวจสอบ
            expected_company: ชื่อบริษัทที่คาดหวัง (optional)
            
        Returns:
            {
                'is_valid': bool,
                'confidence': float,
                'company_name': str,
                'suggestions': List[str]  # ถ้า ticker ผิด
            }
        """
        result = self.search_ticker(f"{ticker} หุ้น SET")
        
        if not result:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'company_name': None,
                'suggestions': []
            }
        
        # ตรวจสอบว่า ticker ตรงกันหรือไม่
        is_valid = result['ticker'] == ticker if result['ticker'] else False
        
        return {
            'is_valid': is_valid,
            'confidence': 1.0 if is_valid else 0.5,
            'company_name': result['company_name'],
            'suggestions': [result['ticker']] if not is_valid and result['ticker'] else []
        }


# Convenience function for LLM
def search_stock_info(query: str) -> str:
    """
    ฟังก์ชันสำหรับ LLM เรียกใช้ตรงๆ
    
    Args:
        query: คำค้นหา
        
    Returns:
        ข้อมูลหุ้นในรูปแบบ string
    """
    tool = StockSearchTool()
    result = tool.search_ticker(query)
    
    if not result:
        return f"ไม่พบข้อมูลสำหรับ: {query}"
    
    return f"""
ชื่อหุ้น: {result['ticker'] or 'ไม่พบ'}
บริษัท: {result['company_name']}
รายละเอียด: {result['description']}
แหล่งที่มา: {result['source']}
""".strip()
