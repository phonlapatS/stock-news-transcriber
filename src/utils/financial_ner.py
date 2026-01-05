#!/usr/bin/env python3
"""
NER (Named Entity Recognition) for Thai Financial News
สกัดชื่อหุ้น, ราคา, และข้อมูลสำคัญจากข้อความ
"""

import re
from typing import List, Dict, Tuple
from collections import defaultdict


class FinancialNER:
    """
    Named Entity Recognition สำหรับข่าวการเงินไทย
    สกัด: ticker, prices, support/resistance levels, dates
    """
    
    def __init__(self):
        # กำหนด Regular Expression Patterns สำหรับสกัดข้อมูลต่างๆ
        self.ticker_pattern = r'\b([A-Z]{2,6})\b'  # ชื่อหุ้น: ตัวพิมพ์ใหญ่ 2-6 ตัว (เช่น AOT, KBANK)
        self.price_pattern = r'(\d+(?:\.\d+)?)\s*บาท'  # ราคา: ตัวเลข + "บาท" (เช่น 105.50 บาท)
        self.date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'  # วันที่: DD/MM/YYYY หรือ DD-MM-YY
        
        # Dictionary แปลงคำไทยเป็นประเภทราคาที่เป็น standard
        self.price_types = {
            'แนวรับ': 'support',      # แนวรับ (Support Level)
            'แนวต้าน': 'resistance',  # แนวต้าน (Resistance Level)
            'cut loss': 'stop_loss',  # จุดตัดขาดทุน
            'ราคา': 'price',          # ราคาทั่วไป
            'ราคาปิด': 'close',       # ราคาปิด
            'ราคาเปิด': 'open'        # ราคาเปิด
        }
    
    def extract_all(self, text: str) -> Dict:
        """
        สกัดข้อมูลทั้งหมดจากข้อความ
        
        Returns:
            {
                'tickers': List[str],
                'ticker_price_pairs': List[Dict],
                'prices': List[Dict],
                'dates': List[str],
                'entities_by_line': List[Dict]
            }
        """
        lines = text.split('\n')
        
        all_tickers = []
        ticker_price_pairs = []
        all_prices = []
        all_dates = []
        entities_by_line = []
        
        # วนลูปแต่ละบรรทัด และสกัด entities จากแต่ละบรรทัด
        for line_num, line in enumerate(lines, 1):  # เริ่มนับจาก 1 (ไม่ใช่ 0)
            line_entities = self._extract_line_entities(line, line_num)
            entities_by_line.append(line_entities)
            
            # รวบรวม entities จากทุกบรรทัด
            all_tickers.extend(line_entities['tickers'])
            ticker_price_pairs.extend(line_entities['ticker_price_pairs'])
            all_prices.extend(line_entities['prices'])
            all_dates.extend(line_entities['dates'])
        
        return {
            'tickers': list(set(all_tickers)),
            'ticker_price_pairs': ticker_price_pairs,
            'prices': all_prices,
            'dates': all_dates,
            'entities_by_line': entities_by_line
        }
    
    def _extract_line_entities(self, line: str, line_num: int) -> Dict:
        """สกัด entities จากบรรทัดเดียว"""
        entities = {
            'line_num': line_num,
            'text': line,
            'tickers': [],
            'prices': [],
            'ticker_price_pairs': [],
            'dates': []
        }
        
        # 1. Extract tickers
        tickers = re.findall(self.ticker_pattern, line)
        entities['tickers'] = tickers
        
        # 2. สกัดราคาพร้อมบริบทรอบๆ
        price_matches = list(re.finditer(self.price_pattern, line))
        for match in price_matches:
            price = float(match.group(1))  # แปลงเป็นตัวเลข
            # ดึงบริบทรอบๆ (50 ตัวอักษรก่อนหน้า + 20 ตัวอักษรหลัง)
            start = max(0, match.start() - 50)
            context = line[start:match.end() + 20]
            
            # ระบุประเภทของราคาจากบริบท (support, resistance, etc.)
            price_type = self._identify_price_type(context)
            
            entities['prices'].append({
                'value': price,
                'type': price_type,
                'context': context.strip()
            })
        
        # 3. สกัดคู่ของ ticker-price (เช่น "AOT แนวรับที่ 105 บาท")
        # Pattern: TICKER + คำสำคัญ (แนวรับ/แนวต้าน/etc.) + ราคา
        for price_keyword, price_type in self.price_types.items():
            # สร้าง pattern ที่หา: TICKER ตามด้วย (0-100 ตัวอักษร) ตามด้วยคำสำคัญ ตามด้วย (0-30 ตัวอักษร) ตามด้วยราคา
            pattern = rf'({self.ticker_pattern}).{{0,100}}?{price_keyword}.{{0,30}}?{self.price_pattern}'
            matches = re.finditer(pattern, line)
            
            for match in matches:
                # เก็บคู่ของ ticker + price พร้อมข้อมูลเพิ่มเติม
                entities['ticker_price_pairs'].append({
                    'ticker': match.group(1),      # ชื่อหุ้น
                    'price_type': price_type,      # ประเภทราคา (support, resistance, etc.)
                    'price': float(match.group(2)),  # ราคา
                    'context': match.group(0),     # ข้อความที่ match ทั้งหมด
                    'line_num': line_num           # เลขบรรทัด
                })
        
        # 4. Extract dates
        dates = re.findall(self.date_pattern, line)
        entities['dates'] = dates
        
        return entities
    
    def _identify_price_type(self, context: str) -> str:
        """ระบุประเภทของราคาจากบริบท"""
        context_lower = context.lower()
        
        for keyword, price_type in self.price_types.items():
            if keyword.lower() in context_lower:
                return price_type
        
        return 'unknown'
    
    def extract_ticker_info(self, text: str, ticker: str) -> Dict:
        """
        สกัดข้อมูลทั้งหมดของ ticker เฉพาะ
        
        Args:
            text: ข้อความทั้งหมด
            ticker: ชื่อหุ้นที่ต้องการ
            
        Returns:
            {
                'ticker': str,
                'prices': {
                    'support': List[float],
                    'resistance': List[float],
                    'stop_loss': List[float]
                },
                'mentions': List[Dict],  # ทุกครั้งที่กล่าวถึง
                'context': str  # บริบทรวม
            }
        """
        all_entities = self.extract_all(text)
        
        # กรองเฉพาะ ticker ที่ต้องการ
        ticker_prices = defaultdict(list)
        mentions = []
        
        for pair in all_entities['ticker_price_pairs']:
            if pair['ticker'] == ticker:
                ticker_prices[pair['price_type']].append(pair['price'])
                mentions.append(pair)
        
        # สร้างบริบทรวม
        context_parts = [m['context'] for m in mentions]
        context = ' | '.join(context_parts)
        
        return {
            'ticker': ticker,
            'prices': dict(ticker_prices),
            'mentions': mentions,
            'context': context
        }
    
    def validate_ticker_price_consistency(self, text: str) -> List[Dict]:
        """
        ตรวจสอบว่าทุกราคามีชื่อหุ้นกำกับหรือไม่
        
        Returns:
            List of price mentions without tickers
        """
        all_entities = self.extract_all(text)
        
        violations = []
        
        # วนลูปตรวจสอบแต่ละบรรทัด
        for line_data in all_entities['entities_by_line']:
            # ถ้ามีราคาแต่ไม่มี ticker ในบรรทัดเดียวกัน → อาจเป็นปัญหา
            if line_data['prices'] and not line_data['tickers']:
                # บันทึกเป็น violation
                for price in line_data['prices']:
                    violations.append({
                        'line_num': line_data['line_num'],  # บรรทัดที่
                        'price': price['value'],            # ราคา
                        'context': line_data['text'][:100], # ข้อความ (แสดง 100 ตัวอักษรแรก)
                        'issue': 'Price without ticker'     # ประเภทปัญหา
                    })
        
        return violations


# Convenience function
def extract_stock_entities(text: str) -> Dict:
    """
    ฟังก์ชันสำหรับเรียกใช้งานง่ายๆ
    
    Args:
        text: ข้อความที่ต้องการสกัด entities
        
    Returns:
        Dictionary ของ entities ทั้งหมด
    """
    ner = FinancialNER()
    return ner.extract_all(text)
