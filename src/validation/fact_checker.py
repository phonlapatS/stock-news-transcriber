#!/usr/bin/env python3
"""
Enhanced Fact Checker - ตรวจสอบความถูกต้องของข้อมูลใน CLEAN output
ป้องกัน LLM Hallucination + ตรวจสอบราคากับตลาดจริง
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
from datetime import datetime, timedelta


class CleanFactChecker:
    """
    Enhanced Fact Checker - ตรวจสอบความถูกต้องของ CLEAN output
    
    Features:
    1. Price-Ticker Consistency - ราคาต้องมีชื่อหุ้นกำกับ
    2. Market Price Validation - ตรวจสอบกับตลาดจริง (yfinance)
    3. Forbidden Keywords - คำที่ไม่ควรมี (Fed, etc.)
    """
    
    # คำต้องห้ามที่ไม่ควรปรากฏในข่าวหุ้นไทยทั่วไป
    FORBIDDEN_KEYWORDS = [
        'Fed', 'Federal Reserve', 'Jerome Powell',
        'ECB', 'European Central Bank',
        'Bank of England', 'BoE'
    ]
    
    def __init__(self):
        # Use persistent cache instead of in-memory
        from src.utils.market_cache import get_market_cache
        self.market_cache = get_market_cache(ttl_hours=24)
    
    def extract_date_from_filename(self, filename: str) -> Optional[str]:
        """
        ดึงวันที่จากชื่อไฟล์ YouTube clip
        
        Supports:
        1. YYYYMMDD format: "_20251224_" → "2025-12-24"
        2. DD/MM/YYYY: "01/12/2568" → "2025-12-01"
        3. DD เดือนย่อ YY: "5 พ.ย. 68" → "2025-11-05"
        4. DD เดือนย่อ YYYY: "03 ธ.ค. 2568" → "2025-12-03"
        5. DD เดือนเต็ม YYYY: "19 พฤศจิกายน 2568" → "2025-11-19"
        
        Returns:
            วันที่ในรูปแบบ YYYY-MM-DD หรือ None
        """
        # Pattern 1: YYYYMMDD (8 ตัวเลขติดกัน) - จาก timestamp ในชื่อไฟล์
        # ตัวอย่าง: "_20251224_" → "2025-12-24"
        match = re.search(r'_(\d{8})_', filename)
        if match:
            date_str = match.group(1)  # เช่น "20251224"
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"  # แปลงเป็น YYYY-MM-DD
        
        # Pattern 2: DD/MM/YYYY or DD/MM/YY (วันที่แบบ slash)
        # ตัวอย่าง: "01/12/2568" → "2025-12-01" หรือ "01/12/68" → "2025-12-01"
        match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', filename)
        if match:
            day = match.group(1).zfill(2)  # เติม 0 ข้างหน้าให้ครบ 2 หลัก
            month = match.group(2).zfill(2)
            year = match.group(3)
            # แปลงปี 2 หลักเป็น 4 หลัก (68 → 2568)
            if len(year) == 2:
                year = f"25{year}"  # สมมติว่าเป็นศตวรรษที่ 25xx
            year_be = int(year)  # ปี พ.ศ.
            year_ce = year_be - 543  # แปลงเป็น ค.ศ.
            return f"{year_ce}-{month}-{day}"
        
        # Thai month mappings (short and full names)
        thai_months = {
            # Short forms (with dots)
            'ม.ค.': '01', 'ก.พ.': '02', 'มี.ค.': '03', 'เม.ย.': '04',
            'พ.ค.': '05', 'มิ.ย.': '06', 'ก.ค.': '07', 'ส.ค.': '08',
            'ก.ย.': '09', 'ต.ค.': '10', 'พ.ย.': '11', 'ธ.ค.': '12',
            # Short forms (without dots)
            'มค': '01', 'กพ': '02', 'มีค': '03', 'เมย': '04',
            'พค': '05', 'มิย': '06', 'กค': '07', 'สค': '08',
            'กย': '09', 'ตค': '10', 'พย': '11', 'ธค': '12',
            # Full names
            'มกราคม': '01', 'กุมภาพันธ์': '02', 'มีนาคม': '03', 'เมษายน': '04',
            'พฤษภาคม': '05', 'มิถุนายน': '06', 'กรกฎาคม': '07', 'สิงหาคม': '08',
            'กันยายน': '09', 'ตุลาคม': '10', 'พฤศจิกายน': '11', 'ธันวาคม': '12'
        }
        
        # Pattern 3 & 4: DD เดือน YYYY or DD เดือน YY
        for thai_month, month_num in thai_months.items():
            # Try with 4-digit year
            pattern = rf'(\d{{1,2}})\s*{re.escape(thai_month)}\s*(\d{{4}})'
            match = re.search(pattern, filename)
            if match:
                day = match.group(1).zfill(2)
                year_be = int(match.group(2))
                year_ce = year_be - 543
                return f"{year_ce}-{month_num}-{day}"
            
            # Try with 2-digit year (e.g., "5 พ.ย. 68")
            pattern = rf'(\d{{1,2}})\s*{re.escape(thai_month)}\s*(\d{{2}})(?!\d)'
            match = re.search(pattern, filename)
            if match:
                day = match.group(1).zfill(2)
                year_short = match.group(2)
                year_be = int(f"25{year_short}")  # 68 → 2568
                year_ce = year_be - 543
                return f"{year_ce}-{month_num}-{day}"
        
        return None
    
    def _extract_ticker_price_pairs(self, text: str) -> List[Dict]:
        """
        ดึงคู่ของ ticker + price โดยตรง
        
        Returns:
            List of {ticker, price, price_type, context, line_num}
        """
        lines = text.split('\n')
        pairs = []
        
        # Pattern: TICKER + ราคา + บริบท (แนวรับ/แนวต้าน/cut loss)
        # เช่น: "RCL แนวรับที่ 27.00 บาท"
        pattern = r'\b([A-Z]{2,6})\b.{0,100}?(แนวรับ|แนวต้าน|cut loss|ราคา).{0,30}?(\d+(?:\.\d+)?)\s*บาท'
        
        for line_num, line in enumerate(lines, 1):
            matches = re.finditer(pattern, line)
            for match in matches:
                pairs.append({
                    'ticker': match.group(1),
                    'price_type': match.group(2),
                    'price': float(match.group(3)),
                    'context': line[max(0, match.start()-20):match.end()+20].strip(),
                    'line_num': line_num
                })
        
        return pairs

    def _extract_price_mentions(self, text: str) -> List[Dict]:
        """
        ดึงการกล่าวถึงราคาทั้งหมด พร้อมบริบท
        
        Returns:
            List of {price, context, line_num, has_ticker}
        """
        lines = text.split('\n')
        price_mentions = []
        
        # Pattern: ตัวเลข + "บาท"
        price_pattern = r'(\d+(?:\.\d+)?)\s*บาท'
        # Pattern: ชื่อหุ้น (2-6 ตัวอักษรพิมพ์ใหญ่)
        ticker_pattern = r'\b([A-Z]{2,6})\b'
        
        for line_num, line in enumerate(lines, 1):
            prices = re.finditer(price_pattern, line)
            
            for match in prices:
                price = match.group(1)
                # ดูบริบทรอบๆ ราคา (50 ตัวอักษรก่อนหน้า)
                start = max(0, match.start() - 50)
                context = line[start:match.end() + 20]
                
                # เช็คว่ามีชื่อหุ้นในบริบทหรือไม่
                has_ticker = bool(re.search(ticker_pattern, context))
                
                price_mentions.append({
                    'price': price,
                    'context': context.strip(),
                    'line_num': line_num,
                    'has_ticker': has_ticker
                })
        
        return price_mentions
    
    def _check_forbidden_keywords(self, text: str) -> List[Dict]:
        """ตรวจสอบคำต้องห้าม"""
        violations = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for keyword in self.FORBIDDEN_KEYWORDS:
                if keyword.lower() in line.lower():
                    violations.append({
                        'keyword': keyword,
                        'line_num': line_num,
                        'context': line[:100].strip()
                    })
        
        return violations
    
    def _extract_tickers(self, text: str) -> List[str]:
        """ดึงชื่อหุ้นทั้งหมด"""
        return list(set(re.findall(r'\b[A-Z]{2,6}\b', text)))
    
    def validate_against_market(self, ticker: str, price: float, 
                                recording_date: str, tolerance: float = 0.20) -> Dict:
        """
        ตรวจสอบราคากับตลาดจริง (yfinance)
        
        Args:
            ticker: ชื่อหุ้น (เช่น "RCL")
            price: ราคาที่ต้องการตรวจสอบ
            recording_date: วันที่อัดคลิป (YYYY-MM-DD)
            tolerance: ช่วงความคลาดเคลื่อนที่ยอมรับได้ (default 20%)
            
        Returns:
            {
                "plausible": True/False/None,
                "market_price": float,
                "deviation": float,
                "confidence": "high"/"medium"/"low",
                "error": str (if any)
            }
        """
        try:
            import yfinance as yf
            
            # ใช้ cache ถ้ามี
            cache_key = f"{ticker}_{recording_date}"
            if cache_key in self.market_cache:
                cached = self.market_cache[cache_key]
            else:
                # ดึงข้อมูลจาก yfinance (ใช้ .BK suffix สำหรับหุ้นไทยในตลาดกรุงเทพฯ)
                stock = yf.Ticker(f"{ticker}.BK")
                
                # แปลงวันที่และกำหนดช่วงเวลาที่จะดึงข้อมูล
                target_date = datetime.strptime(recording_date, "%Y-%m-%d")
                start_date = target_date - timedelta(days=3)  # ย้อนหลัง 3 วัน
                end_date = target_date + timedelta(days=1)    # ถึงวันถัดไป
                # เผื่อกรณีวันที่อัดคลิปเป็นวันหยุด จะได้ข้อมูลจากวันก่อนหน้า
                
                hist = stock.history(start=start_date.strftime("%Y-%m-%d"),
                                    end=end_date.strftime("%Y-%m-%d"))
                
                if hist.empty:
                    return {
                        "plausible": None,
                        "error": f"No market data for {ticker} on {recording_date}"
                    }
                
                cached = {
                    'low': hist['Low'].min(),
                    'high': hist['High'].max(),
                    'close': hist['Close'].iloc[-1] if len(hist) > 0 else None
                }
                self.market_cache[cache_key] = cached
            
            # ตรวจสอบว่าราคาอยู่ในช่วง intraday หรือไม่ (ระหว่าง Low-High ของวันนั้น)
            if cached['low'] <= price <= cached['high']:
                # ราคาอยู่ในช่วงของวัน → น่าเชื่อถือสูง (confidence: high)
                return {
                    "plausible": True,
                    "confidence": "high",
                    "market_price": cached['close'],
                    "deviation": abs(price - cached['close']) / cached['close'] if cached['close'] else 0
                }
            
            # ถ้าไม่อยู่ในช่วง intraday → ตรวจสอบว่าใกล้เคียงกับราคาปิดหรือไม่ (ตาม tolerance)
            if cached['close']:
                deviation = abs(price - cached['close']) / cached['close']  # คำนวณ % ที่ต่างกัน
                if deviation < tolerance:  # ต่างกันน้อยกว่า 20% (default)
                    # ยอมรับได้ แต่ confidence ต่ำกว่า
                    return {
                        "plausible": True,
                        "confidence": "medium",
                        "market_price": cached['close'],
                        "deviation": deviation
                    }
                else:
                    # ต่างกันมากเกินไป → ไม่น่าเชื่อถือ
                    return {
                        "plausible": False,
                        "confidence": "low",
                        "market_price": cached['close'],
                        "deviation": deviation
                    }
            
            return {"plausible": None, "error": "No closing price available"}
            
        except ImportError:
            return {"plausible": None, "error": "yfinance not installed"}
        except Exception as e:
            return {"plausible": None, "error": str(e)}
    
    def validate(self, clean_text: str, filename: str = None) -> Dict:
        """
        ตรวจสอบ CLEAN output
        
        Args:
            clean_text: ข้อความที่ผ่าน LLM correction มาแล้ว
            
        Returns:
            {
                'is_valid': bool,
                'warnings': List[str],
                'errors': List[str],
                'price_issues': List[Dict],
                'forbidden_keywords': List[Dict]
            }
        """
        warnings = []
        errors = []
        market_validation_issues = []
        
        # Extract recording date from filename
        recording_date = None
        if filename:
            recording_date = self.extract_date_from_filename(filename)
            if not recording_date:
                warnings.append("⚠️ ไม่พบวันที่ในชื่อไฟล์ - ข้ามการ validate กับตลาด")
        
        # 1. ตรวจสอบ Ticker-Price Consistency + Market Validation
        ticker_price_pairs = self._extract_ticker_price_pairs(clean_text)
        
        for pair in ticker_price_pairs:
            # ตรวจสอบกับตลาด (ถ้ามีวันที่)
            if recording_date:
                market_result = self.validate_against_market(
                    pair['ticker'], 
                    pair['price'],
                    recording_date
                )
                
                if market_result.get('plausible') == False:
                    issue_msg = (
                        f"🚨 {pair['ticker']} ราคา {pair['price']:.2f} บาท "
                        f"ไม่สอดคล้องกับตลาด (ตลาด: {market_result['market_price']:.2f} บาท, "
                        f"ต่าง {market_result['deviation']*100:.1f}%) "
                        f"บรรทัด {pair['line_num']}: {pair['context'][:60]}..."
                    )
                    errors.append(issue_msg)
                    market_validation_issues.append({
                        **pair,
                        **market_result
                    })
                elif market_result.get('confidence') == 'medium':
                    warnings.append(
                        f"⚠️ {pair['ticker']} ราคา {pair['price']:.2f} บาท "
                        f"ต่างจากตลาดเล็กน้อย ({market_result['deviation']*100:.1f}%)"
                    )
        
        # 2. ตรวจสอบคำต้องห้าม
        forbidden = self._check_forbidden_keywords(clean_text)
        if forbidden:
            for item in forbidden:
                errors.append(
                    f"🚨 พบคำต้องห้าม '{item['keyword']}' ที่บรรทัด {item['line_num']}: "
                    f"{item['context'][:80]}..."
                )
        
        # 3. ตรวจสอบ Price-Ticker Consistency (ราคาที่ไม่มีชื่อหุ้น)
        price_mentions = self._extract_price_mentions(clean_text)
        price_issues = []
        
        for mention in price_mentions:
            if not mention['has_ticker']:
                # ราคาไม่มีชื่อหุ้นกำกับ
                warnings.append(
                    f"⚠️ ราคา {mention['price']} บาท ไม่มีชื่อหุ้นกำกับ "
                    f"(บรรทัด {mention['line_num']}): {mention['context'][:60]}..."
                )
                price_issues.append(mention)
        
        # 4. สถิติพื้นฐาน
        tickers = self._extract_tickers(clean_text)
        
        # ตัดสินผล
        is_valid = len(errors) == 0  # มี errors = ไม่ผ่าน
        
        return {
            'is_valid': is_valid,
            'warnings': warnings,
            'errors': errors,
            'price_issues': price_issues,
            'market_validation_issues': market_validation_issues,  # NEW
            'forbidden_keywords': forbidden,
            'statistics': {
                'total_price_mentions': len(price_mentions),
                'prices_without_ticker': len(price_issues),
                'ticker_price_pairs': len(ticker_price_pairs),  # NEW
                'market_validated': len([p for p in ticker_price_pairs if recording_date]),  # NEW
                'total_tickers': len(tickers),
                'tickers': tickers,
                'recording_date': recording_date  # NEW
            }
        }
    
    def generate_report(self, validation_result: Dict) -> str:
        """สร้างรายงานแบบอ่านง่าย"""
        report = []
        report.append("=" * 80)
        report.append("🔍 FACT CHECKER REPORT")
        report.append("=" * 80)
        
        # Status
        status = "✅ PASSED" if validation_result['is_valid'] else "❌ FAILED"
        report.append(f"\nStatus: {status}")
        
        # Errors
        if validation_result['errors']:
            report.append(f"\n🚨 ERRORS ({len(validation_result['errors'])}):")
            for error in validation_result['errors']:
                report.append(f"   {error}")
        
        # Warnings
        if validation_result['warnings']:
            report.append(f"\n⚠️  WARNINGS ({len(validation_result['warnings'])}):")
            for warning in validation_result['warnings'][:5]:  # แสดงแค่ 5 รายการแรก
                report.append(f"   {warning}")
            if len(validation_result['warnings']) > 5:
                report.append(f"   ... และอีก {len(validation_result['warnings']) - 5} รายการ")
        
        # Statistics
        stats = validation_result['statistics']
        report.append(f"\n📊 STATISTICS:")
        report.append(f"   Total price mentions: {stats['total_price_mentions']}")
        report.append(f"   Prices without ticker: {stats['prices_without_ticker']}")
        report.append(f"   Total tickers found: {stats['total_tickers']}")
        if stats['tickers']:
            report.append(f"   Tickers: {', '.join(sorted(stats['tickers'][:10]))}")
        
        report.append("=" * 80)
        
        return '\n'.join(report)


# Convenience function
def check_clean_transcript(clean_text: str, verbose: bool = True) -> Dict:
    """
    ตรวจสอบ CLEAN transcript อย่างรวดเร็ว
    
    Args:
        clean_text: CLEAN output text
        verbose: พิมพ์รายงานหรือไม่
        
    Returns:
        Validation result dict
    """
    checker = CleanFactChecker()
    result = checker.validate(clean_text)
    
    if verbose:
        print(checker.generate_report(result))
    
    return result
