"""
Market Price Validator using yfinance
Validates stock prices mentioned in transcripts against real market data
Uses 0 Gemini API requests - only HTTP calls to Yahoo Finance

INVESTOR SAFETY: 15% tolerance to catch dangerous misinformation
while allowing for ASR+LLM transcription errors
"""

import yfinance as yf
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


class MarketPriceValidator:
    """Validates stock prices using yfinance (Yahoo Finance)"""
    
    def __init__(self):
        self.cache = {}  # Cache prices to avoid redundant API calls
        # 15% tolerance - STRICT for investment safety
        # Allows ASR+LLM errors (10-20%) but catches dangerous misinformation
        self.tolerance_pct = 0.15
        
    def get_stock_price(self, ticker: str, date: str = None) -> Optional[float]:
        """
        Get stock price for a ticker on a specific date
        
        Args:
            ticker: Stock symbol (e.g., "AOT")
            date: Date in YYYY-MM-DD format (defaults to latest)
        
        Returns:
            Price in Thai Baht, or None if not found
        """
        # Add .BK suffix for Thai stocks
        yahoo_ticker = f"{ticker}.BK"
        cache_key = f"{yahoo_ticker}_{date}"
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(yahoo_ticker)
            
            if date:
                # Get historical data for specific date
                # Add buffer days in case market was closed
                start_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
                end_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                
                hist = stock.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Get closest available date
                    price = float(hist['Close'].iloc[-1])
                else:
                    return None
            else:
                # Get latest price
                hist = stock.history(period="5d")
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                else:
                    return None
            
            # Cache it
            self.cache[cache_key] = price
            return price
            
        except Exception as e:
            print(f"   ⚠️ Price lookup failed for {ticker}: {e}")
            return None
    
    def get_52week_range(self, ticker: str) -> Optional[Dict]:
        """
        Get 52-week high and low for a ticker
        
        Args:
            ticker: Stock symbol (e.g., "MINT")
        
        Returns:
            {"52w_high": float, "52w_low": float, "current": float} or None
        """
        yahoo_ticker = f"{ticker}.BK"
        cache_key = f"{yahoo_ticker}_52w"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(yahoo_ticker)
            info = stock.info
            
            # Get 52-week range from info
            w52_high = info.get('fiftyTwoWeekHigh')
            w52_low = info.get('fiftyTwoWeekLow')
            current = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if w52_high and w52_low:
                result = {
                    "52w_high": float(w52_high),
                    "52w_low": float(w52_low),
                    "current": float(current) if current else None
                }
                self.cache[cache_key] = result
                return result
            else:
                return None
                
        except Exception as e:
            print(f"   ⚠️ 52-week range lookup failed for {ticker}: {e}")
            return None

    def extract_stock_prices(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract stock tickers and prices from text
        
        Returns:
            List of (ticker, price) tuples
        """
        # Pattern: Stock ticker followed by price in baht
        # Matches: "AOT 105 บาท", "DELTA อย่าหลุด 200 บาท"
        pattern = r'\b([A-Z]{2,5})\b[^\d]*?(\d+(?:\.\d{1,2})?)\s*บาท'
        
        matches = re.findall(pattern, text)
        
        # Convert to float and filter valid tickers
        results = []
        for ticker, price_str in matches:
            try:
                price = float(price_str)
                # Filter out unreasonable prices
                if 0.1 <= price <= 10000:  # Thai stock price range
                    results.append((ticker, price))
            except ValueError:
                continue
        
        return results
    
    def validate_with_warnings(self, text: str, max_workers: int = 10) -> Dict:
        """
        Validate prices and flag warnings (NO AUTO-CORRECTION) - ROBUST VERSION
        
        Checks prices against 52-week high/low to detect unreasonable prices.
        Returns warnings but does NOT modify the text.
        
        Features:
        - Parallel API calls for performance
        - Proper error handling
        - Aggressive caching
        
        Args:
            text: Text to validate
            max_workers: Max parallel threads for API calls (default 10)
        
        Returns:
            {
                'warnings': [...],
                'text': original text (unchanged),
                'has_warnings': bool,
                'stats': {'checked': int, 'warnings': int, 'errors': int}
            }
        """
        import concurrent.futures
        
        stocks_mentioned = self.extract_stock_prices(text)
        
        if not stocks_mentioned:
            return {
                'warnings': [],
                'text': text,
                'has_warnings': False,
                'stats': {'checked': 0, 'warnings': 0, 'errors': 0}
            }
        
        # Extract unique tickers
        unique_tickers = list(set(ticker for ticker, _ in stocks_mentioned))
        
        # Parallel API calls for 52-week ranges
        range_data_map = {}
        errors = 0
        
        def fetch_range(ticker):
            try:
                return ticker, self.get_52week_range(ticker)
            except Exception as e:
                print(f"   ⚠️ Error fetching {ticker}: {e}")
                return ticker, None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(fetch_range, unique_tickers)
            for ticker, range_data in results:
                range_data_map[ticker] = range_data
                if range_data is None:
                    errors += 1
        
        # Validate each price
        warnings = []
        
        for ticker, stated_price in stocks_mentioned:
            range_data = range_data_map.get(ticker)
            
            if not range_data:
                continue  # Skip if can't get data
            
            # Validate 52-week data completeness
            if not all([
                range_data.get('52w_high'),
                range_data.get('52w_low')
            ]):
                print(f"   ⚠️ Incomplete 52-week data for {ticker}, skipping")
                errors += 1
                continue
            
            w52_high = range_data['52w_high']
            w52_low = range_data['52w_low']
            current = range_data.get('current')
            
            # 2-TIER THRESHOLD SYSTEM for comprehensive coverage
            # Medium: >5% deviation (catches edge cases like MINT 32 = +8.5%)
            # High: >10% deviation (clear outliers)
            high_threshold_medium = w52_high * 1.05  # 5% above 52w high
            high_threshold_high = w52_high * 1.10    # 10% above 52w high
            low_threshold_medium = w52_low * 0.95    # 5% below 52w low
            low_threshold_high = w52_low * 0.90      # 10% below 52w low
            
            severity = None
            deviation_pct = 0
            message = ""
            
            # Check if price exceeds 52-week high
            if stated_price > high_threshold_medium:
                deviation_pct = ((stated_price - w52_high) / w52_high) * 100
                
                if stated_price > high_threshold_high:
                    severity = 'high'
                    message = f"⚠️ ราคา {stated_price:.2f} บาท เกิน 52-week high ({w52_high:.2f} บาท) +{deviation_pct:.1f}% [HIGH RISK]"
                else:
                    severity = 'medium'
                    message = f"⚡ ราคา {stated_price:.2f} บาท เกิน 52-week high ({w52_high:.2f} บาท) +{deviation_pct:.1f}%"
                
            # Check if price below 52-week low
            elif stated_price < low_threshold_medium:
                deviation_pct = ((w52_low - stated_price) / w52_low) * 100
                
                if stated_price < low_threshold_high:
                    severity = 'high'
                    message = f"⚠️ ราคา {stated_price:.2f} บาท ต่ำกว่า 52-week low ({w52_low:.2f} บาท) -{deviation_pct:.1f}% [HIGH RISK]"
                else:
                    severity = 'medium'
                    message = f"⚡ ราคา {stated_price:.2f} บาท ต่ำกว่า 52-week low ({w52_low:.2f} บาท) -{deviation_pct:.1f}%"
            
            if severity:
                # Extract context (surrounding text) - improved regex
                pattern = rf'({ticker}\s*[^.{{0,50}}]*?{stated_price}[^.{{0,50}}]*?บาท[^.{{0,30}}]*?)'
                match = re.search(pattern, text, re.DOTALL)
                context = match.group(1).strip() if match else f"{ticker} {stated_price} บาท"
                
                # Limit context length
                if len(context) > 150:
                    context = context[:150] + "..."
                
                warnings.append({
                    'ticker': ticker,
                    'stated_price': stated_price,
                    '52w_high': w52_high,
                    '52w_low': w52_low,
                    'current': current,
                    'severity': severity,
                    'deviation_pct': abs(deviation_pct),
                    'message': message,
                    'context': context
                })
        
        return {
            'warnings': warnings,
            'text': text,  # UNCHANGED - no auto-correction
            'has_warnings': len(warnings) > 0,
            'stats': {
                'checked': len(stocks_mentioned),
                'warnings': len(warnings),
                'errors': errors
            }
        }
    
    def validate_prices(self, text: str, date: str = None) -> Dict:
        """
        Validate all stock prices in text against market data
        
        Args:
            text: Transcript text
            date: Date of recording (YYYY-MM-DD)
        
        Returns:
            {
                "total_found": int,
                "validated": int,
                "errors": List[str],
                "warnings": List[str],
                "accuracy": float
            }
        """
        stocks_mentioned = self.extract_stock_prices(text)
        
        if not stocks_mentioned:
            return {
                "total_found": 0,
                "validated": 0,
                "errors": [],
                "warnings": ["No stock prices found in text"],
                "accuracy": 0.0
            }
        
        errors = []
        warnings = []
        validated = 0
        
        for ticker, stated_price in stocks_mentioned:
            actual_price = self.get_stock_price(ticker, date)
            
            if actual_price is None:
                warnings.append(f"{ticker}: Could not fetch market price")
                continue
            
            # Calculate difference
            diff_pct = abs(stated_price - actual_price) / actual_price
            
            if diff_pct <= self.tolerance_pct:
                # Within acceptable range (ASR+LLM error tolerance)
                validated += 1
            else:
                # DANGEROUS: Misinformation that could harm investors
                diff_baht = stated_price - actual_price
                errors.append(
                    f"{ticker}: กล่าวถึง {stated_price:.2f} บาท "
                    f"แต่ราคาตลาด {actual_price:.2f} บาท "
                    f"(ห่าง {abs(diff_baht):.2f} บาท หรือ {diff_pct:.0%}) "
                    f"→ เกินค่าเผื่อ 15% - อาจเป็นอันตราย"
                )
        
        total_checked = len(stocks_mentioned)
        accuracy = validated / total_checked if total_checked > 0 else 0.0
        
        return {
            "total_found": total_checked,
            "validated": validated,
            "errors": errors,
            "warnings": warnings,
            "accuracy": accuracy
        }


def demo():
    """Demo function to test the validator"""
    validator = MarketPriceValidator()
    
    # Test with sample text
    test_text = """
    AOT ปรับตัวขึ้น แนวรับ 105 บาท
    DELTA หากหลุด 200 บาท ถือว่าจบ
    ADVANC แนวรับ 300 บาท
    """
    
    print("Testing Market Price Validator...")
    print(f"Tolerance: {validator.tolerance_pct:.0%} (STRICT for investor safety)")
    print(f"Sample text: {test_text[:100]}...")
    
    result = validator.validate_prices(test_text, date="2022-12-02")
    
    print(f"\nResults:")
    print(f"  Total found: {result['total_found']}")
    print(f"  Validated: {result['validated']}")
    print(f"  Accuracy: {result['accuracy']:.1%}")
    
    if result['errors']:
        print(f"\n  Errors:")
        for error in result['errors']:
            print(f"    ❌ {error}")
    
    if result['warnings']:
        print(f"\n  Warnings:")
        for warning in result['warnings']:
            print(f"    ⚠️ {warning}")


if __name__ == "__main__":
    demo()
