#!/usr/bin/env python3
"""
3-Node Validation Pipeline Nodes
1. Entity Tagger - Extract entities using LLM
2. Market Validator - Validate against yfinance
3. Content Repair - Repair with validated data
"""

import json
import os  # For auto_learning integration
from typing import Dict, List
from datetime import datetime, timedelta


def entity_tagger_node(state: Dict) -> Dict:
    """
    Node 1: Entity Tagger
    ใช้ Gemini Flash Lite สกัดชื่อหุ้นและราคาเป็น JSON
    
    CRITICAL: This must work reliably as it feeds the market validation pipeline
    """
    print(f"\n🏷️ [Node] Entity Tagger...")
    
    try:
        from src.agents.llm_prompts import llm
        from langchain_core.prompts import ChatPromptTemplate
        
        # Enhanced prompt with explicit JSON format
        tagger_prompt = ChatPromptTemplate.from_messages([
            ("system", """คุณคือ Financial Entity Extractor for Thai Stock Market

งาน: สกัดชื่อหุ้นและราคาจากข้อความภาษาไทย

กฎสำคัญ:
1. สกัดเฉพาะ stock tickers และราคาที่ถูกกล่าวถึงชัดเจน
2. ห้ามเดา! ถ้าไม่แน่ใจ ไม่ต้องใส่
3. OUTPUT ต้องเป็น JSON array เท่านั้น

รูปแบบ OUTPUT (ต้องเป็น valid JSON):
[
  {{"stock": "ชื่อหุ้นตามที่พูด", "price": ตัวเลข, "price_type": "support/resistance/price/target"}}
]

ตัวอย่าง:
Input: "ปตท แนวรับที่ 35 บาท แนวต้าน 38 บาท เคแบงก์ราคา 145 บาท"
Output: [{{"stock": "ปตท", "price": 35.0, "price_type": "support"}}, {{"stock": "ปตท", "price": 38.0, "price_type": "resistance"}}, {{"stock": "เคแบงก์", "price": 145.0, "price_type": "price"}}]

ห้าม:
- ห้ามเพิ่มข้อความอธิบาย
- ห้ามใส่ markdown
- OUTPUT เป็น JSON array เท่านั้น

ถ้าไม่พบหุ้นหรือราคา: []"""),
            ("human", "ข้อความ:\n{text}\n\nJSON OUTPUT:")])
        
        chain = tagger_prompt | llm
        
        # Process text
        result = chain.invoke({"text": state["current_text"]})
        content = result.content if hasattr(result, 'content') else str(result)
        
        # Enhanced JSON parsing with multiple strategies
        entities = []
        
        # Strategy 1: Direct JSON parse
        try:
            content_clean = content.strip()
            if content_clean.startswith('['):
                entities = json.loads(content_clean)
                print(f"   ✅ Parsed JSON directly: {len(entities)} entities")
        except:
            pass
        
        # Strategy 2: Extract JSON from markdown or text
        if not entities:
            try:
                # Remove markdown code block if present
                if '```' in content:
                    # Extract content between ```json and ```
                    import re
                    json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
                    if json_match:
                        entities = json.loads(json_match.group(1))
                        print(f"   ✅ Extracted from markdown: {len(entities)} entities")
            except:
                pass
        
        # Strategy 3: Find JSON array anywhere in text
        if not entities:
            try:
                start = content.find('[')
                end = content.rfind(']') + 1
                if start != -1 and end > start:
                    json_str = content[start:end]
                    entities = json.loads(json_str)
                    print(f"   ✅ Found JSON in text: {len(entities)} entities")
            except:
                pass
        
        # Validation: ensure entities are valid
        if entities:
            validated_entities = []
            for entity in entities:
                if isinstance(entity, dict) and 'stock' in entity and 'price' in entity:
                    # Ensure price is float
                    try:
                        entity['price'] = float(entity['price'])
                        entity['price_type'] = entity.get('price_type', 'price')
                        validated_entities.append(entity)
                    except:
                        continue
            
            entities = validated_entities
            
        if not entities:
            print(f"   ⚠️  No entities extracted (LLM response may not contain stocks/prices)")
        else:
            print(f"   ✅ Extracted {len(entities)} entities")
            for entity in entities[:5]:  # Show first 5
                print(f"     • {entity.get('stock', 'N/A')}: {entity.get('price', 'N/A')} บาท ({entity.get('price_type', 'N/A')})")
        
        return {"extracted_entities": entities}
            
    except Exception as e:
        print(f"   ❌ Entity Tagger failed: {e}")
        import traceback
        traceback.print_exc()
        return {"extracted_entities": []}
        
        chain = tagger_prompt | llm
        
        # Process text
        result = chain.invoke({"text": state["current_text"]})
        content = result.content if hasattr(result, 'content') else str(result)
        
        # Parse JSON
        try:
            # ลอง extract JSON จาก response
            json_match = content[content.find('['):content.rfind(']')+1]
            entities = json.loads(json_match)
            print(f"   ✅ Extracted {len(entities)} entities")
            
            return {"extracted_entities": entities}
            
        except json.JSONDecodeError:
            print(f"   ⚠️  Failed to parse JSON, using empty list")
            return {"extracted_entities": []}
            
    except Exception as e:
        print(f"   ❌ Entity Tagger failed: {e}")
        import traceback
        traceback.print_exc()
        return {"extracted_entities": []}


def market_validator_node(state: Dict) -> Dict:
    """
    Node 2: Market Validator
    ตรวจสอบกับตลาดจริงด้วย yfinance + Time Logic
    """
    print(f"\n📊 [Node] Market Validator...")
    
    try:
        import yfinance as yf
        from src.core.data_managers import SmartMarketResolver, StockContextManager, FinanceTermManager
        from src.utils.market_cache import get_market_cache  # NEW: Persistent cache
        
        entities = state.get("extracted_entities", [])
        if not entities:
            print("   ⚠️  No entities to validate")
            return {"validated_data": {}, "validation_logs": []}
        
        # Get data managers (with safety check)
        data_managers = state.get("data_managers", {})
        if not data_managers:
            print("   ⚠️  No data managers found, skipping market validation")
            return {"validated_data": {}, "validation_logs": ["No data managers available"]}
        
        ctx_mgr = data_managers.get("stock_context")
        term_mgr = data_managers.get("finance_term")
        
        if not ctx_mgr or not term_mgr:
            print("   ⚠️  Missing stock_context or finance_term manager")
            return {"validated_data": {}, "validation_logs": ["Missing required managers"]}
        
        resolver = SmartMarketResolver(ctx_mgr, term_mgr)
        
        # Initialize persistent cache (NEW)
        cache = get_market_cache(ttl_hours=24)  # 24-hour TTL
        
        # Extract recording date
        recording_date = state.get("video_metadata", {}).get("upload_date")
        if not recording_date:
            # Try to extract from title
            from src.validation.fact_checker import CleanFactChecker
            checker = CleanFactChecker()
            title = state.get("video_metadata", {}).get("title", "")
            recording_date = checker.extract_date_from_filename(title)
        
        if not recording_date:
            recording_date = datetime.now().strftime("%Y-%m-%d")
            print(f"   ⚠️  No recording date found, using today: {recording_date}")
        
        # Validate each entity
        validated_data = {}
        validation_logs = []
       
        for entity in entities:
            stock_raw = entity.get("stock", "")
            price_raw = entity.get("price")
            price_type = entity.get("price_type", "unknown")
            
            # 1. Resolve ticker name
            resolved = resolver.resolve(stock_raw)  # Returns ticker with .BK already
            if resolved:
                ticker = resolved
            else:
                # Fallback: add .BK if not present
                ticker = stock_raw.upper()
                if not ticker.endswith('.BK'):
                    ticker = f"{ticker}.BK"
            
            # 2. Get market data (with cache)
            try:
                # Check cache first (NEW)
                # Remove .BK for cache key (cleaner)
                ticker_clean = ticker.replace('.BK', '')
                cached_data = cache.get(ticker_clean, recording_date)
                
                if cached_data:
                    print(f"   💾 {ticker}: Using cached data")
                    market_price = cached_data['market_price']
                    price_low = cached_data['price_range']['low']
                    price_high = cached_data['price_range']['high']
                else:
                    # Fetch from yfinance (ticker already has .BK)
                    stock = yf.Ticker(ticker)
                    
                    # Time Logic: Handle weekends/holidays
                    target_date = datetime.strptime(recording_date, "%Y-%m-%d")
                    
                    # Fetch historical data (±3 days to handle weekends)
                    start_date = target_date - timedelta(days=3)
                    end_date = target_date + timedelta(days=1)
                    
                    hist = stock.history(start=start_date.strftime("%Y-%m-%d"),
                                        end=end_date.strftime("%Y-%m-%d"))
                    
                    if hist.empty:
                        validation_logs.append(f"No market data for {ticker_clean} on {recording_date}")
                        continue
                    
                    market_price = hist['Close'].iloc[-1]
                    price_low = hist['Low'].min()
                    price_high = hist['High'].max()
                    
                    # Save to cache (NEW) - use clean ticker for key
                    cache.set(ticker_clean, recording_date, {
                        'market_price': float(market_price),
                        'price_range': {'low': float(price_low), 'high': float(price_high)}
                    })
                    print(f"   ✅ {ticker_clean}: Fetched from yfinance & cached")
                
                # 3. Compare with ASR price
                status = "unknown"
                if price_raw:
                    if price_low <= price_raw <= price_high:
                        status = "match"
                    elif abs(price_raw - market_price) / market_price < 0.15:
                        status = "close"
                    else:
                        status = "mismatch"
                        # Log for auto-learning
                        validation_logs.append(
                            f"Price mismatch: {stock_raw} ({ticker}) - "
                            f"ASR: {price_raw}, Market: {market_price:.2f}"
                        )
                
                validated_data[ticker_clean] = {
                    "original_name": stock_raw,
                    "market_price": float(market_price),
                    "price_range": {"low": float(price_low), "high": float(price_high)},
                    "asr_price": price_raw,
                    "status": status,
                    "price_type": price_type
                }
                
                print(f"   ✅ {ticker_clean}: ASR={price_raw} Market={market_price:.2f} [{status}]")
                
            except Exception as e:
                print(f"   ⚠️  Failed to validate {ticker_clean}: {e}")
                validation_logs.append(f"Validation error for {ticker_clean}: {str(e)}")
        
        return {
            "validated_data": validated_data,
            "validation_logs": validation_logs
        }
        
    except Exception as e:
        print(f"   ❌ Market Validator failed: {e}")
        import traceback
        traceback.print_exc()
        return {"validated_data": {}, "validation_logs": [str(e)]}


def content_repair_node(state: Dict) -> Dict:
    """
    Node 3: Content Repair
    แก้ไขเนื้อหาโดยใช้ validated_data
    """
    print(f"\n🔧 [Node] Content Repair...")
    
    try:
        from src.agents.llm_prompts import llm
        from langchain_core.prompts import ChatPromptTemplate
        
        validated_data = state.get("validated_data", {})
        
        if not validated_data:
            print("   ⚠️  No validated data, skipping repair")
            return {"current_text": state["current_text"]}
        
        # Build validated data string for prompt (escape for f-string)
        validated_str = json.dumps(validated_data, indent=2, ensure_ascii=False)
        # Escape curly braces for f-string
        validated_str_escaped = validated_str.replace('{', '{{').replace('}', '}}')
        
        # Repair prompt
        repair_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""คุณคือ Content Repair Specialist

คุณจะได้รับ 'Validated Data' ที่เชื่อถือได้จากตลาดหลักทรัพย์:

{validated_str_escaped}

กฎการแก้ไข:
1. หาก ASR ถอดความมาผิด (เช่น 'เซตเด็ก', 'ปตท') → แก้เป็นชื่อ ticker ที่ถูกต้องทันที
2. หากราคา status='mismatch' (ต่างกันผิดปกติ) → แก้เป็นราคาจาก market_price
3. หากราคา status='close' (ต่างกันเล็กน้อย) → คงตามเสียงพูดไว้ (อาจเป็น real-time)
4. หากราคา status='match' → ไม่ต้องแก้

ห้ามลบเนื้อหาใดๆ! เพียงแก้ไขชื่อหุ้นและราคาที่ผิดพลาดเท่านั้น"""),
            ("human", "{text}")])
        
        chain = repair_prompt | llm
        
        result = chain.invoke({"text": state["current_text"]})
        repaired_text = result.content if hasattr(result, 'content') else str(result)
        
        print(f"   ✅ Content repaired with {len(validated_data)} validated entities")
        
        return {"current_text": repaired_text}
        
    except Exception as e:
        print(f"   ❌ Content Repair failed: {e}")
        import traceback
        traceback.print_exc()
        return {"current_text": state.get("current_text", "")}
