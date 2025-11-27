import os
import io
import time
import re
import httpx
import json
import yt_dlp
import warnings
import concurrent.futures
import difflib
import random
import logging

from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydub import AudioSegment, effects

# -------------------------------------------------
# Configuration & Setup
# -------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger('httpx').setLevel(logging.WARNING)

# --- API KEYS ---
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_API_KEY = "sk-vCE2QnUydpGnzic35kI3IcoTsAeWzb2X3jYCCAXDPmfT2JnN"

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GOOGLE_API_KEY = "AIzaSyCkjWUucxzLaRPnuklKxKgYP0fUyQhTwHA"
LLM_MODEL_NAME = "gemini-2.0-flash"

# --- TARGET VIDEO ---
YOUTUBE_URL = "https://www.youtube.com/watch?v=opEIqiPzx64"

# --- PATHS ---
DOWNLOAD_DIR = "downloads"
TRANSCRIPT_OUTPUT_DIR = "transcripts_output"
RAW_TRANSCRIPT_FILE = "raw_transcript_full.txt"
CLEAN_TRANSCRIPT_FILE = "final_transcript_clean.txt"
MARKDOWN_FILE = "final_summary_markdown.md"
MASTER_KB_FILE = "knowledge_base.json"
CACHE_FILE = "ticker_cache.json"

# --- SETTINGS ---
CHUNK_DURATION_SEC = 45
OVERLAP_DURATION_SEC = 10
MAX_WORKERS = 5

# ตรวจสอบ Optional Libs
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None
try:
    import yfinance as yf
except ImportError:
    yf = None

# -------------------------------------------------
# PART 1: AUDIO PROCESSOR
# -------------------------------------------------
class AudioProcessor:
    """จัดการคุณภาพเสียงก่อนส่งเข้า ASR"""
    @staticmethod
    def preprocess_audio(file_path):
        print(f"🔊 Processing Audio: {file_path}")
        print("   - Loading...")
        try:
            audio = AudioSegment.from_file(file_path)
            
            # 1. Convert to Mono (ASR models prefer mono)
            if audio.channels > 1:
                audio = audio.set_channels(1)
                print("   - Converted to Mono")
            
            # 2. Resample to 16000Hz (Native rate for Whisper/Typhoon)
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
                print("   - Resampled to 16kHz")
            
            # 3. Normalize (Adjust volume)
            audio = effects.normalize(audio)
            print("   - Normalized Volume")
            
            return audio
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            return None

# -------------------------------------------------
# PART 2: CONTEXT & DYNAMIC PROMPT
# -------------------------------------------------
class StockContextManager:
    def __init__(self, kb_file=MASTER_KB_FILE):
        self.kb_file = kb_file
        self.sector_data = {}
        self.flat_memory = {} # Map alias -> Ticker
        self.all_tickers = set()
        self.load_kb()

    def load_kb(self):
        if os.path.exists(self.kb_file):
            try:
                with open(self.kb_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.sector_data = data
                    for sector, stocks in data.items():
                        if isinstance(stocks, dict):
                            for ticker, aliases in stocks.items():
                                clean_ticker = ticker.replace('.BK', '')
                                self.all_tickers.add(clean_ticker)
                                self.flat_memory[clean_ticker.lower()] = ticker
                                for alias in aliases:
                                    self.flat_memory[alias.lower()] = ticker
                print(f"📚 Knowledge Base Loaded: {len(self.all_tickers)} Tickers")
            except Exception as e:
                print(f"⚠️ Error loading KB: {e}")

    def get_sector_prompt_str(self):
        lines = []
        for sector, stocks in self.sector_data.items():
            if isinstance(stocks, dict):
                tickers = [t.replace('.BK', '') for t in stocks.keys()]
                lines.append(f"- {sector}: {', '.join(tickers)}")
        return "\n".join(lines)

class DynamicPromptBuilder:
    """สร้าง ASR Prompt แบบ Dynamic จาก Metadata"""
    def __init__(self, context_mgr: StockContextManager):
        self.ctx = context_mgr
        
    def extract_potential_tickers(self, text: str) -> List[str]:
        if not text: return []
        # หาคำภาษาอังกฤษ 2-8 ตัวอักษร
        candidates = re.findall(r"\b[A-Z]{2,8}\b", text.upper())
        # กรองเฉพาะที่มีใน KB หรือดูเหมือนชื่อหุ้น
        found = []
        blacklist = {"LIVE", "THE", "AND", "FOR", "DAY", "SET", "MAI", "BREAK", "NEWS", "TODAY"}
        for c in candidates:
            if c in self.ctx.all_tickers and c not in blacklist:
                found.append(c)
            elif c not in blacklist and len(c) >= 3:
                # ถ้าไม่อยู่ใน KB แต่อยู่ใน Title ก็เก็บไว้ก่อน
                found.append(c)
        return list(set(found))

    def build_prompt(self, metadata: dict) -> str:
        title = metadata.get('title', '')
        desc = metadata.get('description', '') or ''
        tags = metadata.get('tags', []) or []
        
        # 1. High Priority: หุ้นที่อยู่ใน Title/Desc/Tags
        combined_text = f"{title} {desc} {' '.join(tags)}"
        priority_tickers = self.extract_potential_tickers(combined_text)
        
        # 2. Medium Priority: ศัพท์เทคนิค
        base_vocab = [
            "แนวรับ", "แนวต้าน", "Stop Loss", "Profit Run", "ดัชนี", "SET Index", "SET50", 
            "งบการเงิน", "กำไรสุทธิ", "ไตรมาส", "Upside", "Downside", "Volume", "RSI", "MACD"
        ]
        
        # 3. Construct Prompt
        prompt_parts = []
        if priority_tickers:
            prompt_parts.append(f"ชื่อหุ้นในคลิป: {', '.join(priority_tickers)}")
        
        prompt_parts.append(f"คำศัพท์: {', '.join(base_vocab)}")
        
        full_prompt = " | ".join(prompt_parts)
        return full_prompt[:800] # ตัดเพื่อความปลอดภัย

# -------------------------------------------------
# PART 3: SMART RESOLVER (ENTITY VERIFICATION)
# -------------------------------------------------
class MarketTools:
    @staticmethod
    def search_ticker(query: str) -> Optional[str]:
        if DDGS is None: return None
        try:
            search_q = f"หุ้น {query} stock symbol ticker settrade"
            with DDGS() as ddgs:
                results = list(ddgs.text(search_q, max_results=2))
                if not results: return None
                blob = " ".join([r.get("title", "") + " " + r.get("body", "") for r in results]).upper()
                candidates = re.findall(r"\b([A-Z]{2,8})\.BK\b", blob)
                if candidates: return candidates[0] + ".BK"
                candidates = re.findall(r"\b([A-Z]{2,8})\b", blob)
                blacklist = {"SET", "MAI", "BKK", "THAI", "PRICE", "NEWS", "STOCK", "TRADE", "DATA", "INFO", "REAL", "TIME"}
                for ticker in candidates:
                    if ticker not in blacklist and len(ticker) >= 2:
                        return ticker
        except: return None
        return None

    @staticmethod
    def verify_ticker(ticker: str) -> bool:
        if yf is None or not ticker: return False
        if "SET.BK" in ticker or "^" in ticker: return False
        try:
            hist = yf.Ticker(ticker).history(period="1d")
            return not hist.empty
        except: return False

class SmartMarketResolver:
    def __init__(self, context_mgr, cache_file=CACHE_FILE):
        self.memory = context_mgr.flat_memory.copy()
        self.cache_file = cache_file
        self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.memory.update(json.load(f))
            except: pass

    def save_cache(self, mention, ticker):
        new_data = {}
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    new_data = json.load(f)
            except: pass
        new_data[mention.lower()] = ticker
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        self.memory[mention.lower()] = ticker

    def resolve(self, mention):
        mention_key = mention.lower().strip()
        # 1. Check Memory/Cache
        if mention_key in self.memory: return self.memory[mention_key]
        
        # 2. Check Strict Pattern (e.g., if mention is already valid ticker like "KBANK")
        if mention.upper() in ctx_mgr.all_tickers:
            return mention.upper() + ".BK"

        # 3. Search
        print(f"    🌐 [Searching] '{mention}' ...")
        found = MarketTools.search_ticker(mention)
        time.sleep(1) # Rate limit protection
        
        if found:
            clean = found.upper().strip()
            # Try appending .BK first
            if ".BK" not in clean:
                thai = clean + ".BK"
                if MarketTools.verify_ticker(thai):
                    self.save_cache(mention_key, thai)
                    return thai
            
            # Check raw found
            if MarketTools.verify_ticker(clean):
                self.save_cache(mention_key, clean)
                return clean
                
        return None

# -------------------------------------------------
# PART 4: LANGCHAIN AGENTS (IMPROVED PROMPTS)
# -------------------------------------------------

# Init LLM
llm = ChatOpenAI(
    base_url=GOOGLE_BASE_URL, 
    api_key=GOOGLE_API_KEY, 
    model=LLM_MODEL_NAME, 
    temperature=0.1, 
    max_tokens=8192
)

# --- Prompts ---

# 1. Cleaning
clean_prompt = ChatPromptTemplate.from_messages([
    ("system", "คุณคือ Transcript Editor หน้าที่: ตัดคำฟุ่มเฟือย (เอ่อ, นะครับ, อา, แบบว่า) ออก และจัดย่อหน้าให้อ่านง่าย แต่ห้ามเปลี่ยนเนื้อหา หรือตัดข้อมูลตัวเลข/ชื่อหุ้นทิ้ง"),
    ("user", "Transcript:\n\"\"\"{raw_text}\"\"\"")
])
cleaning_chain = clean_prompt | llm | StrOutputParser()

# 2. Correction (Senior Investment Analyst)
correction_system_prompt = (
    "คุณคือ **'Senior Investment Analyst Editor'** ที่มีหน้าที่เรียบเรียงบทบรรยายการลงทุนให้สมบูรณ์ **ถูกต้องตามข้อเท็จจริงทางการเงิน** และอ่านลื่นไหลที่สุด\n"
    "ภารกิจ: แก้ไขข้อมูลให้ถูกต้องตามหลักการลงทุน และจัดรูปแบบเป็น **'ความเรียง (Paragraphs)'** ให้อ่านลื่นไหล\n\n"
    
    "--- กฎการทำงาน (Rules) ---\n"
    "1. **ความสมเหตุสมผลทางการเงิน (Investment Logic & Data Integrity) [สำคัญที่สุด!]**: \n"
    "   - **Data Audit**: ตรวจสอบความถูกต้องของตัวเลขสำคัญ (ราคา, แนวรับ/ต้าน) ให้ตรงกับบริบทโดยไม่แต่งเติม หากตัวเลขฟังไม่ชัดเจน ให้ใช้บริบทแวดล้อมเพื่อยืนยัน\n"
    "   - **Logic Consistency**: แก้ไขความขัดแย้งทางตรรกะ (เช่น หุ้น 'ลงแรง' แต่บอกว่า 'เข้าซื้อเพิ่ม') ให้สอดคล้องกับสถานการณ์\n"
    "   - **Contextual Ticker Check**: ตรวจสอบว่าหุ้นที่กล่าวถึงสอดคล้องกับธีมข่าวในขณะนั้นหรือไม่ (เช่น ข่าวโรงไฟฟ้า ก็ควรเป็น GULF, GPSC ไม่ใช่หุ้นขนม)\n"
    "2. **ความครบถ้วน (Completeness)**: **ห้ามตัดทอนเนื้อหาที่เกี่ยวกับการลงทุน/หุ้นออกเด็ดขาด!** ข้อมูลต้องอยู่ครบตั้งแต่ต้นจนจบ\n"
    "3. **แก้คำผิด (Ticker Correction)**: แก้คำที่ฟังผิดให้เป็นชื่อหุ้น (Ticker) ที่ถูกต้อง (เช่น 'กราฟ' -> 'GULF', 'เอ็นแคป' -> 'NCAP', 'บีไอเอ' -> 'BRI') โดยใช้บริบทประกอบ\n"
    "4. **รูปแบบ (Formatting)**: \n"
    "   - เขียนเป็น **'ย่อหน้า (Paragraph)'** ต่อเนื่องกัน (Narrative Style)\n"
    "   - **ห้ามใช้ Bullet Points หรือตัวเลขนำหน้า** ในขั้นตอนนี้\n\n"
    "**Output**: ส่งคืน Transcript ฉบับสมบูรณ์แบบความเรียง"
)
correction_prompt = ChatPromptTemplate.from_messages([
    ("system", correction_system_prompt),
    ("user", "Transcript to Correct:\n\"\"\"{clean_text}\"\"\"")
])
correction_chain = correction_prompt | llm | StrOutputParser()

# 3. NER (Structured Output)
class StockEntity(BaseModel):
    text_found: str = Field(...)
    
class EntityList(BaseModel):
    entities: List[StockEntity]

ner_prompt = ChatPromptTemplate.from_messages([
    ("system", "ดึงรายชื่อหุ้น (Ticker) หรือชื่อบริษัทที่ถูกกล่าวถึงในข้อความ ออกมาเป็นรายการ JSON"),
    ("user", "Text:\n\"\"\"{text}\"\"\"")
])
ner_chain = ner_prompt | llm.with_structured_output(EntityList)

# 4. Summary (Infographic Lead) - [UPDATED: NO BOLD]
summary_system_prompt = (
    "คุณคือ **'Infographic Content Lead'** ที่มีความเชี่ยวชาญด้านการย่อยข้อมูลหุ้น\n"
    "โจทย์: สรุปข้อมูลจาก Transcript เพื่อทำ Infographic ที่ **'เจาะลึก (Insightful)'** และ **'มีเนื้อหาครบถ้วน'** ไม่ใช่แค่รายการย่อ\n\n"
    
    "--- กฎเหล็กการจัดรูปแบบ (Strict Rules) ---\n"
    "1. **Clean Text (No Bold)**: ห้ามใช้เครื่องหมาย ** (ตัวหนา) ในเนื้อหาโดยเด็ดขาด เพื่อความสะอาดและอ่านง่าย (ใช้ได้เฉพาะ Header #)\n"
    "2. **ต้องลงรายละเอียด (Must be Detailed)**: ในแต่ละ Bullet Point ต้องระบุ **'สาเหตุ (Why)'**, **'ตัวเลขสำคัญ (Key Numbers)'**, หรือ **'ปัจจัยบวก/ลบ (Catalysts)'** เสมอ\n"
    "   - *แย่:* กลุ่มไฟแนนซ์ปรับตัวขึ้น\n"
    "   - *ดี:* กลุ่มไฟแนนซ์ปรับตัวขึ้น รับข่าวงบ THANI กำไรดีกว่าคาดที่ 300 ลบ. จากการตั้งสำรองลดลง\n"
    "3. **ความถูกต้อง**: ชื่อหุ้นและตัวเลข (ราคา, แนวรับ/ต้าน) ต้องเป๊ะ 100% ตาม Transcript\n"
    "4. **ชื่อหุ้น**: ตัวพิมพ์ใหญ่ (UPPERCASE) ห้ามตัวหนา\n"
    "5. **โครงสร้าง**: ใช้ Header (#, ##) และ Bullet Points (-) เท่านั้น\n"
    "6. **ความครบถ้วน**: ห้ามตัดข้อมูลสำคัญทิ้ง โดยเฉพาะส่วน Technical และ Strategy\n\n"
    
    "--- โครงสร้าง (Mandatory Template) ---\n"
    "# สรุปภาวะตลาด\n"
    "## ภาพรวมตลาดและปัจจัยขับเคลื่อน\n"
    "   - (สรุปดัชนี, มูลค่าซื้อขาย, และปัจจัยข่าวต่างประเทศที่กระทบ พร้อมรายละเอียด)\n"
    "## หุ้นที่ปรับตัวขึ้นและประเด็นสำคัญ\n"
    "   - (ระบุกลุ่มอุตสาหกรรม และรายชื่อหุ้น พร้อมสาเหตุที่ขึ้นอย่างละเอียด)\n"
    "## หุ้นที่ปรับตัวลงและประเด็นที่ต้องระวัง\n"
    "   - (ระบุหุ้นที่ลง พร้อมสาเหตุ เช่น งบแย่, ข่าวลบ, หรือ 52-week low)\n"
    "## หุ้นแนะนำเชิงกลยุทธ์ (Strategy Picks)\n"
    "   - (สรุปธีมการลงทุนหลัก และรายชื่อหุ้นแนะนำพร้อมเหตุผล)\n"
    "## หุ้นแนะนำทางเทคนิค (Technical Picks)\n"
    "   - (ชื่อหุ้น พร้อม Pattern กราฟ และเลขแนวรับ/แนวต้าน/Stop Loss ให้ครบถ้วน)\n"
    "## ประเด็นข่าวสารอื่นๆ\n"
    "   - (ข่าวรายตัวอื่น ๆ และหุ้นที่นักวิเคราะห์คาดว่างบจะดี)\n\n"
    
    "[MAPPING Reference] (ใช้ชื่อหุ้นที่ถูกต้องตามนี้)\n{mapping_str}"
)
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", summary_system_prompt),
    ("user", "Transcript:\n\"\"\"{corrected_text}\"\"\"")
])
summary_chain = summary_prompt | llm | StrOutputParser()

# -------------------------------------------------
# PART 5: MAIN LOGIC
# -------------------------------------------------

# Global Objects
ctx_mgr = StockContextManager()
prompt_builder = DynamicPromptBuilder(ctx_mgr)
resolver = SmartMarketResolver(ctx_mgr)

# Typhoon Client
try:
    http_client = httpx.Client(timeout=120.0)
    from openai import OpenAI
    asr_client = OpenAI(base_url=TYPHOON_BASE_URL, api_key=TYPHOON_API_KEY, http_client=http_client)
except Exception as e:
    print(f"❌ Error init Typhoon Client: {e}"); exit()

def transcribe_chunk(chunk_data, chunk_index, dynamic_prompt):
    retries = 0
    while retries < 3:
        try:
            file_like = io.BytesIO(chunk_data)
            file_like.name = f"chunk_{chunk_index}.wav"
            
            # เรียก Typhoon พร้อม Dynamic Prompt
            response = asr_client.audio.transcriptions.create(
                model="typhoon-asr-realtime", 
                file=file_like, 
                language="th", 
                prompt=dynamic_prompt, # <--- KEY POINT
                temperature=0.2
            )
            print(f"    ✅ [Chunk {chunk_index:02d}] Done.")
            return response.text
        except Exception as e:
            retries += 1; print(f"    ⚠️ [Chunk {chunk_index:02d}] Error: {e}"); time.sleep(1)
    return ""

def merge_transcriptions(transcripts):
    if not transcripts: return ""
    full_text = transcripts[0].strip()
    for i in range(1, len(transcripts)):
        prev = full_text
        curr = transcripts[i].strip()
        if not curr: continue
        # Simple overlap matching
        matcher = difflib.SequenceMatcher(None, prev[-500:], curr[:500])
        match = matcher.find_longest_match(0, len(prev[-500:]), 0, 500)
        if match.size > 10: 
            full_text += curr[match.b + match.size:]
        else: 
            full_text += " " + curr
    return re.sub(r"\s+", " ", full_text).strip()

def main():
    if not os.path.exists(TRANSCRIPT_OUTPUT_DIR): os.makedirs(TRANSCRIPT_OUTPUT_DIR)
    
    # 1. Get Video Metadata & Download
    print("\n⬇️  [Step 1] Fetching Metadata & Audio...")
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
        "quiet": True, "no_warnings": True
    }
    
    video_meta = {}
    audio_filename = ""
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # 1.1 Extract Info First
        info = ydl.extract_info(YOUTUBE_URL, download=False)
        video_meta = {
            'title': info.get('title', ''),
            'description': info.get('description', ''),
            'tags': info.get('tags', [])
        }
        print(f"    📄 Title: {video_meta['title']}")
        
        # 1.2 Generate Dynamic Prompt
        dynamic_prompt = prompt_builder.build_prompt(video_meta)
        print(f"    🎯 Dynamic Prompt: {dynamic_prompt}")
        
        # 1.3 Download
        print("    ⬇️  Downloading...")
        ydl.download([YOUTUBE_URL])
        audio_filename = ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".m4a", ".mp3")
        if not os.path.exists(audio_filename):
            # Fallback check
            audio_filename = os.path.join(DOWNLOAD_DIR, f"{info['id']}.mp3")

    # 2. Preprocess Audio
    print("\n🔊 [Step 2] Preprocessing Audio...")
    audio = AudioProcessor.preprocess_audio(audio_filename)
    if not audio: return

    # 3. Transcribe
    print(f"\n🚀 [Step 3] Transcribing with Typhoon (Prompt Aware)...")
    chunk_ms = CHUNK_DURATION_SEC * 1000
    overlap_ms = OVERLAP_DURATION_SEC * 1000
    chunks = []
    
    # Create chunks
    for i, s in enumerate(range(0, len(audio), chunk_ms - overlap_ms)):
        buf = io.BytesIO()
        chunk_segment = audio[s:min(s+chunk_ms, len(audio))]
        chunk_segment.export(buf, format="wav")
        chunks.append({"data": buf.getvalue(), "index": i})
    
    # Parallel Transcribe
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(transcribe_chunk, c["data"], c["index"], dynamic_prompt): c["index"] for c in chunks}
        for f in concurrent.futures.as_completed(futures):
            results[futures[f]] = f.result()
            
    raw_text = merge_transcriptions([results[i] for i in sorted(results.keys())])
    
    with open(os.path.join(TRANSCRIPT_OUTPUT_DIR, RAW_TRANSCRIPT_FILE), "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"    ✅ Raw Transcript Saved ({len(raw_text)} chars)")

    # 4. Cleaning
    print("\n🧠 [Step 4] Cleaning...")
    clean_text = cleaning_chain.invoke({"raw_text": raw_text})
    
    # 5. Correction (Senior Analyst)
    print("\n🧠 [Step 5] Correcting with Investment Logic...")
    corrected_text = correction_chain.invoke({"clean_text": clean_text})
    with open(os.path.join(TRANSCRIPT_OUTPUT_DIR, CLEAN_TRANSCRIPT_FILE), "w", encoding="utf-8") as f:
        f.write(corrected_text)
        
    # 6. NER & Resolution
    print("\n🤖 [Step 6] Identifying & Verifying Tickers...")
    try:
        ner_res = ner_chain.invoke({"text": corrected_text[:30000]})
        entities = ner_res.entities
    except: entities = []
    
    mappings = []
    seen = set()
    for ent in entities:
        if ent.text_found in seen: continue
        seen.add(ent.text_found)
        
        real_ticker = resolver.resolve(ent.text_found)
        if real_ticker:
            clean_tk = real_ticker.replace('.BK', '')
            mappings.append(f"- {ent.text_found} -> {clean_tk}")
            print(f"    ✅ {ent.text_found} -> {clean_tk}")
    
    mapping_str = "\n".join(mappings)

    # 7. Final Summary
    print("\n📝 [Step 7] Generating Final Infographic Summary...")
    final_md = summary_chain.invoke({
        "corrected_text": corrected_text,
        "mapping_str": mapping_str
    })
    
    with open(os.path.join(TRANSCRIPT_OUTPUT_DIR, MARKDOWN_FILE), "w", encoding="utf-8") as f:
        f.write(final_md)
        
    print(f"\n✅ SUCCESS! All files saved in '{TRANSCRIPT_OUTPUT_DIR}'")
    
    # ---------------------------------------------------------
    # DISPLAY BOTH OUTPUTS (Output 1 & Output 2)
    # ---------------------------------------------------------
    
    print("\n" + "="*20 + " OUTPUT 1: FINAL TRANSCRIPT (FULL NARRATIVE) " + "="*20 + "\n")
    print(corrected_text)
    print("\n" + "="*22 + " END OUTPUT 1 " + "="*22 + "\n")

    print("\n" + "="*20 + " OUTPUT 2: FINAL SUMMARY (INFOGRAPHIC) " + "="*20 + "\n")
    print(final_md)
    print("\n" + "="*25 + " END OUTPUT 2 " + "="*25 + "\n")

if __name__ == "__main__":
    main()