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
import argparse
import math
from datetime import datetime

from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydub import AudioSegment, effects
from tqdm import tqdm

# Load Optional Libraries
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None
    print("⚠️ Warning: 'langchain-text-splitters' not installed. Fallback to simple splitting.")

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from thefuzz import process
except ImportError:
    process = None
    print("⚠️ Warning: 'thefuzz' not installed. Fuzzy matching will be disabled.")

# -------------------------------------------------
# Configuration & Suppression
# -------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("yt_dlp").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

# --- API KEYS ---
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_API_KEY = "sk-vCE2QnUydpGnzic35kI3IcoTsAeWzb2X3jYCCAXDPmfT2JnN"  # <--- เปลี่ยนเอง

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GOOGLE_API_KEY = "AIzaSyCkjWUucxzLaRPnuklKxKgYP0fUyQhTwHA"    # <--- เปลี่ยนเอง
LLM_MODEL_NAME = "gemini-2.0-flash"

# --- PATHS ---
DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=ks2e22C2zGA"
DOWNLOAD_DIR = "downloads"
TRANSCRIPT_OUTPUT_DIR = "transcripts_output"
MASTER_KB_FILE = "knowledge_base.json"
CACHE_FILE = "ticker_cache.json"
FINANCE_TERM_FILE = "finance_terms.json"

MAX_RETRIES = 5  # เพิ่ม Retry ให้ทนทานขึ้น


# -------------------------------------------------
# Helper Functions & Dynamic Config
# -------------------------------------------------
def get_adaptive_config(duration_sec: float) -> dict:
    """
    คำนวณค่า Config ตามความยาวคลิป (3 Tiers Strategy)
    """
    config = {}
    
    if duration_sec <= 1740: # Tier 1: 0 - 29 นาที
        config = {
            "CHUNK_DURATION": 45,
            "OVERLAP_DURATION": 15,
            "MAX_WORKERS": 5,
            "TEXT_CHUNK_SIZE": 4000,
            "MODE_NAME": "Tier 1: Short-Form (Fast & Agile)"
        }
    elif duration_sec <= 3600: # Tier 2: 30 - 60 นาที
        config = {
            "CHUNK_DURATION": 60,
            "OVERLAP_DURATION": 15,
            "MAX_WORKERS": 3,
            "TEXT_CHUNK_SIZE": 4000,
            "MODE_NAME": "Tier 2: Medium-Form (Balanced)"
        }
    else: # Tier 3: 1 ชม. ขึ้นไป (Max 2 ชม.)
        # ใช้ Chunk ใหญ่ (10 นาที) เพื่อลดจำนวน Request ลง 10 เท่า ป้องกัน Rate Limit
        config = {
            "CHUNK_DURATION": 600, 
            "OVERLAP_DURATION": 20,
            "MAX_WORKERS": 2, # ลด Worker เพื่อความเสถียร
            "TEXT_CHUNK_SIZE": 3000, # ลดขนาด Text เพื่อกัน Output ตัดจบ
            "MODE_NAME": "Tier 3: Long-Form (Stability Focus)"
        }
    return config

def sanitize_filename(name: str) -> str:
    if not name: return "audio_output"
    name = re.sub(r'[^\w\u0E00-\u0E7F\s-]', '', name)
    name = name.strip().replace(' ', '_')
    return name[:50]

def format_transcript_paragraphs(text: str) -> str:
    if not text: return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    cleaned = []
    for line in lines:
        if line == "":
            if cleaned and cleaned[-1] == "": continue
            cleaned.append("")
        else:
            cleaned.append(line)
    return "\n".join(cleaned).strip()

def normalize_markdown_bullets(md: str) -> str:
    """ เปลี่ยน * เป็น - เพื่อให้อ่านง่าย """
    if not md: return md
    return re.sub(r"^(\s*)\* ", r"\1- ", md, flags=re.MULTILINE)

def get_file_size_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024 * 1024)

def format_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"

def split_text_smart(text, chunk_size):
    """ หั่น Text ตามย่อหน้า (Smart Chunking) """
    if RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0, # จัดการ overlap เองใน logic
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)
    else:
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# -------------------------------------------------
# PART 1: AUDIO PROCESSOR
# -------------------------------------------------
class AudioProcessor:
    @staticmethod
    def preprocess_audio(file_path: str):
        try:
            audio = AudioSegment.from_file(file_path)
            
            duration_sec = len(audio) / 1000.0
            size_mb = get_file_size_mb(file_path)
            
            print(f"\n📊 Audio Stats:")
            print(f"   - File Size:  {size_mb:.2f} MB")
            print(f"   - Duration:   {format_duration(duration_sec)}")
            
            if audio.channels > 1:
                audio = audio.set_channels(1)
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
            audio = effects.normalize(audio)
            return audio
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            return None


# -------------------------------------------------
# PART 2: CONTEXT & MANAGERS
# -------------------------------------------------
class StockContextManager:
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
                        self.search_corpus[clean_ticker.lower()] = ticker
                        self.fuzzy_corpus[clean_ticker.lower()] = ticker
                        for alias in aliases:
                            self.search_corpus[alias.lower()] = ticker
                            self.fuzzy_corpus[alias.lower()] = ticker
            print(f"📚 Knowledge Base Loaded: {len(self.all_tickers)} Tickers")
        except Exception as e:
            print(f"⚠️ Error loading KB: {e}")

    def get_sector_prompt_str(self) -> str:
        lines = ["--- รายชื่อหุ้นแยกตามกลุ่มอุตสาหกรรม (Reference Context) ---"]
        for sector, stocks in self.sector_data.items():
            if isinstance(stocks, dict):
                tickers = [t.replace(".BK", "") for t in stocks.keys()]
                lines.append(f"## กลุ่ม {sector}:\n    - {', '.join(tickers)}")
        return "\n".join(lines)


class FinanceTermManager:
    def __init__(self, term_file: str = FINANCE_TERM_FILE):
        self.term_file = term_file
        self.data = {}
        self._load_terms()

    def _load_terms(self):
        if not os.path.exists(self.term_file): return
        try:
            with open(self.term_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            print(f"📘 Finance Terms Loaded")
        except Exception as e:
            print(f"⚠️ Error loading finance_terms.json: {e}")

    def get_prompt_str(self) -> str:
        if not self.data: return "--- ไม่มี finance terms context ---"
        lines = ["--- Domain Finance Terms & Pronunciation Patterns (Reference Only) ---"]
        for group, mapping in self.data.items():
            lines.append(f"### {group}")
            for canonical, aliases in mapping.items():
                lines.append(f"- {canonical}: {', '.join(aliases)}")
        return "\n".join(lines)


class DynamicPromptBuilder:
    def __init__(self, context_mgr: StockContextManager, term_mgr: FinanceTermManager):
        self.ctx = context_mgr
        self.term_mgr = term_mgr
        
    def build_prompt(self, metadata: dict) -> str:
        title = metadata.get("title", "")
        channel = metadata.get("channel", "")
        combined_text = f"{channel} {title}"
        candidates = re.findall(r"\b[A-Z]{2,8}\b", combined_text.upper())
        priority_tickers = [c for c in candidates if c in self.ctx.all_tickers]

        vocab_list = []
        if self.term_mgr.data:
            for group, mapping in self.term_mgr.data.items():
                vocab_list.extend(mapping.keys())
        
        selected_vocab = random.sample(vocab_list, min(len(vocab_list), 30)) if vocab_list else []

        prompt_parts = []
        if channel: prompt_parts.append(f"ชื่อรายการ: {channel}")
        if priority_tickers: prompt_parts.append(f"หุ้นในคลิป: {', '.join(set(priority_tickers))}")
        if selected_vocab: prompt_parts.append(f"คำศัพท์: {', '.join(selected_vocab)}")
            
        return " | ".join(prompt_parts)[:1000]


# -------------------------------------------------
# PART 3: SMART RESOLVER (SAFE GUARDED)
# -------------------------------------------------
class MarketTools:
    BLACKLIST_KEYWORDS = {
        "SET", "MAI", "BKK", "THAI", "PRICE", "NEWS", 
        "STOCK", "TRADE", "DATA", "INFO", "VS", "AND",
        "TODAY", "NOW", "LIVE", "UPDATE", "INDEX"
    }

    @staticmethod
    def search_ticker(query: str) -> Optional[str]:
        if DDGS is None: return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with DDGS() as ddgs:
                    results = list(ddgs.text(f"หุ้น {query} stock symbol ticker settrade", max_results=2))
                    if not results: return None
                    blob = " ".join([r.get("title", "") + " " + r.get("body", "") for r in results]).upper()
                    
                    candidates = re.findall(r"\b([A-Z]{2,8})\.BK\b", blob)
                    if candidates: return candidates[0] + ".BK"
                    
                    candidates = re.findall(r"\b([A-Z]{2,8})\b", blob)
                    for ticker in candidates:
                        if ticker not in MarketTools.BLACKLIST_KEYWORDS and len(ticker) >= 2:
                            return ticker
        except Exception:
            return None
        return None

    @staticmethod
    def verify_ticker(ticker: str) -> bool:
        if yf is None or not ticker: return False
        if "SET.BK" in ticker or "^" in ticker: return False
        try:
            hist = yf.Ticker(ticker).history(period="1d")
            return not hist.empty
        except Exception:
            return False


class SmartMarketResolver:
    def __init__(self, context_mgr: StockContextManager, cache_file: str = CACHE_FILE):
        self.ctx = context_mgr
        self.memory = context_mgr.search_corpus.copy()
        self.cache_file = cache_file
        self.load_cache()

    def load_cache(self):
        if not os.path.exists(self.cache_file): return
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
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
        mention_upper = mention.upper().strip()

        # 1. Exact Match
        if mention_key in self.memory: return self.memory[mention_key]
        if mention_upper in self.ctx.all_tickers: return mention_upper + ".BK"

        # 2. Fuzzy Match (With Safety Filters)
        if process:
            best_match, score = process.extractOne(mention_key, self.ctx.fuzzy_corpus.keys())
            ticker = self.ctx.fuzzy_corpus[best_match]
            ticker_clean = ticker.replace(".BK", "")

            # Safety Rule A: หุ้นชื่อสั้น (<= 3 ตัว) ต้อง Exact Match 100%
            if len(ticker_clean) <= 3 and score < 100:
                return None

            # Safety Rule B: ภาษาอังกฤษต้องมั่นใจสูง (ป้องกัน Bitcoin -> SSET)
            is_english = re.match(r"^[A-Za-z]+$", mention)
            if is_english and score < 95:
                return None

            if score >= 90:
                print(f"    🔍 Fuzzy match: '{mention}' -> '{best_match}' ({score}%) -> {ticker}")
                self.save_cache(mention_key, ticker)
                return ticker
        
        return None


# -------------------------------------------------
# PART 4: LLM AGENTS
# -------------------------------------------------
llm = ChatOpenAI(
    base_url=GOOGLE_BASE_URL,
    api_key=GOOGLE_API_KEY,
    model=LLM_MODEL_NAME,
    temperature=0.1,
    max_tokens=8192,
)

correction_system_prompt = (
    "คุณคือ 'Senior Investment Analyst Editor' ประจำช่อง '{channel_name}'\n"
    "หน้าที่: ตรวจสอบและแก้ไข Transcript ให้ถูกต้อง โดย **ต้องคงเนื้อหาเดิมไว้ให้ครบถ้วนทุกประโยค** (Verbatim Editing)\n\n"
    "{sector_context}\n\n"
    "{domain_terms_context}\n\n"
    "--- กฎการทำงาน (Strict Rules) ---\n"
    "1) **ห้ามสรุปความ (No Summarization):** ห้ามตัดทอนเนื้อหา ห้ามรวมประโยค ให้แก้แค่คำผิด วรรคตอน และย่อหน้าเท่านั้น\n"
    "2) **Context Aware:** Input จะมี [บริบทก่อนหน้า: ...] แปะมาด้วย ให้อ่านเพื่อความเข้าใจ แต่ **ห้ามพิมพ์ส่วนบริบทนั้นกลับมา** ให้เริ่มพิมพ์เฉพาะเนื้อหาใน Chunk ปัจจุบัน\n"
    "3) **Accuracy:** ตรวจสอบชื่อหุ้น/Crypto ระวังหุ้นชื่อซ้ำคำทั่วไป\n"
    "4) **Format:** แบ่งย่อหน้าให้อ่านง่าย แต่เนื้อหาต้องครบ 100%"
)
correction_prompt = ChatPromptTemplate.from_messages([
    ("system", correction_system_prompt),
    ("user", 'Transcript Chunk:\n""" {clean_text} """'),
])
correction_chain = correction_prompt | llm | StrOutputParser()

class StockEntity(BaseModel):
    text_found: str = Field(...)
class EntityList(BaseModel):
    entities: List[StockEntity]

ner_prompt = ChatPromptTemplate.from_messages([
    ("system", "ดึงรายชื่อหุ้น (Ticker), Crypto หรือชื่อบริษัทลงทุน ออกมาเป็น JSON"),
    ("user", 'Text:\n""" {text} """'),
])
ner_chain = ner_prompt | llm.with_structured_output(EntityList)

summary_system_prompt = (
    "คุณคือ 'Infographic Content Lead' ของช่อง '{channel_name}'\n"
    "โจทย์: สรุปข้อมูลเป็น Markdown\n"
    "1) ห้ามเขียนสั้นห้วน ระบุ WHAT, HOW MUCH, WHY\n"
    "2) ห้ามทำหัวข้อซ้ำ\n"
    "3) ใช้เครื่องหมาย '-' สำหรับ bullet\n\n"
    "--- Template ---\n"
    "# สรุปภาวะตลาด\n"
    "## ภาพรวม\n"
    "## หุ้นขึ้น\n"
    "## หุ้นลง\n"
    "## กลยุทธ์\n"
    "## เทคนิค\n"
    "## ข่าวสาร\n\n"
    "[VERIFIED LIST]\n"
    "{mapping_str}"
)
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", summary_system_prompt),
    ("user", 'Transcript:\n""" {corrected_text} """'),
])
summary_chain = summary_prompt | llm | StrOutputParser()


# -------------------------------------------------
# PART 5: ROBUST WORKFLOW LOGIC
# -------------------------------------------------
def process_large_text_robust(chain, full_text, chunk_size, **kwargs):
    """
    Stateful Processing with Retry Logic & Smart Chunking
    """
    chunks = split_text_smart(full_text, chunk_size)
    print(f"   ... Processing {len(chunks)} chunks sequentially ...")
    
    processed_parts = []
    previous_context = "" 
    
    for i, chunk in enumerate(chunks):
        input_args = kwargs.copy()
        
        # 1. Prepare Input with Context
        current_input_text = chunk
        if i > 0 and previous_context:
            current_input_text = f"[บริบทก่อนหน้า: ...{previous_context}]\n{chunk}"
            
        if "clean_text" in input_args: input_args["clean_text"] = current_input_text
        elif "raw_text" in input_args: input_args["raw_text"] = current_input_text
        else: input_args["clean_text"] = current_input_text

        # 2. Retry Loop
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                res = chain.invoke(input_args)
                
                # Clean up context echo
                if "[บริบทก่อนหน้า:" in res: res = res.split("]\n")[-1].strip()
                if "[บริบทก่อนหน้า" in res: res = res.split("]")[-1].strip()

                processed_parts.append(res)
                previous_context = res[-300:] 
                success = True
                break
            except Exception as e:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"      ⚠️ Chunk {i+1} Error (Attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        if not success:
            print(f"      ❌ Chunk {i+1} Failed after retries. Using original text.")
            processed_parts.append(chunk)
            previous_context = chunk[-300:]

    return "\n\n".join(processed_parts)


def run_correction_loop(clean_text, ctx_mgr, term_mgr, resolver, channel_name, chunk_size):
    # [UPDATED] Use robust processing with dynamic chunk size
    corrected = process_large_text_robust(
        correction_chain, 
        clean_text, 
        chunk_size=chunk_size,
        clean_text="placeholder",
        sector_context=ctx_mgr.get_sector_prompt_str(),
        domain_terms_context=term_mgr.get_prompt_str(),
        feedback_msg="ไม่มีคำแนะนำพิเศษ",
        channel_name=channel_name
    )

    try:
        ner_res = ner_chain.invoke({"text": corrected[:30000]})
        entities = ner_res.entities
    except:
        entities = []

    valid_mappings = []
    unknown_tickers = []
    seen = set()

    for ent in entities:
        if ent.text_found in seen: continue
        seen.add(ent.text_found)
        
        tk = resolver.resolve(ent.text_found)
        if tk:
            clean_tk = tk.replace(".BK", "")
            valid_mappings.append(f"- {ent.text_found} -> {clean_tk}")
        else:
            unknown_tickers.append(ent.text_found)

    if unknown_tickers:
        print(f"   ⚠️ Found Unknown Tickers: {unknown_tickers}")
    else:
        print("   ✅ Verification Passed! All tickers found.")

    return corrected, "\n".join(valid_mappings)


# -------------------------------------------------
# PART 6: MAIN EXECUTION
# -------------------------------------------------
def transcribe_chunk(client, chunk_data, idx, prompt):
    retries = 0
    while retries < 3:
        try:
            file_like = io.BytesIO(chunk_data)
            file_like.name = f"chunk_{idx}.wav"
            response = client.audio.transcriptions.create(
                model="typhoon-asr-realtime",
                file=file_like,
                language="th",
                prompt=prompt,
                temperature=0.2,
            )
            return response.text
        except Exception as e:
            retries += 1
            time.sleep(2)
    return ""

def merge_transcriptions(transcripts: List[str]) -> str:
    if not transcripts: return ""
    full_text = transcripts[0].strip()
    for i in range(1, len(transcripts)):
        prev = full_text
        curr = transcripts[i].strip()
        if not curr: continue
        matcher = difflib.SequenceMatcher(None, prev[-500:], curr[:500])
        match = matcher.find_longest_match(0, len(prev[-500:]), 0, 500)
        if match.size > 10:
            full_text += curr[match.b + match.size :]
        else:
            full_text += " " + curr
    return re.sub(r"\s+", " ", full_text).strip()

def main(target_url: str):
    ctx_mgr = StockContextManager()
    term_mgr = FinanceTermManager()
    prompt_builder = DynamicPromptBuilder(ctx_mgr, term_mgr)
    resolver = SmartMarketResolver(ctx_mgr)

    try:
        http_client = httpx.Client(timeout=120.0)
        from openai import OpenAI
        asr_client = OpenAI(
            base_url=TYPHOON_BASE_URL,
            api_key=TYPHOON_API_KEY,
            http_client=http_client,
        )
    except Exception as e:
        print(f"❌ Error init Typhoon Client: {e}")
        return

    if not os.path.exists(DOWNLOAD_DIR): os.makedirs(DOWNLOAD_DIR)
    if not os.path.exists(TRANSCRIPT_OUTPUT_DIR): os.makedirs(TRANSCRIPT_OUTPUT_DIR)

    print(f"\n⬇️  [Step 1] Fetching Metadata & Audio from: {target_url}")
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(target_url, download=False)
        video_meta = {
            "title": info.get("title", "Unknown_Video"),
            "description": info.get("description", ""),
            "tags": info.get("tags", []),
            "channel": info.get("uploader", ""),
            "duration": info.get("duration", 0)
        }
        print(f"    📄 Title: {video_meta['title']}")
        print(f"    📺 Channel: {video_meta['channel']}")
        
        dynamic_prompt = prompt_builder.build_prompt(video_meta)
        print(f"    🎯 Dynamic Prompt: {dynamic_prompt}")

        # [NEW] Adaptive Configuration
        duration_sec = video_meta.get("duration", 0)
        adaptive_cfg = get_adaptive_config(duration_sec)
        
        print(f"\n⚙️  Adaptive Config for {format_duration(duration_sec)}:")
        print(f"    - Mode: {adaptive_cfg['MODE_NAME']}")
        print(f"    - ASR Chunk: {adaptive_cfg['CHUNK_DURATION']}s")
        print(f"    - Workers: {adaptive_cfg['MAX_WORKERS']}")
        print(f"    - Text Split: {adaptive_cfg['TEXT_CHUNK_SIZE']} chars")

        print("    ⬇️  Downloading audio...")
        ydl.download([target_url])
        audio_filename = ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".m4a", ".mp3")
        if not os.path.exists(audio_filename):
            audio_filename = os.path.join(DOWNLOAD_DIR, f"{info['id']}.mp3")

    # Filename Logic
    safe_title = sanitize_filename(video_meta['title'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    raw_filename = f"{safe_title}_RAW_{timestamp}.txt"
    clean_filename = f"{safe_title}_CLEAN_{timestamp}.txt"
    markdown_filename = f"{safe_title}_SUMMARY_{timestamp}.md"

    print("\n🔊 [Step 2] Preprocessing Audio...")
    audio = AudioProcessor.preprocess_audio(audio_filename)
    if not audio: return

    print("\n🚀 [Step 3] Transcribing with Typhoon...")
    # Use adaptive config
    chunk_ms = adaptive_cfg["CHUNK_DURATION"] * 1000
    step = chunk_ms - (adaptive_cfg["OVERLAP_DURATION"] * 1000)
    
    chunks = []
    for i, s in enumerate(range(0, len(audio), step)):
        buf = io.BytesIO()
        audio[s : min(s + chunk_ms, len(audio))].export(buf, format="wav")
        chunks.append({"data": buf.getvalue(), "index": i})

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=adaptive_cfg["MAX_WORKERS"]) as ex:
        futures = {
            ex.submit(transcribe_chunk, asr_client, c["data"], c["index"], dynamic_prompt): c["index"]
            for c in chunks
        }
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="   Processing Chunks", unit="chunk"):
            idx = futures[f]
            results[idx] = f.result()

    raw_text = merge_transcriptions([results[i] for i in sorted(results.keys())])
    
    with open(os.path.join(TRANSCRIPT_OUTPUT_DIR, raw_filename), "w", encoding="utf-8") as f:
        f.write(raw_text)

    print("\n🧠 [Step 4] Intelligent Cleaning & Correction...")
    channel_name = video_meta.get("channel", "Unknown Channel")
    
    # [UPDATED] Use adaptive text chunk size
    corrected_text, mapping_str = run_correction_loop(
        raw_text, 
        ctx_mgr, 
        term_mgr, 
        resolver, 
        channel_name,
        chunk_size=adaptive_cfg["TEXT_CHUNK_SIZE"]
    )
    
    corrected_text = format_transcript_paragraphs(corrected_text)
    
    with open(os.path.join(TRANSCRIPT_OUTPUT_DIR, clean_filename), "w", encoding="utf-8") as f:
        f.write(corrected_text)

    print("\n📝 [Step 6] Generating Final Infographic Summary...")
    final_md = summary_chain.invoke({
        "corrected_text": corrected_text, 
        "mapping_str": mapping_str, 
        "channel_name": channel_name
    })
    
    final_md = normalize_markdown_bullets(final_md)

    with open(os.path.join(TRANSCRIPT_OUTPUT_DIR, markdown_filename), "w", encoding="utf-8") as f:
        f.write(final_md)

    print(f"\n✅ SUCCESS! All files saved in '{TRANSCRIPT_OUTPUT_DIR}':")
    print(f"   1. {raw_filename}")
    print(f"   2. {clean_filename}")
    print(f"   3. {markdown_filename}")
    
    print("\n" + "="*20 + " OUTPUT 1: FINAL TRANSCRIPT " + "="*20 + "\n")
    # Show more context for checking
    print(corrected_text[:3000] + ("..." if len(corrected_text) > 3000 else ""))
    print("\n" + "="*22 + " END OUTPUT 1 " + "="*22 + "\n")

    print("\n" + "="*20 + " OUTPUT 2: FINAL SUMMARY " + "="*20 + "\n")
    print(final_md)
    print("\n" + "="*25 + " END OUTPUT 2 " + "=" * 25 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Agentic ASR Workflow")
    parser.add_argument("--url", type=str, help="Youtube URL")
    args = parser.parse_args()
    
    target_url = args.url if args.url else DEFAULT_YOUTUBE_URL
    main(target_url)