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

from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydub import AudioSegment

# -------------------------------------------------
# ปิด Warning
# -------------------------------------------------
warnings.filterwarnings("ignore")

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

try:
    import yfinance as yf
except ImportError:
    yf = None

# ---------------- CONFIGURATION ----------------
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_API_KEY = "sk-vCE2QnUydpGnzic35kI3IcoTsAeWzb2X3jYCCAXDPmfT2JnN"

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GOOGLE_API_KEY = "AIzaSyCkjWUucxzLaRPnuklKxKgYP0fUyQhTwHA"
LLM_MODEL_NAME = "gemini-2.0-flash"

YOUTUBE_URL = "https://www.youtube.com/watch?v=opEIqiPzx64"

DOWNLOAD_DIR = "downloads"
TRANSCRIPT_OUTPUT_DIR = "transcripts_output"
RAW_TRANSCRIPT_FILE = "raw_transcript_full.txt"
CLEAN_TRANSCRIPT_FILE = "final_transcript_clean.txt"
MARKDOWN_FILE = "final_summary_markdown.md"

MASTER_KB_FILE = "knowledge_base.json"
CACHE_FILE = "ticker_cache.json"

CHUNK_DURATION_SEC = 45
OVERLAP_DURATION_SEC = 15
MAX_WORKERS = 5

# ---------------- PART 1: SMART CONTEXT MANAGER ----------------

class StockContextManager:
    def __init__(self, kb_file=MASTER_KB_FILE):
        self.kb_file = kb_file
        self.sector_data = {}
        self.flat_memory = {}
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
                                for alias in aliases:
                                    self.flat_memory[alias.lower()] = ticker
                print(f"📚 Context Manager Loaded: {len(self.sector_data)} Sectors")
            except Exception as e:
                print(f"⚠️ Error loading KB: {e}")

    def get_sector_prompt_str(self):
        lines = []
        for sector, stocks in self.sector_data.items():
            if isinstance(stocks, dict):
                tickers = list(stocks.keys())
                clean_tickers = [t.replace('.BK', '') for t in tickers]
                lines.append(f"- {sector}: {', '.join(clean_tickers)}")
        return "\n".join(lines)

    def get_vocab_prompt(self):
        vocab_list = []
        for sector, stocks in self.sector_data.items():
            if isinstance(stocks, dict):
                for ticker, aliases in stocks.items():
                    clean_ticker = ticker.replace('.BK', '')
                    vocab_list.append(clean_ticker)
                    thai_aliases = [a for a in aliases if re.search(r'[ก-๙]', a)]
                    vocab_list.extend(thai_aliases)
        vocab_list.extend(["งบการเงิน", "แนวรับ", "แนวต้าน", "ดัชนี", "SET", "SET50", "กำไร", "ไตรมาส"])
        if len(vocab_list) > 100:
            selected_vocab = random.sample(vocab_list, 100)
        else:
            selected_vocab = vocab_list
        return f"คำศัพท์เฉพาะในคลิป: {', '.join(selected_vocab)}"

ctx_mgr = StockContextManager()
SECTOR_CONTEXT_STR = ctx_mgr.get_sector_prompt_str()
ASR_VOCAB_PROMPT = ctx_mgr.get_vocab_prompt()
print(f"🎤 ASR Vocab Hint: {ASR_VOCAB_PROMPT[:100]}...")

# ---------------- PART 2: SMART RESOLVER ----------------

class MarketTools:
    @staticmethod
    def search_ticker(query: str) -> Optional[str]:
        if DDGS is None: return None
        try:
            search_q = f"หุ้น {query} stock symbol ticker settrade"
            with DDGS() as ddgs:
                results = list(ddgs.text(search_q, max_results=3))
                if not results: return None
                blob = " ".join([r.get("title", "") + " " + r.get("body", "") for r in results]).upper()
                candidates = re.findall(r"\b([A-Z]{2,8})\.BK\b", blob)
                if candidates: return candidates[0] + ".BK"
                candidates = re.findall(r"\b([A-Z]{2,8})\b", blob)
                blacklist = {"SET", "MAI", "BKK", "THAI", "PRICE", "NEWS", "STOCK", "TRADE", "DATA", "INFO"}
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
            return not yf.Ticker(ticker).history(period="5d").empty
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
        if mention_key in self.memory: return self.memory[mention_key]
        print(f"   🌐 [Searching] '{mention}' ...")
        found = MarketTools.search_ticker(mention)
        time.sleep(1.5)
        if found:
            clean = found.upper().strip()
            if MarketTools.verify_ticker(clean):
                self.save_cache(mention_key, clean)
                return clean
            if ".BK" not in clean:
                thai = clean + ".BK"
                if MarketTools.verify_ticker(thai):
                    self.save_cache(mention_key, thai)
                    return thai
        return None

resolver = SmartMarketResolver(ctx_mgr)

# ---------------- PART 3: ASR & DOWNLOAD ----------------

def download_audio_from_youtube(url, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print("⬇️  Downloading audio...")
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": True, "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".m4a", ".mp3")
        if not os.path.exists(filename): filename = os.path.join(output_dir, f"{info['id']}.mp3")
    return filename

from openai import OpenAI
try:
    http_client = httpx.Client(timeout=120.0)
    asr_client = OpenAI(base_url=TYPHOON_BASE_URL, api_key=TYPHOON_API_KEY, http_client=http_client)
except Exception as e:
    print(f"❌ Error clients: {e}"); raise SystemExit

def transcribe_chunk_safe(chunk_data, chunk_index, prompt):
    retries = 0
    while retries < 3:
        try:
            file_like = io.BytesIO(chunk_data)
            file_like.name = f"chunk_{chunk_index}.wav"
            response = asr_client.audio.transcriptions.create(
                model="typhoon-asr-realtime", file=file_like, language="th", prompt=prompt, temperature=0.2
            )
            print(f"   ✅ [Chunk {chunk_index:02d}] Done.")
            return response.text
        except Exception as e:
            retries += 1; print(f"   ⚠️ [Chunk {chunk_index:02d}] Error: {e}"); time.sleep(1)
    return ""

def merge_transcriptions_fuzzy_overlap(transcripts):
    if not transcripts: return ""
    final_text = transcripts[0].strip()
    for i in range(1, len(transcripts)):
        prev = final_text
        curr = transcripts[i].strip()
        if not curr: continue
        matcher = difflib.SequenceMatcher(None, prev[-600:], curr[:600])
        match = matcher.find_longest_match(0, len(prev[-600:]), 0, 600)
        if match.size > 10: final_text += curr[match.b + match.size:]
        else: final_text += " " + curr
    return re.sub(r"\s+", " ", final_text).strip()

# ---------------- PART 4: LANGCHAIN AGENTS (CLEAN FORMAT) ----------------

llm = ChatOpenAI(base_url=GOOGLE_BASE_URL, api_key=GOOGLE_API_KEY, model=LLM_MODEL_NAME, temperature=0.15, max_tokens=4096)

class StockEntity(BaseModel):
    text_found: str = Field(...)
    predicted_ticker: Optional[str] = Field(None)
    confidence: str = Field(...)

class EntityList(BaseModel):
    entities: List[StockEntity]

# Chain 1: Cleaning
clean_prompt = ChatPromptTemplate.from_messages([
    ("system", "คุณคือ Transcript Editor หน้าที่: ตัดคำฟุ่มเฟือย (เอ่อ, นะครับ) และจัดย่อหน้าให้อ่านง่าย แต่ห้ามเปลี่ยนเนื้อหา"),
    ("user", "Transcript:\n\"\"\"{raw_text}\"\"\"")
])
cleaning_chain = clean_prompt | llm | StrOutputParser()

# Chain 2: Correction (Logic + Ecosystem Check)
correction_system_prompt = (
    "คุณคือ 'Senior Investment Analyst' ที่มีความเชี่ยวชาญเรื่อง 'ความสัมพันธ์ของกลุ่มหุ้น'\n"
    "ภารกิจ: ตรวจสอบและแก้ไข Transcript โดยใช้ **'ตรรกะความสมเหตุสมผล'**\n\n"
    
    "--- ข้อมูลอ้างอิง (Sector Knowledge) ---\n"
    f"{SECTOR_CONTEXT_STR}\n\n"
    
    "--- กฎการแก้ไข (Correction Rules) ---\n"
    "1. **Neighbor Analysis**: หุ้นที่อยู่ติดกันมักเป็นกลุ่มเดียวกัน\n"
    "   - เจอ JMART, SINGER -> เพื่อนที่หายไปคือ **JMT** (ห้ามแก้เป็น MTC หรือ EMC)\n"
    "   - เจอ SAWAD -> คู่หูคือ **MTC** หรือ **TIDLOR**\n"
    "2. **Logic Check**: ถ้าพูดถึงกลุ่มไหน หุ้นในกลุ่มต้องถูกต้อง ห้ามมีหุ้นผิดกลุ่มโผล่มา\n"
    "3. **Phonetic**: ถ้าผิด Logic ให้ใช้เสียงช่วย (VANA->WHA, กราฟ->GULF)\n\n"
    "**คำสั่ง**: แก้ไขให้ถูกต้อง 100% และคืนค่าเป็น Transcript ฉบับสมบูรณ์"
)
correction_prompt = ChatPromptTemplate.from_messages([
    ("system", correction_system_prompt),
    ("user", "Transcript to Correct:\n\"\"\"{clean_text}\"\"\"")
])
correction_chain = correction_prompt | llm | StrOutputParser()

# Chain 3: NER
ner_prompt = ChatPromptTemplate.from_messages([
    ("system", "ดึงรายชื่อหุ้น (Ticker) จากข้อความ ตอบเป็น JSON"),
    ("user", "Text:\n\"\"\"{text}\"\"\"")
])
ner_chain = ner_prompt | llm.with_structured_output(EntityList)

# Chain 4: Summary (*** NO BOLD FORMAT ***)
summary_system_prompt = (
    "คุณคือ Content Editor สรุปข่าวหุ้น\n"
    "โจทย์: สรุปข้อมูลเป็น Markdown เพื่อนำไปใช้งานต่อ โดยต้องสะอาดและอ่านง่าย\n\n"
    
    "--- กฎเหล็กการจัดรูปแบบ (Strict Formatting) ---\n"
    "1. **ห้ามใช้ตัวหนา (Bold) ที่ชื่อหุ้นเด็ดขาด**: (เช่น ห้ามเขียน **BEM**, ให้เขียน BEM เฉยๆ)\n"
    "2. **ห้ามใช้ตัวเอียง (Italic)**\n"
    "3. ใช้ Header (#, ##) แบ่งหัวข้อ\n"
    "4. ใช้ Bullet (-) สำหรับรายการ\n\n"

    "--- โครงสร้าง (Template) ---\n"
    "# สรุปภาวะตลาด\n\n"
    "## ภาพรวม\n"
    "- (สรุปสั้นๆ)\n\n"
    "## หุ้นที่น่าสนใจ\n"
    "- (แยกรายกลุ่ม เช่น กลุ่มเทค, กลุ่มไฟแนนซ์)\n\n"
    "## หุ้นแนะนำ\n"
    "- Ticker: (แนวรับ/แนวต้าน/กลยุทธ์/ธีม)\n"
    "  (ระบุให้ครบทุกตัว)\n\n"
    "## กลยุทธ์\n"
    "- (สรุปธีมการลงทุน)\n\n"
    "## หุ้นที่ต้องระมัดระวัง (ต่ำกว่า 52 สัปดาห์)\n"
    "- (ระบุชื่อหุ้น)\n\n"
    "## ข่าวอื่นๆ\n"
    "- (ข่าวรายตัว)\n\n"
    
    "[MAPPING]\n{mapping_str}"
)
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", summary_system_prompt),
    ("user", "Transcript:\n\"\"\"{corrected_text}\"\"\"")
])
summary_chain = summary_prompt | llm | StrOutputParser()

# Chain 5: Audit (*** FINAL FORMAT CHECK ***)
audit_system_prompt = (
    "คุณคือ Editor-in-Chief ตรวจสอบงานชิ้นสุดท้าย\n"
    "หน้าที่: ตรวจสอบบทสรุป Markdown\n"
    "1. **Format Check**: **ต้องไม่มีเครื่องหมายดอกจัน (*) ล้อมรอบชื่อหุ้น** (เช่น **BEM** ผิด, BEM ถูก) -> ถ้ามีให้ลบออกทันที\n"
    "2. **Logic Check**: ข้อมูลต้องถูกต้องตาม Transcript\n"
    "3. **Output**: ส่งคืนเนื้อหาบทสรุปฉบับสมบูรณ์ (Full Text)"
)
audit_prompt = ChatPromptTemplate.from_messages([
    ("system", audit_system_prompt),
    ("user", "Transcript:\n\"\"\"{transcript}\"\"\"\n\nDraft Summary:\n\"\"\"{markdown}\"\"\"")
])
audit_chain = audit_prompt | llm | StrOutputParser()

# ---------------- CONTROLLER ----------------

def run_ner_search_verify_agent(text: str):
    print("\n🤖 [Step 3: NER] Extracting & Verifying...")
    try:
        res = ner_chain.invoke({"text": text[:35000]})
        entities = res.entities
    except: return []

    mappings = []
    seen = set()
    for ent in entities:
        if ent.text_found in seen: continue
        seen.add(ent.text_found)
        ticker = resolver.resolve(ent.text_found)
        if ticker:
            clean = ticker.replace('.BK', '')
            mappings.append(f"- {ent.text_found} -> {clean}")
            print(f"   ✅ Mapped: {ent.text_found} -> {clean}")
    return "\n".join(mappings)

# ---------------- MAIN ----------------

def main():
    try:
        if not os.path.exists(TRANSCRIPT_OUTPUT_DIR): os.makedirs(TRANSCRIPT_OUTPUT_DIR)

        # 1. Download & ASR
        audio_path = download_audio_from_youtube(YOUTUBE_URL, DOWNLOAD_DIR)
        audio = AudioSegment.from_file(audio_path)
        print(f"\n🎧 Audio: {len(audio)/1000.0:.1f}s")

        chunk_ms = CHUNK_DURATION_SEC * 1000
        chunks = []
        for i, s in enumerate(range(0, len(audio), chunk_ms - OVERLAP_DURATION_SEC*1000)):
            buf = io.BytesIO()
            audio[s:min(s+chunk_ms, len(audio))].export(buf, format="wav")
            chunks.append({"data": buf.getvalue(), "index": i})

        print(f"🚀 Transcribing {len(chunks)} chunks...")
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(transcribe_chunk_safe, c["data"], c["index"], ASR_VOCAB_PROMPT): c["index"] for c in chunks}
            for f in concurrent.futures.as_completed(futures): results[futures[f]] = f.result()
        
        raw_text = merge_transcriptions_fuzzy_overlap([results[i] for i in sorted(results.keys())])
        with open(os.path.join(TRANSCRIPT_OUTPUT_DIR, RAW_TRANSCRIPT_FILE), "w", encoding="utf-8") as f: f.write(raw_text)

        # 2. Cleaning
        print("\n🧠 [Step 1] Cleaning Format...")
        time.sleep(2)
        clean_text = cleaning_chain.invoke({"raw_text": raw_text})

        # 3. Correction
        print("\n🧠 [Step 2] Repairing ASR with Deep Logic...")
        time.sleep(5)
        corrected_text = correction_chain.invoke({"clean_text": clean_text})
        with open(os.path.join(TRANSCRIPT_OUTPUT_DIR, CLEAN_TRANSCRIPT_FILE), "w", encoding="utf-8") as f:
            f.write(corrected_text)

        # 4. NER & Verify
        mapping_str = run_ner_search_verify_agent(corrected_text)

        # 5. Summary
        print("\n🧠 [Step 4] Summarizing...")
        time.sleep(5)
        draft = summary_chain.invoke({"corrected_text": corrected_text, "mapping_str": mapping_str})

        # 6. Audit
        print("\n🧠 [Step 5] Auditing...")
        time.sleep(5)
        final_md = audit_chain.invoke({"transcript": corrected_text, "markdown": draft})

        with open(os.path.join(TRANSCRIPT_OUTPUT_DIR, MARKDOWN_FILE), "w", encoding="utf-8") as f:
            f.write(final_md)

        print(f"\n✅ Done! Saved to {TRANSCRIPT_OUTPUT_DIR}")
        
        print("\n" + "="*20 + " FINAL TRANSCRIPT (CORRECTED) " + "="*20 + "\n")
        print(corrected_text)
        print("\n" + "="*22 + " END TRANSCRIPT " + "="*22 + "\n")

        print("\n" + "="*20 + " FINAL SUMMARY (CLEAN FORMAT) " + "="*20 + "\n")
        print(final_md)
        print("\n" + "="*25 + " END SUMMARY " + "="*25 + "\n")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()