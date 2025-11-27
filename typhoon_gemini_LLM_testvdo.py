import os
import io
import time
import random
import difflib
import re
import httpx
import json
from openai import OpenAI, RateLimitError, APITimeoutError
from pydub import AudioSegment
import concurrent.futures

# NEW: tools สำหรับ Agent
from duckduckgo_search import DDGS
import yfinance as yf

# ---------------- CONFIGURATION ----------------

# 1. Typhoon ASR
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_API_KEY = ""

# 2. Google Gemini (LLM สำหรับ NER + post-correction)
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GOOGLE_API_KEY = ""
LLM_MODEL_NAME = "gemini-2.0-flash"

# 3. Input / Output
LOCAL_AUDIO_FILE = "soundtest1/วิเคราะห์หุ้นรายวัน_03112568.mp3"

# 📌 vocab สำหรับ ASR อย่างเดียว (ชุดเล็ก ๆ)
ASR_VOCAB_FILE = "asr_vocab_data.json"   # ใช้สร้าง prompt ให้ ASR

# 4. Chunk Settings
CHUNK_DURATION_SEC = 45
OVERLAP_DURATION_SEC = 15
CHUNK_DURATION_MS = CHUNK_DURATION_SEC * 1000
OVERLAP_DURATION_MS = OVERLAP_DURATION_SEC * 1000

CACHE_DIR = "yt_cache"
TRANSCRIPT_OUTPUT_DIR = "transcripts_output"
TRANSCRIPT_PREFIX = "final_prod_agent"
MAX_WORKERS = 5

# ---------------- HELPER FUNCTIONS (DATA LOADING) ----------------

def load_domain_knowledge(filepath):
    """
    โหลดข้อมูล Knowledge Base จากไฟล์ JSON สำหรับ ASR context เท่านั้น
    structure:
    {
        "investment_prompt": "...",   # optional
        "vocab_list": [
            {"term": "...", "desc": "...", "hints": [...], "logic": "..."},
            ...
        ]
    }
    """
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: ไม่พบไฟล์ {filepath} ใช้ฐานข้อมูลว่างแทน")
        return {"investment_prompt": "", "vocab_list": []}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            vocab_len = len(data.get("vocab_list", []))
            print(f"✅ Loaded ASR vocab from {filepath}: {vocab_len} items.")
            if data.get("investment_prompt"):
                print(f"✅ Found investment_prompt in {filepath}")
            return data
    except json.JSONDecodeError as e:
        print(f"❌ Error: ไฟล์ {filepath} รูปแบบ JSON ไม่ถูกต้อง ({e})")
        return {"investment_prompt": "", "vocab_list": []}
    except Exception as e:
        print(f"❌ Error loading vocab file {filepath}: {e}")
        return {"investment_prompt": "", "vocab_list": []}

# 4. LOAD KNOWLEDGE BASE (เฉพาะ ASR)
ASR_KNOWLEDGE = load_domain_knowledge(ASR_VOCAB_FILE)

# ---------------- INIT CLIENTS ----------------

try:
    http_client = httpx.Client(timeout=120.0)
    asr_client = OpenAI(
        base_url=TYPHOON_BASE_URL,
        api_key=TYPHOON_API_KEY,
        http_client=http_client
    )
    llm_client = OpenAI(
        base_url=GOOGLE_BASE_URL,
        api_key=GOOGLE_API_KEY,
        http_client=http_client
    )
except Exception as e:
    print(f"❌ Error initializing Clients: {e}")
    raise SystemExit

# ---------------- ASR CONTEXT BIASING ----------------

def build_asr_prompt_from_kb(kb, extra_context=None):
    """
    ใช้ ASR vocab (จาก asr_vocab_data.json) มาสร้าง prompt สำหรับ ASR:
    - ดึง investment_prompt ถ้ามี
    - ดึง term และ hints มารวมเป็น keyword
    ควรใช้ vocab ชุดเล็ก (หุ้นยอดฮิต + ศัพท์หลัก) เพื่อไม่ให้ prompt บวมเกินไป
    """
    base = kb.get("investment_prompt", "") or ""
    terms = []
    for item in kb.get("vocab_list", []):
        term = item.get("term")
        if term:
            terms.append(term)
        for h in item.get("hints", []) or []:
            terms.append(h)

    cleaned = []
    seen = set()
    for p in terms:
        p = (p or "").strip()
        if not p:
            continue
        if p not in seen:
            seen.add(p)
            cleaned.append(p)

    keywords_str = " ".join(cleaned)
    ctx = extra_context.strip() + " " if extra_context else ""
    if base.strip():
        prompt = f"{ctx}{base.strip()} {keywords_str}".strip()
    else:
        fallback_ctx = (
            "บริบท: รายการวิเคราะห์หุ้นภาษาไทย เน้นพูดถึงหุ้นไทย SET Index ดอกเบี้ย Fed "
        )
        prompt = f"{fallback_ctx} {keywords_str}".strip()

    print(f"🧩 ASR prompt length: {len(prompt)} characters")
    return prompt

# ใช้ ASR_KNOWLEDGE สร้าง prompt สำหรับ Typhoon
INVESTMENT_PROMPT = build_asr_prompt_from_kb(
    ASR_KNOWLEDGE,
    extra_context="บริบท: รายการวิเคราะห์หุ้นรายวัน เน้นมุมมองการลงทุนระยะสั้นและกลาง"
)

# ---------------- CORE FUNCTIONS ----------------

def get_unique_output_path(prefix, directory, extension=".txt"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    i = 1
    while True:
        filename = f"{prefix}_{i:02d}{extension}"
        full_path = os.path.join(directory, filename)
        if not os.path.exists(full_path):
            return full_path
        i += 1

def transcribe_chunk_safe(chunk_data, chunk_index, prompt, max_retries=5):
    print(f"   ▶️  [Chunk {chunk_index:02d}] Transcribing...")
    retries = 0
    while retries < max_retries:
        try:
            file_like = io.BytesIO(chunk_data)
            file_like.name = f"chunk_{chunk_index}.wav"
            response = asr_client.audio.transcriptions.create(
                model="typhoon-asr-realtime",
                file=file_like,
                language="th",
                prompt=prompt,
            )
            print(f"   ✅ [Chunk {chunk_index:02d}] Done.")
            return response.text
        except (RateLimitError, APITimeoutError):
            retries += 1
            wait_time = (2 * (2 ** retries)) + random.random()
            print(f"   ⏳ [Chunk {chunk_index:02d}] Retry {retries}/{max_retries} in {wait_time:.1f}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"   ❌ [Chunk {chunk_index:02d}] Error: {e}")
            return ""
    return ""

def merge_transcriptions_fuzzy_overlap(transcripts):
    if not transcripts:
        return ""
    final_text = transcripts[0].strip()
    for i in range(1, len(transcripts)):
        prev_chunk = final_text
        curr_chunk = transcripts[i].strip()
        if not curr_chunk:
            continue

        check_len = 400
        prev_suffix = prev_chunk[-check_len:] if len(prev_chunk) > check_len else prev_chunk
        search_range = min(len(curr_chunk), check_len)

        matcher = difflib.SequenceMatcher(None, prev_suffix, curr_chunk[:search_range])
        match = matcher.find_longest_match(0, len(prev_suffix), 0, search_range)

        if match.size > 15:
            trim_idx = match.b + match.size
            text_to_append = curr_chunk[trim_idx:]
            final_text += text_to_append
        else:
            final_text += " " + curr_chunk

    return final_text.replace("  ", " ").strip()

def clean_fillers_and_repetition(text):
    if not text:
        return ""
    fillers = [r"เอ่อ+", r"อ่า+", r"อืม+", r"อ๋อ+", r"ออ+", r"แบบว่า", r"คือแบบ"]
    for filler in fillers:
        text = re.sub(filler, "", text)

    phrases = re.split(r'[\n]+', text)
    cleaned_phrases = []

    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue

        is_duplicate = False
        lookback_count = 5
        start_check = max(0, len(cleaned_phrases) - lookback_count)
        for prev_phrase in cleaned_phrases[start_check:]:
            if difflib.SequenceMatcher(None, prev_phrase, phrase).ratio() > 0.85:
                is_duplicate = True
                break

        if not is_duplicate:
            cleaned_phrases.append(phrase)

    return "\n".join(cleaned_phrases)

# ---------------- AGENT: NER + TOOLS ----------------

def extract_financial_entities_with_llm(raw_text):
    """
    ใช้ Gemini ดึง entity ที่เกี่ยวกับการลงทุนจาก transcript
    ให้ตอบเป็น JSON list เท่านั้น เช่น:
    [
      {"mention": "ไทยคม", "type": "stock_th", "note": ""},
      {"mention": "กสิกรไทย", "type": "stock_th", "note": "bank"},
      {"mention": "SET Index", "type": "index", "note": ""}
    ]
    """
    system_prompt = (
        "คุณคือ NER agent สำหรับข้อความวิเคราะห์หุ้นภาษาไทย\n"
        "หน้าที่ของคุณคือดึง 'ชื่อที่เกี่ยวกับการลงทุน' จาก transcript ได้แก่:\n"
        "- ชื่อหุ้นไทยหรือบริษัทจดทะเบียน\n"
        "- ชื่อดัชนีตลาดหุ้น (เช่น SET Index)\n"
        "- ชื่อกองทุนหรือ ETF ถ้ามี\n"
        "ให้ตอบกลับเป็น JSON array เพียงอย่างเดียว ห้ามมีคำอธิบายอื่น\n"
        "แต่ละ object ต้องมี key: mention, type, note\n"
        "type สามารถเป็น 'stock_th', 'company', 'index', 'fund', 'other_financial'\n"
        "ถ้าไม่พบ entity ให้ตอบเป็น []"
    )

    user_prompt = f"""
Transcript (ภาษาไทย):

\"\"\"{raw_text}\"\"\"

ให้ออกผลเป็น JSON ตามรูปแบบที่กำหนดเท่านั้น
"""

    try:
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=1024
        )
        content = resp.choices[0].message.content.strip()
        # พยายามดึง JSON ออกจาก text
        # เผื่อกรณี model ใส่ ```json ... ```
        json_str = content
        if "```" in content:
            # ดึงเฉพาะส่วนใน code block
            m = re.search(r"```(?:json)?(.*?)```", content, re.S)
            if m:
                json_str = m.group(1).strip()
        entities = json.loads(json_str)
        if not isinstance(entities, list):
            print("⚠️ NER output is not a list, fallback []")
            return []
        print(f"🧩 NER Agent found {len(entities)} entities")
        return entities
    except Exception as e:
        print(f"❌ NER Agent Error: {e}")
        return []

def guess_ticker_from_ddg(name_th, max_results=5):
    """
    ใช้ DuckDuckGo ช่วยเดา ticker ขึ้นต้น (เช่น KBANK.BK)
    เป็น heuristic ง่าย ๆ พอใช้เป็น context ให้ LLM ได้
    """
    query = f"{name_th} หุ้น ไทย"
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    ticker_candidates = []
    pattern = re.compile(r"\b([A-Z]{2,6}\.BK)\b")
    for r in results:
        text = " ".join([
            str(r.get("title", "")),
            str(r.get("body", "")),
            str(r.get("href", "")),
        ])
        for m in pattern.findall(text):
            ticker_candidates.append(m)

    ticker_candidates = list(dict.fromkeys(ticker_candidates))  # dedupe แต่รักษาลำดับ
    return ticker_candidates[0] if ticker_candidates else None

def validate_ticker_with_yfinance(ticker):
    """
    เช็กว่าตัว ticker พอใช้ได้ไหม ด้วย yfinance แบบเบา ๆ
    """
    if not ticker:
        return False
    try:
        t = yf.Ticker(ticker)
        _ = t.fast_info  # ถ้า call แล้วไม่พัง แปลว่าใช้ได้ระดับหนึ่ง
        return True
    except Exception:
        return False

def enrich_entities_with_tools(entities):
    """
    สำหรับ entity ที่เป็นหุ้น/บริษัท ลองใช้ DuckDuckGo + yfinance หาข้อมูลเพิ่ม
    return เป็น list ของ mapping ที่พร้อมใช้เป็น context ให้ LLM
    เช่น:
    [
      {"mention": "ไทยคม", "ticker": "THCOM.BK", "name_en": "THAICOM PCL", "source": "ddg+yf"},
      ...
    ]
    """
    enriched = []
    for e in entities:
        e_type = e.get("type")
        mention = e.get("mention")
        if not mention:
            continue

        if e_type not in ["stock_th", "company"]:
            # ตอนนี้ enrich เฉพาะหุ้น/บริษัท
            enriched.append({
                "mention": mention,
                "type": e_type,
                "ticker": None,
                "name_en": None,
                "source": "ner_only",
            })
            continue

        print(f"🔍 Enriching entity: {mention} ({e_type})")
        ticker = guess_ticker_from_ddg(mention)
        if ticker and validate_ticker_with_yfinance(ticker):
            name_en = None
            try:
                t = yf.Ticker(ticker)
                info = getattr(t, "fast_info", None)
                # บางทีจะมี shortName หรือ longName ใน .info แต่อันนี้อาจช้า
                # ถ้าต้องการชื่อ EN จริง ๆ ค่อยขยายทีหลัง
            except Exception:
                info = None
            enriched.append({
                "mention": mention,
                "type": e_type,
                "ticker": ticker,
                "name_en": name_en,
                "source": "ddg+yf",
            })
        else:
            enriched.append({
                "mention": mention,
                "type": e_type,
                "ticker": None,
                "name_en": None,
                "source": "ddg_only_or_failed",
            })

    print(f"✅ Enriched {len(enriched)} entities")
    return enriched

def build_entity_context_for_llm(enriched_entities):
    """
    แปลง mapping ที่ enrich แล้วให้เป็น context เบา ๆ ส่งเข้า Gemini
    """
    if not enriched_entities:
        return "ไม่มี mapping ชื่อหุ้นจาก agent/tools สำหรับ transcript นี้"

    lines = ["ENTITY MAPPING จาก Agent + Tools (ใช้ช่วยเขียนชื่อหุ้นให้ถูก):"]
    for e in enriched_entities:
        mention = e.get("mention")
        etype = e.get("type")
        ticker = e.get("ticker") or "UNKNOWN"
        src = e.get("source")
        line = f"- mention: {mention} | type: {etype} | ticker: {ticker} | source: {src}"
        lines.append(line)
    return "\n".join(lines)

# ---------------- LLM CORRECTION (ใช้ Agent context) ----------------

def correct_transcript_with_llm(raw_text, enriched_entities):
    print(f"\n🧠 Sending to Gemini ({LLM_MODEL_NAME}) for Logical Reconstruction with Agent Context...")

    # 1) ลบ trigger ที่ชวนให้ model เพ้อ
    hallucination_triggers = ["Subtitles by", "Amara.org", "Unidentified speaker"]
    for trigger in hallucination_triggers:
        raw_text = raw_text.replace(trigger, "")

    # 2) เคลียร์ filler + ประโยคซ้ำก่อน
    raw_text = clean_fillers_and_repetition(raw_text)

    # 3) แปลง entity mapping เป็น context เบา ๆ
    entity_ctx = build_entity_context_for_llm(enriched_entities)

    system_prompt = (
        "คุณคือ AI Financial Reconstruction Engine สำหรับงานวิเคราะห์หุ้นและการลงทุน\n"
        "ทุกข้อความที่คุณได้รับคือบทพูดเกี่ยวกับการลงทุน หุ้น เศรษฐกิจ และตลาดการเงิน\n"
        "คุณจะได้รับทั้ง transcript ดิบจาก ASR และ ENTITY MAPPING ที่มาจาก Agent + Tools\n"
        "ให้ใช้ ENTITY MAPPING เฉพาะเพื่อช่วยเขียนชื่อหุ้น/บริษัทให้ถูกต้อง และช่วยเลือก ticker ให้เหมาะสม\n"
        "ห้ามใช้ ENTITY MAPPING เพื่อเดาตัวเลขใหม่ หรือสร้างคำแนะนำลงทุนเพิ่มจากข้อมูลภายนอก\n"
        "หน้าที่ของคุณ:\n"
        "- ทำให้ข้อความอ่านรู้เรื่อง มีตรรกะต่อเนื่อง ใช้ศัพท์การเงินถูกต้อง\n"
        "- โฟกัสที่เนื้อหาการลงทุน ตัด small talk ออก ห้ามแต่งตัวเลขหรือมุมมองใหม่ที่ไม่มีในต้นฉบับ\n"
    )

    user_prompt = f"""
นี่คือ ENTITY MAPPING ที่ได้จาก Agent + DuckDuckGo + yfinance:

{entity_ctx}


นี่คือข้อความดิบจาก ASR (อาจมีคำเพี้ยน / ปีผิด / พูดซ้ำ / คำฟุ่มเฟือย):

\"\"\"{raw_text}\"\"\"


โจทย์ของคุณ:
1. ทำให้ข้อความด้านบนอ่านรู้เรื่อง ต่อเนื่อง และมีตรรกะเหมาะสมสำหรับ 'บทวิเคราะห์หุ้นและการลงทุน'
2. แก้คำเพี้ยน/คำผิดให้กลายเป็นคำศัพท์ทางการเงินที่ถูกต้อง โดยใช้บริบทในข้อความเป็นหลัก และใช้งาน ENTITY MAPPING เฉพาะเพื่อช่วยเลือกชื่อหุ้น/ ticker
3. ตรวจสอบความสมเหตุสมผลของปี / ไตรมาส / ตัวย่อทางการเงิน ตามบริบท ถ้าไม่แน่ใจให้คงไว้ตามเดิม
4. เปลี่ยนชื่อหุ้นไทยให้เป็น Ticker ตัวใหญ่ตาม mapping ถ้า ENTITY MAPPING มี ticker ให้แล้ว
5. ตัดคำอย่าง 'เอ่อ', 'อ่า', 'คือแบบ' รวมถึงประโยคที่ซ้ำซ้อนและไม่เกี่ยวกับการลงทุน
6. อย่าใส่ข้อมูลใหม่ที่ไม่มีในต้นฉบับ เช่น เป้าราคาใหม่ ตัวเลขใหม่ หรือคำแนะนำซื้อ/ขายใหม่
7. ให้ตอบเป็นข้อความฉบับสุดท้ายที่แก้ไขแล้วเท่านั้น ไม่ต้องใส่คำอธิบายเพิ่ม
"""

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.15,
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        # fallback เป็น raw_text ที่ล้าง filler แล้ว
        return raw_text

# ---------------- MAIN EXECUTION ----------------

def main():
    try:
        if not os.path.exists(LOCAL_AUDIO_FILE):
            print(f"❌ File not found: {LOCAL_AUDIO_FILE}")
            return

        print(f"\n🎧 Processing: {LOCAL_AUDIO_FILE}")
        audio = AudioSegment.from_file(LOCAL_AUDIO_FILE)

        duration_sec = len(audio) / 1000
        print(f"** Duration: {int(duration_sec//3600):02d}:{int((duration_sec%3600)//60):02d}:{duration_sec%60:05.2f}")

        print(f"📦 Chunking (Chunk={CHUNK_DURATION_SEC}s, Overlap={OVERLAP_DURATION_SEC}s)...")
        chunks = []
        start = 0
        idx = 0
        while start < len(audio):
            end = min(start + CHUNK_DURATION_MS, len(audio))
            chunk = audio[start:end]
            buf = io.BytesIO()
            chunk.export(buf, format="wav")
            chunks.append({'data': buf.getvalue(), 'index': idx})
            if end == len(audio):
                break
            start += (CHUNK_DURATION_MS - OVERLAP_DURATION_MS)
            idx += 1
        print(f"✅ Created {len(chunks)} chunks.")

        print(f"🚀 Starting Transcription with ASR prompt biasing...")
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(
                    transcribe_chunk_safe,
                    c['data'],
                    c['index'],
                    INVESTMENT_PROMPT
                ): c['index']
                for c in chunks
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = ""

        print("🔄 Merging text...")
        sorted_text = [results[i] for i in sorted(results.keys())]
        raw_transcript = merge_transcriptions_fuzzy_overlap(sorted_text)

        print("\n🧹 Raw transcript after merge (before Agent/LLM):")
        print("-" * 40)
        print(raw_transcript[:1000], "..." if len(raw_transcript) > 1000 else "")
        print("-" * 40)

        # STEP ใหม่: Agent NER + Tools
        entities = extract_financial_entities_with_llm(raw_transcript)
        enriched_entities = enrich_entities_with_tools(entities)

        # STEP สุดท้าย: LLM correction โดยใช้ entity mapping แทน vocab JSON ยักษ์
        final_output = correct_transcript_with_llm(raw_transcript, enriched_entities)

        print("\n" + "=" * 40)
        print("📄 --- FINAL TRANSCRIPTION RESULT ---")
        print("=" * 40)
        print(final_output)
        print("=" * 40 + "\n")

        out_path = get_unique_output_path(TRANSCRIPT_PREFIX, TRANSCRIPT_OUTPUT_DIR)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_output)
        print(f"✅ Saved result to: {out_path}")

    except Exception as e:
        print(f"❌ Main Error: {e}")

if __name__ == "__main__":
    main()
