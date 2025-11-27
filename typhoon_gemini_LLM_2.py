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

# ---------------- CONFIGURATION ----------------

# 1. Typhoon ASR
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_API_KEY = ""

# 2. Google Gemini
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GOOGLE_API_KEY = "" # <--- Key ของคุณ
LLM_MODEL_NAME = "gemini-2.0-flash" 

LOCAL_AUDIO_FILE = "soundtest2/วิเคราะห์หุ้นรายวัน_06_10_2568.mp3"

# 3. Context Biasing
INVESTMENT_PROMPT = (
    "รายการวิเคราะห์หุ้นวันนี้เจาะลึกประเด็น GUNKUL, THAICOM, Nvidia และ Tesla "
    "ติดตามดัชนี SET Index, Fed rate, และตัวเลขเศรษฐกิจ QoQ, YoY "
    "อัปเดตโครงการ USO Phase, การประมูล TOR, Budget และการ Ramp up ของ Yuanta"
)

# 4. SYSTEMATIC KNOWLEDGE BASE (ฐานข้อมูลจับคู่เสียงผิด-ถูก)
# ต้องใช้แบบนี้เท่านั้น Gemini ถึงจะแก้คำเพี้ยนหนักๆ ได้
DOMAIN_KNOWLEDGE = {
    "vocab_list": [
        # กลุ่มชื่อหุ้น
        {"term": "THCOM", "desc": "บมจ.ไทยคม (Ticker)", "hints": ["ไทยคม", "THAICOM"]},
        {"term": "GUNKUL", "desc": "บมจ.กันกุล (Ticker)", "hints": ["กันกุล", "กุนกุล", "กูนกุล"]},
        
        # กลุ่มศัพท์เทคนิค (Jargon)
        {"term": "QoQ", "desc": "Quarter on Quarter", "hints": ["Gooncull", "จีวรคิว", "คิวคิว"]},
        {"term": "YoY", "desc": "Year on Year", "hints": ["Y Y", "วายวาย"]},
        {"term": "Assumption", "desc": "สมมติฐาน", "hints": ["Assution", "Astumption", "Apsumption"]},
        {"term": "Quarter", "desc": "ไตรมาส", "hints": ["คอเตอร์", "ควอเตอร์", "คอเตอ"]},
        {"term": "TOR", "desc": "Terms of Reference", "hints": ["PR", "QR", "ER", "ทอ"]},
        {"term": "Budget", "desc": "งบประมาณ", "hints": ["บัตรเจ็ด", "บัตร", "มัดเจ็ด", "มาเจ็ด"]},
        {"term": "Ramp up", "desc": "การเร่งงาน", "hints": ["แลมา", "แลมาบ", "แรมปั๊พ"]},
        {"term": "Reaction", "desc": "ปฏิกิริยาตอบรับ", "hints": ["Reax"]},
        
        # กลุ่มคำทั่วไปที่มักผิด (Traps)
        {"term": "สวัสดี", "desc": "คำทักทาย", "hints": ["โซดี", "ซอดี"]},
        {"term": "ขอบคุณสำหรับการ", "desc": "คำขอบคุณ", "hints": ["ขอบคุณสถานการณ์"]},
        {"term": "กดปุ่ม", "desc": "Action", "hints": ["กดกลุ่ม", "กดฝุ่น"]},
        {"term": "น่าจะออก", "desc": "คาดการณ์การประกาศ", "hints": ["น่าจะนะ", "น่าจะ On"]},
        
        # กลุ่ม Logic พิเศษ
        {"term": "ว่า / วันที่", "desc": "คำเชื่อม (ไม่ใช่ Valuation)", "hints": ["Valuation (ถ้าตามด้วย 'ได้งาน')"]},
        {"term": "USO Phase", "desc": "โครงการยูโซ", "hints": ["User Facebook", "Use Face"], 
         "logic": "บริบทรายได้=Phase 2, บริบทประมูล=Phase 3"},
        {"term": "Yuanta", "desc": "บล.หยวนต้า", "hints": ["หลอดใต้", "หลวงต้า", "หลอดต้า", "โหลต้า"]}
    ]
}

# Chunk Settings
CHUNK_DURATION_SEC = 45       
OVERLAP_DURATION_SEC = 15     
CHUNK_DURATION_MS = CHUNK_DURATION_SEC * 1000
OVERLAP_DURATION_MS = OVERLAP_DURATION_SEC * 1000

CACHE_DIR = "yt_cache"
TRANSCRIPT_OUTPUT_DIR = "transcripts_output"
TRANSCRIPT_PREFIX = "final_prod_fixed" 
MAX_WORKERS = 5

# ---------------- INIT CLIENTS ----------------

try:
    http_client = httpx.Client(timeout=120.0)
    asr_client = OpenAI(base_url=TYPHOON_BASE_URL, api_key=TYPHOON_API_KEY, http_client=http_client)
    llm_client = OpenAI(base_url=GOOGLE_BASE_URL, api_key=GOOGLE_API_KEY, http_client=http_client)
except Exception as e:
    print(f"❌ Error initializing Clients: {e}")
    raise SystemExit

# ---------------- CORE FUNCTIONS ----------------

def get_unique_output_path(prefix, directory, extension=".txt"):
    if not os.path.exists(directory): os.makedirs(directory)
    i = 1
    while True:
        filename = f"{prefix}_{i:02d}{extension}"
        full_path = os.path.join(directory, filename)
        if not os.path.exists(full_path): return full_path
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
        except (RateLimitError, APITimeoutError) as e:
            retries += 1
            wait_time = (2 * (2 ** retries)) + random.random()
            time.sleep(wait_time)
        except Exception as e:
            print(f"   ❌ [Chunk {chunk_index:02d}] Error: {e}")
            return ""
    return ""

def merge_transcriptions_fuzzy_overlap(transcripts):
    if not transcripts: return ""
    final_text = transcripts[0].strip()
    for i in range(1, len(transcripts)):
        prev_chunk = final_text
        curr_chunk = transcripts[i].strip()
        if not curr_chunk: continue
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
    if not text: return ""
    fillers = [r"เอ่อ+", r"อ่า+", r"อืม+", r"อ๋อ+", r"ออ+", r"แบบว่า", r"คือแบบ"]
    for filler in fillers:
        text = re.sub(filler, "", text)
    phrases = re.split(r'[\n]+', text) 
    cleaned_phrases = []
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase: continue
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

def generate_knowledge_prompt(kb):
    """สร้าง Prompt จาก Knowledge Base"""
    prompt = "REFERENCE KNOWLEDGE BASE (Use this mapping to fix errors):\n"
    for item in kb["vocab_list"]:
        line = f"- Correct Term: **{item['term']}**"
        if item.get("hints"):
            line += f" (Fix if sounds like: {', '.join(item['hints'])})"
        if item.get("logic"):
            line += f" [Rule: {item['logic']}]"
        prompt += line + "\n"
    return prompt

def correct_transcript_with_llm(raw_text, knowledge_base):
    print(f"\n🧠 Sending to Gemini ({LLM_MODEL_NAME}) for Knowledge-Based Correction...")
    
    hallucination_triggers = ["Subtitles by", "Amara.org", "Unidentified speaker"]
    for trigger in hallucination_triggers:
        raw_text = raw_text.replace(trigger, "")
    
    raw_text = clean_fillers_and_repetition(raw_text)
    kb_prompt_str = generate_knowledge_prompt(knowledge_base)
    
    system_prompt = (
        "คุณคือ 'AI นักพิสูจน์อักษรการเงิน' หน้าที่คือแก้ไขคำผิดในบทถอดความโดยใช้ 'Knowledge Base' ที่ให้เท่านั้น "
        "1. **Strict Mapping:** หากเจอคำที่เสียงเหมือนในรายการ 'Fix if sounds like' ให้แก้เป็น 'Correct Term' ทันที (เช่น 'มัดเจ็ด' -> 'Budget')"
        "2. **Logic Check:** ตรวจสอบบริบทก่อนแก้ (เช่น Valuation vs ว่า, Phase 2 vs 3)"
        "3. **Verbatim:** ห้ามตัดทอนเนื้อหา ห้ามเปลี่ยนสำนวน"
        "4. **Format:** ไม่ทำตัวหนา, เว้นบรรทัดให้อ่านง่าย"
    )
    
    user_prompt = f"""
    ข้อความดิบ:
    \"\"\"{raw_text}\"\"\"
    
    คำสั่ง:
    1. ตรวจสอบและแก้คำผิดตาม Knowledge Base ด้านล่างอย่างเคร่งครัด
    2. เปลี่ยนชื่อหุ้นไทยเป็น Ticker (THCOM, GUNKUL)
    3. ตัดคำฟุ่มเฟือย (เอ่อ, อ่า) และประโยคซ้ำซ้อน
    
    {kb_prompt_str}
    
    Output: ขอข้อความที่แก้ไขแล้ว (Clean Text)
    """
    
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, 
            max_tokens=4096 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ LLM Error: {e}")
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
            if end == len(audio): break
            start += (CHUNK_DURATION_MS - OVERLAP_DURATION_MS) 
            idx += 1
        print(f"✅ Created {len(chunks)} chunks.")

        print(f"🚀 Starting Transcription...")
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(transcribe_chunk_safe, c['data'], c['index'], INVESTMENT_PROMPT): c['index'] 
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
        
        final_output = correct_transcript_with_llm(raw_transcript, DOMAIN_KNOWLEDGE)
        
        print("\n" + "="*40)
        print("📄 --- FINAL TRANSCRIPTION RESULT ---")
        print("="*40)
        print(final_output) 
        print("="*40 + "\n")
        
        out_path = get_unique_output_path(TRANSCRIPT_PREFIX, TRANSCRIPT_OUTPUT_DIR)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_output)
        print(f"✅ Saved result to: {out_path}")

    except Exception as e:
        print(f"❌ Main Error: {e}")

if __name__ == "__main__":
    main()
