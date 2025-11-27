import os
import io
import time
import random
import difflib
import re
import httpx
from openai import OpenAI, RateLimitError, APITimeoutError
from pydub import AudioSegment
import concurrent.futures

# ---------------- CONFIGURATION ----------------

# 1. Typhoon ASR (ถอดเสียง - เก่งไทยสุด)
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_API_KEY = "sk-vCE2QnUydpGnzic35kI3IcoTsAeWzb2X3jYCCAXDPmfT2JnN"

# 2. Google Gemini (แก้คำผิด - ฉลาดและฟรี)
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GOOGLE_API_KEY = "AIzaSyCkjWUucxzLaRPnuklKxKgYP0fUyQhTwHA"
LLM_MODEL_NAME = "gemini-2.0-flash" 

LOCAL_AUDIO_FILE = "soundtest2/วิเคราะห์หุ้นรายวัน_06_10_2568.mp3"

# 3. Context Biasing (สำหรับ ASR)
INVESTMENT_PROMPT = (
    "รายการวิเคราะห์หุ้นวันนี้เจาะลึกประเด็น GUNKUL, THAICOM, Nvidia และ Tesla "
    "ติดตามดัชนี SET Index, Fed rate, และตัวเลขเศรษฐกิจ QoQ, YoY "
    "อัปเดตโครงการ USO Phase, การประมูล TOR, Budget และการ Ramp up ของ Yuanta"
)

# 4. Vocab List (สำหรับ Reference)
CORRECTION_VOCAB = """
- THAICOM (บริษัท ไทยคม - Ticker: THAICOM)
- GUNKUL (บริษัท กันกุล - Ticker: GUNKUL)
- Assumption (สมมติฐาน)
- Quarter (ไตรมาส)
- Valuation (มูลค่า)
- SET Index
- Fed (เฟด)
- Yield (ผลตอบแทน)
- QoQ (Quarter on Quarter - ห้ามตัดทิ้ง)
- YoY (Year on Year - ห้ามตัดทิ้ง)
- USO Phase (โครงการยูโซ เฟส - เน็ตชายขอบ)
- USO Phase 2 (เฟส 2 - บริบทคือ: ของเข้าแล้ว/รับรู้รายได้/Budget/Ramp up)
- USO Phase 3 (เฟส 3 - บริบทคือ: อนาคต/รอประมูล/TOR/Reaction/มูลค่าสูง)
- TOR (ที-โอ-อาร์ / ขอบเขตงาน)
- Budget (งบประมาณ)
- Ramp up (แรมป์อัพ / การเพิ่มกำลังการผลิต)
- Yuanta (บล.หยวนต้า)
- Preview (พรีวิว)
"""

# Chunk Settings (45s/15s Optimized)
CHUNK_DURATION_SEC = 45       
OVERLAP_DURATION_SEC = 15     
CHUNK_DURATION_MS = CHUNK_DURATION_SEC * 1000
OVERLAP_DURATION_MS = OVERLAP_DURATION_SEC * 1000

CACHE_DIR = "yt_cache"
TRANSCRIPT_OUTPUT_DIR = "transcripts_output"
TRANSCRIPT_PREFIX = "final_hybrid_transcript" 
MAX_WORKERS = 5

# ---------------- INIT CLIENTS ----------------

try:
    http_client = httpx.Client(timeout=120.0)
    
    # Client 1: Typhoon (ASR Only)
    asr_client = OpenAI(
        base_url=TYPHOON_BASE_URL,
        api_key=TYPHOON_API_KEY,
        http_client=http_client,
    )
    
    # Client 2: Gemini (LLM Only)
    llm_client = OpenAI(
        base_url=GOOGLE_BASE_URL,
        api_key=GOOGLE_API_KEY,
        http_client=http_client,
    )
    
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
            print(f"   ⚠️ [Chunk {chunk_index:02d}] Retry in {wait_time:.2f}s...")
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

def clean_repetitive_text(text):
    """
    Python Cleaner: ลบประโยคซ้ำซ้อน (Lookback 3 lines)
    """
    if not text: return ""
    phrases = re.split(r'[\n]+', text) 
    cleaned_phrases = []
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase: continue
        is_duplicate = False
        lookback_count = 3
        start_check = max(0, len(cleaned_phrases) - lookback_count)
        for prev_phrase in cleaned_phrases[start_check:]:
            if difflib.SequenceMatcher(None, prev_phrase, phrase).ratio() > 0.85:
                is_duplicate = True
                break
        if not is_duplicate:
            cleaned_phrases.append(phrase)
    return "\n".join(cleaned_phrases)

def correct_transcript_with_llm(raw_text, vocab_list):
    print(f"\n🧠 Sending to Gemini ({LLM_MODEL_NAME}) for Logic-Based Correction...")
    
    # Pre-cleaning
    hallucination_triggers = [
        "Subtitles by", "Amara.org", "Thank you for watching", 
        "Unidentified speaker", "บรรยายโดย", "ขอบคุณที่รับชม", "ซับไตเติ้ลโดย"
    ]
    for trigger in hallucination_triggers:
        raw_text = raw_text.replace(trigger, "")
    
    raw_text = clean_repetitive_text(raw_text)
    
    # 🔥 SYSTEM PROMPT: ใช้ Logic แทน Rules 🔥
    system_prompt = (
        "คุณคือ 'ผู้เชี่ยวชาญด้านข้อมูลตลาดทุน' (Capital Market Specialist) "
        "หน้าที่ของคุณคือตรวจสอบความถูกต้องของบทวิเคราะห์หุ้น (Transcript Verification) โดยใช้ 'วิจารณญาณ' ในการแก้ไข "
        "หลักการทำงาน:"
        "1. **Context Awareness:** คุณต้องอ่านบริบทก่อนแก้เสมอ ห้ามแทนที่คำแบบหุ่นยนต์ (Blind Replace)"
        "2. **Entity Recognition:** แยกแยะให้ออกว่าคำไหนคือ 'ชื่อบริษัท' (Company) คำไหนคือ 'คำทั่วไป' (Common Noun) หรือ 'ชื่อประเทศ' (Country)"
        "3. **Verbatim Integrity:** รักษาสำนวนและเนื้อหาเดิมไว้ให้มากที่สุด ห้ามตัดทอนตัวเลข QoQ/YoY"
    )
    
    user_prompt = f"""
    นี่คือข้อความดิบ (Raw Transcript):
    \"\"\"{raw_text}\"\"\"
    
    คำสั่งแก้ไข (ใช้ Logic ไม่ใช่แค่กฎ):
    
    1. **มาตรฐานชื่อหุ้น (Ticker Standardization):**
       - ให้เปลี่ยนชื่อหุ้นภาษาไทยเป็น **Ticker ภาษาอังกฤษตัวพิมพ์ใหญ่** (เช่น THAICOM, GUNKUL, ADVANC)
       - ⚠️ **สำคัญ:** เปลี่ยนเฉพาะเมื่อคำนั้นหมายถึง **"บริษัท"** หรือ **"หุ้น"** เท่านั้น 
       - *ตัวอย่าง:* "ประเทศไทย" (ห้ามเปลี่ยน), "คนไทย" (ห้ามเปลี่ยน), "หุ้นไทยคม" -> "หุ้น THAICOM" (เปลี่ยนได้)
    
    2. **ศัพท์เทคนิคและการเงิน:**
       - ตรวจสอบคำผิดทางเทคนิค (เช่น Gooncull -> QoQ, บัตรเจ็ด -> Budget) โดยดูจากบริบทแวดล้อม
       - หากเจอ "User Facebook" หรือเสียงคล้ายๆ ให้พิจารณาบริบท:
         * ถ้าพูดถึงรายได้/การดำเนินงาน -> แก้เป็น **USO Phase 2**
         * ถ้าพูดถึงการประมูล/TOR/อนาคต -> แก้เป็น **USO Phase 3**
         * ถ้าพูดถึง หยวนต้า/หลอดใต้ -> แก้เป็น **หยวนต้า (Yuanta)**
         
    3. **ความครบถ้วน:**
       - ตรวจสอบว่า QoQ และ YoY อยู่ครบถ้วนตามต้นฉบับ
       
    4. **การจัดรูปแบบ:**
       - จัดย่อหน้าให้อ่านง่าย (เว้นบรรทัดเมื่อจบประเด็น) และตัดประโยคซ้ำซ้อนช่วงท้ายออก
    
    ข้อมูลอ้างอิง (Vocabulary List):
    {vocab_list}
    
    Output: ขอข้อความที่แก้ไขเสร็จสมบูรณ์
    """
    
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, # ต่ำสุดเพื่อให้ AI ตัดสินใจด้วย Logic ที่แม่นยำ ไม่มั่ว
            max_tokens=4096 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return raw_text

# ---------------- MAIN EXECUTION ----------------

def main():
    try:
        # 1. Check File
        if not os.path.exists(LOCAL_AUDIO_FILE):
            print(f"❌ File not found: {LOCAL_AUDIO_FILE}")
            return

        print(f"\n🎧 Processing: {LOCAL_AUDIO_FILE}")
        audio = AudioSegment.from_file(LOCAL_AUDIO_FILE)
        
        duration_sec = len(audio) / 1000
        file_size = os.path.getsize(LOCAL_AUDIO_FILE) / (1024 * 1024)
        print(f"** Size: {file_size:.2f} MB")
        print(f"** Duration: {int(duration_sec//3600):02d}:{int((duration_sec%3600)//60):02d}:{duration_sec%60:05.2f}")

        # 2. Chunking
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

        # 3. Transcription
        print(f"🚀 Starting Transcription (Typhoon ASR)...")
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

        # 4. Merge
        print("🔄 Merging text...")
        sorted_text = [results[i] for i in sorted(results.keys())]
        raw_transcript = merge_transcriptions_fuzzy_overlap(sorted_text)
        
        # 5. Correction
        final_output = correct_transcript_with_llm(raw_transcript, CORRECTION_VOCAB)
        
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