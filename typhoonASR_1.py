import os
import io
import time
import yt_dlp 
from openai import OpenAI
from pydub import AudioSegment

# ---------------- CONFIG ----------------

BASE_URL = "https://api.opentyphoon.ai/v1"
API_KEY = ""

LOCAL_AUDIO_FILE = "soundtest2/วิเคราะห์หุ้นรายวัน_06_10_2568.mp3"
YOUTUBE_URL = "https://www.youtube.com/watch?v=ET8TDclC2O0" 
USE_YOUTUBE = False

# Context Prompt สำหรับการลงทุน*
INVESTMENT_PROMPT = (
    "วิเคราะห์แนวโน้มตลาดหุ้น, เศรษฐกิจ, การลงทุน, SET Index, "
    "Fed, Jerome Powell, อัตราดอกเบี้ย, เงินเฟ้อ, GDP, Recession, "
    "หุ้นกลุ่ม Tech, Magnificent Seven, Google, Sundar Pichai, Microsoft, Nvidia, Apple, Tesla, "
    "กองทุนรวม, ETF, RMF, SSF, หุ้นกู้, พันธบัตร, "
    "Portfolio, Valuation, Yield, Dividend, Technical Analysis, "
    "Pi Securities, InnovestX, KTB, SCB, กราบสวัสดี, Thaicom, พรีวิว, QoQ, Assumption, Q, Year, Q&Q, USO, เฟส2, budget, up, นะครับ, querter, ผลประกอบการหลัก, ยังไม่ฟื้น, TOR, เฟส3, เก็ง, ไทยคม, assume, อาจ, จะ, ต้อง, ดู, จับจังหวะ"
)

# ตั้งค่า Chunking (30 วิ / ซ้อน 5 วิ)**
CHUNK_DURATION_SEC = 30 
OVERLAP_DURATION_SEC = 5 

CHUNK_DURATION_MS = CHUNK_DURATION_SEC * 1000
OVERLAP_DURATION_MS = OVERLAP_DURATION_SEC * 1000

CACHE_DIR = "yt_cache"

TRANSCRIPT_OUTPUT_DIR = "transcripts_output"
TRANSCRIPT_PREFIX = "คลิปยาว_191125"


# ---------------- INIT CLIENT ----------------

try:
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
except Exception as e:
    print(f"❌ Error initializing OpenAI/Typhoon client: {e}")
    raise SystemExit


# ---------------- CORE FUNCTIONS ----------------

def get_unique_output_path(prefix: str, directory: str, extension: str = ".txt") -> str:
    """สร้างชื่อไฟล์ที่ไม่ซ้ำกันด้วยการรันหมายเลขอัตโนมัติ (เช่น prefix_01.txt)"""
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    i = 1
    while True:
        filename = f"{prefix}_{i:02d}{extension}" 
        full_path = os.path.join(directory, filename)
        
        if not os.path.exists(full_path):
            return full_path
        
        i += 1

def download_youtube_audio(url, cache_dir) -> str | None:
    """ดาวน์โหลดเสียงจาก YouTube โดยใช้ yt-dlp พร้อม Caching"""
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    ydl_opts = {
        'format': 'bestaudio/best', 
        'extract_audio': True,
        'audioformat': 'mp3',
        'outtmpl': os.path.join(cache_dir, '%(id)s.%(ext)s'), 
        'noplaylist': True,
        'quiet': True,
        'cachedir': False,
    }
    
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info.get('id', 'temp_file')
            final_output_path = os.path.join(cache_dir, f"{video_id}.mp3")
            
        if os.path.exists(final_output_path):
            print(f"✅ พบไฟล์ใน Cache: ใช้ไฟล์ {final_output_path} ที่ดาวน์โหลดไว้แล้ว")
            return final_output_path
            
        print(f"🔗 กำลังดาวน์โหลดเสียงจาก: {info.get('title', 'YouTube Video')} ไปที่ Cache...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        time.sleep(2) 
        
        print(f"✅ ดาวน์โหลดสำเร็จ บันทึกที่: {final_output_path}")
        return final_output_path
            
    except Exception as e:
        print(f"❌ Error downloading YouTube audio with yt-dlp: {e}")
        return None

def transcribe_chunk(chunk_data: bytes, chunk_index: int, prompt: str) -> str:
    """ส่ง Chunk ไฟล์เสียงไปที่ Typhoon ASR API พร้อม Context Prompt"""
    try:
        file_like_object = io.BytesIO(chunk_data)
        file_like_object.name = f"chunk_{chunk_index}.wav" 

        response = client.audio.transcriptions.create(
            model="typhoon-asr-realtime",
            file=file_like_object,
            language="th",
            prompt=prompt,
        )
        return response.text
    except Exception as e:
        print(f"❌ Error Transcribing Chunk {chunk_index}: {e}")
        print(f"   (ขนาด Chunk: {len(chunk_data) / (1024*1024):.2f} MB - อาจเกินขีดจำกัด API หรือ Timeout)")
        return f"[ERROR IN CHUNK {chunk_index}]" 

def merge_transcriptions(transcripts, overlap_duration_sec):
    """รวมข้อความที่ถอดเสียงแล้ว และตัดคำซ้ำซ้อนในส่วนที่เหลื่อมซ้อนออก"""
    if not transcripts:
        return ""

    final_text = transcripts[0]
    
    for i in range(1, len(transcripts)):
        prev_text = final_text
        current_text = transcripts[i].strip()
        
        words_in_current = current_text.split()
        
        overlap_found = False
        # ขยายช่วงตรวจสอบเป็น 30 คำ
        for j in range(min(30, len(words_in_current)), 0, -1):
            overlap_candidate = " ".join(words_in_current[:j])
            
            if prev_text.strip().endswith(overlap_candidate):
                final_text += " " + " ".join(words_in_current[j:])
                overlap_found = True
                break
        
        if not overlap_found:
             final_text += " " + current_text

    return final_text.strip().replace("  ", " ").replace(". .", ".")


# ---------------- MAIN EXECUTION ----------------

try:
    chunk_transcripts = []
    
    # 1. กำหนด Input Path
    audio_path_to_use = LOCAL_AUDIO_FILE
    
    if USE_YOUTUBE:
        audio_path_to_use = download_youtube_audio(YOUTUBE_URL, CACHE_DIR)
        
        if not audio_path_to_use: 
            raise SystemExit 

    if not os.path.exists(audio_path_to_use):
        print(f"❌ Error: หาไฟล์ไม่เจอที่ {audio_path_to_use}")
        raise SystemExit

    # 2. โหลดและตรวจสอบไฟล์เสียง
    print(f"\n🎧 กำลังโหลดไฟล์เสียง: {audio_path_to_use}")
    audio = AudioSegment.from_file(audio_path_to_use) 
    
    duration_sec = len(audio) / 1000
    file_size_bytes = os.path.getsize(audio_path_to_use)
    
    hours = int(duration_sec // 3600)
    minutes = int((duration_sec % 3600) // 60)
    remaining_seconds = duration_sec % 60
    
    print(f"** ขนาดไฟล์: {file_size_bytes / (1024*1024):.2f} MB")
    # HH:MM:SS.ms (ใช้ :02d สำหรับชั่วโมง/นาที และ :05.2f สำหรับวินาทีพร้อมทศนิยม 2 ตำแหน่ง)
    print(f"** ความยาวไฟล์: {hours:02d}:{minutes:02d}:{remaining_seconds:05.2f}") 
    
    # 3. เริ่มกระบวนการ Chunking
    print(f"⚠️ เริ่มใช้ Overlapping Chunking: {CHUNK_DURATION_SEC}s / Overlap {OVERLAP_DURATION_SEC}s")
    
    start_time = 0
    chunk_index = 0
    
    while start_time < len(audio):
        end_time = min(start_time + CHUNK_DURATION_MS, len(audio))
        chunk = audio[start_time:end_time]

        buffer = io.BytesIO()
        chunk.export(buffer, format="wav") 
        
        print(f"   - Transcribing Chunk {chunk_index}: {start_time/1000:.1f}s - {end_time/1000:.1f}s")
        
        chunk_transcript = transcribe_chunk(buffer.getvalue(), chunk_index, INVESTMENT_PROMPT)
        chunk_transcripts.append(chunk_transcript)
        
        if end_time == len(audio):
            break 
        
        start_time = end_time - OVERLAP_DURATION_MS 
        chunk_index += 1
        time.sleep(1) 

    # 4. รวมผลลัพธ์สุดท้ายและจัดการการซ้ำซ้อน
    final_transcript = merge_transcriptions(chunk_transcripts, OVERLAP_DURATION_SEC)
    
    print("\n--- Final Consolidated Transcription (รวมผลลัพธ์) ---")
    print(final_transcript)
    print("-----------------------------------------------------")

    # 5. 🎯 บันทึกผลลัพธ์ลงไฟล์ด้วยชื่อที่ไม่ซ้ำกัน
    output_path = get_unique_output_path(TRANSCRIPT_PREFIX, TRANSCRIPT_OUTPUT_DIR)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_transcript)
    
    print(f"\n✅ บันทึกผลลัพธ์ที่ถอดเสียงเรียบร้อยแล้วที่: {output_path}")

except Exception as e:
    print(f"❌ Error in main process: {e}")
