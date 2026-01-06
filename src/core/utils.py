# src/core/utils.py
import os
import io
import re
import difflib
import time
from typing import List
from pydub import AudioSegment, effects

# ลอง Import Library ตัดคำ (ถ้ามี)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

# --- HELPER FUNCTIONS ---

def format_duration(seconds: float) -> str:
    """
    แปลงจำนวนวินาทีให้เป็นรูปแบบ 'นาทีm วินาทีs' ที่อ่านง่าย

    Args:
        seconds (float): จำนวนวินาที

    Returns:
        str: สตริงในรูปแบบ "Xm Ys"
    """
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}m {remaining_seconds}s"


def get_file_size_mb(file_path: str) -> float:
    """
    คำนวณขนาดของไฟล์ในหน่วย Megabytes (MB)

    Args:
        file_path (str): ที่อยู่ของไฟล์

    Returns:
        float: ขนาดไฟล์ (MB)
    """
    return os.path.getsize(file_path) / (1024 * 1024)


def sanitize_filename(name: str) -> str:
    """
    ทำความสะอาดชื่อไฟล์ให้ปลอดภัยสำหรับทุกระบบปฏิบัติการ

    โดยการลบอักขระพิเศษ, แทนที่ช่องว่างด้วย '_', และจำกัดความยาว

    Args:
        name (str): ชื่อไฟล์ดิบ

    Returns:
        str: ชื่อไฟล์ที่ปลอดภัย
    """
    if not name:
        return "audio_output"
    # ใช้ Regex เพื่อลบอักขระที่ไม่ใช่:
    # \w -> (a-z, A-Z, 0-9, _)
    # \u0E00-\u0E7F -> ตัวอักษรภาษาไทยทั้งหมด
    # \s- -> เว้นวรรค (space) และยัติภังค์ (hyphen)
    name = re.sub(r"[^\w\u0E00-\u0E7F\s-]", "", name)
    name = name.strip().replace(" ", "_")
    return name[:50]


def _wrap_text_simple(text: str, width: int = 120) -> str:
    """
    ตัดบรรทัดข้อความธรรมดา (Plain Text) โดยไม่สนย่อหน้าเดิม

    เป็นฟังก์ชันภายในสำหรับ `format_transcript_paragraphs`

    Args:
        text (str): ข้อความที่ต้องการตัดบรรทัด (สันนิษฐานว่าเป็นบรรทัดเดียว)
        width (int): ความยาวสูงสุดของแต่ละบรรทัด

    Returns:
        str: ข้อความที่ถูกตัดบรรทัดแล้ว
    """
    if not text:
        return text

    # รวม whitespace ต่อเนื่อง ๆ เป็นช่องว่างเดียว (ไม่ต้องเก็บ \n เดิม เพราะเราจะ wrap เอง)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")  # แยกข้อความเป็นคำๆ

    wrapped_lines: List[str] = []  # เก็บบรรทัดที่ตัดแล้ว
    line = ""  # บรรทัดปัจจุบันที่กำลังสร้าง

    # วนลูปทีละคำเพื่อสร้างบรรทัดใหม่ โดยไม่ให้เกิน width ที่กำหนด
    for word in words:
        if not line:
            # ถ้าบรรทัดว่างอยู่ ให้เริ่มด้วยคำปัจจุบัน
            line = word
        elif len(line) + 1 + len(word) > width:
            # ถ้าเพิ่มคำนี้แล้วจะยาวเกิน width (นับรวม space 1 ตัว) ให้ขึ้นบรรทัดใหม่
            wrapped_lines.append(line)
            line = word  # เริ่มบรรทัดใหม่ด้วยคำปัจจุบัน
        else:
            # ถ้ายังไม่เกิน ให้เติมคำนี้ต่อในบรรทัดเดิม (เว้นช่องว่างด้วย)
            line += " " + word

    # ใส่บรรทัดสุดท้ายที่ค้างอยู่
    if line:
        wrapped_lines.append(line)

    return "\n".join(wrapped_lines)


def format_transcript_paragraphs(text: str, width: int = 120) -> str:
    """
    จัดย่อหน้าและตัดบรรทัดข้อความ Transcript ให้อ่านง่าย

    - หากข้อความมีลักษณะเป็นก้อนเดียว (มี newline น้อย) จะทำการตัดบรรทัดใหม่ทั้งหมด
    - หากข้อความมีหลายย่อหน้าอยู่แล้ว จะเคารพย่อหน้าเดิมและทำการตัดบรรทัดภายในแต่ละย่อหน้า

    Args:
        text (str): ข้อความ Transcript ดิบ
        width (int): ความยาวสูงสุดของแต่ละบรรทัด

    Returns:
        str: ข้อความที่จัดย่อหน้าและตัดบรรทัดใหม่แล้ว
    """
    if not text:
        return text

    # ตรวจสอบจำนวนการขึ้นบรรทัดใหม่ (\n) ที่มีอยู่เดิม
    # หากมีน้อยกว่าหรือเท่ากับ 2 บรรทัด สันนิษฐานว่าเป็นข้อความก้อนเดียวที่ยังไม่ได้จัดย่อหน้า
    # จึงเรียกใช้ _wrap_text_simple เพื่อจัดใหม่ทั้งหมด
    if text.count("\n") <= 2:
        return _wrap_text_simple(text, width=width)

    # หากมี newline อยู่แล้ว ให้เคารพการแบ่งย่อหน้าเดิม และ wrap ภายในแต่ละย่อหน้า
    lines = [line.strip() for line in text.split("\n")]
    cleaned: List[str] = []

    for line in lines:
        if line == "":
            if cleaned and cleaned[-1] == "":
                continue
            cleaned.append("")
        else:
            wrapped = _wrap_text_simple(line, width=width)
            cleaned.extend(wrapped.split("\n"))

    return "\n".join(cleaned).strip()


def normalize_markdown_bullets(markdown_text: str) -> str:
    """
    แปลงสัญลักษณ์ Bullet ใน Markdown จาก '*' ให้เป็น '-' เพื่อความเป็นมาตรฐาน

    Args:
        markdown_text (str): ข้อความ Markdown

    Returns:
        str: ข้อความ Markdown ที่ใช้ '-' เป็น bullet
    """
    if not markdown_text:
        return markdown_text
    return re.sub(r"^(\s*)\* ", r"\1- ", markdown_text, flags=re.MULTILINE)


def remove_markdown_bold(text: str) -> str:
    """
    ลบเครื่องหมายตัวหนา (**) ออกจากข้อความ

    Args:
        text (str): ข้อความที่มี Markdown

    Returns:
        str: ข้อความที่ไม่มีเครื่องหมายตัวหนา
      """
    if not text:
        return text
    # แก้ไข Bug: จากเดิมที่เป็น Syntax Error ให้เป็นการใช้ re.sub ที่ถูกต้อง
    return re.sub(r"\*\*(.*?)\*\*", r"\1", text)
      
def split_text_smart(text: str, chunk_size: int) -> List[str]:
    """
    หั่นข้อความยาวๆ ออกเป็นส่วนๆ (Chunks) โดยพยายามไม่ตัดกลางประโยค

    ใช้ `RecursiveCharacterTextSplitter` จาก LangChain (ถ้ามี) เพื่อความฉลาดในการตัด
    หากไม่มี จะใช้การตัดแบบธรรมดาตามจำนวนตัวอักษรเป็น Fallback

    Args:
        text (str): ข้อความทั้งหมดที่ต้องการหั่น
        chunk_size (int): ขนาดสูงสุดของแต่ละ Chunk

    Returns:
        List[str]: ลิสต์ของข้อความที่ถูกหั่นแล้ว
    """
    # ตรวจสอบว่าสามารถ import `RecursiveCharacterTextSplitter` ได้หรือไม่
    if RecursiveCharacterTextSplitter: # วิธีที่ 1 (หลัก): ใช้ LangChain
        # ใช้วิธีตัดข้อความแบบเรียกซ้ำ (Recursive) ซึ่งจะพยายามรักษาความสมบูรณ์ของประโยค
        # โดยจะตัดตามลำดับความสำคัญ: ย่อหน้า -> ประโยค -> คำ
        splitter = RecursiveCharacterTextSplitter(
            chunk_overlap=0,  # ไม่ต้องสร้าง Overlap ที่นี่ เพราะจะจัดการใน Logic การส่งไป ASR
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size, # ใช้ขนาด chunk ที่รับเข้ามา
        )
        return splitter.split_text(text)
    else:
        # วิธีที่ 2 (สำรอง): หากไม่มี LangChain ให้ใช้วิธีตัดตามจำนวนตัวอักษรแบบตรงๆ
        return [text[start_index : start_index + chunk_size] for start_index in range(0, len(text), chunk_size)]


def merge_transcriptions(transcripts: List[str]) -> str:
    """
    รวมผลลัพธ์ Transcript จากหลายๆ Chunks เข้าด้วยกันอย่างชาญฉลาด
    โดยใช้วิธีเปรียบเทียบความคล้ายของ "ประโยคต่อประโยค" (Sentence-level Similarity)
    เพื่อค้นหาและตัดส่วนที่ทับซ้อนกัน (Overlap) ซึ่งอาจมีการใช้คำต่างกันเล็กน้อยได้
    
    IMPROVED: ใช้ window ใหญ่ขึ้น และเช็คหลายประโยคแทนที่จะเช็คแค่ประโยคสุดท้าย

    Args:
        transcripts (List[str]): ลิสต์ของข้อความจากแต่ละ Chunk

    Returns:
        str: ข้อความที่รวมกันเป็นเนื้อหาเดียวและลดความซ้ำซ้อนแล้ว
    """
    # Import helper function
    from src.utils.text_deduplication import split_sentences_smart
    
    if not transcripts:
        return ""

    full_text = transcripts[0].strip()

    for chunk_index in range(1, len(transcripts)):
        prev_text = full_text
        next_text = transcripts[chunk_index].strip()
        if not next_text:
            continue

        # --- ขั้นตอนที่ 1: กำหนดขอบเขต (Window) สำหรับค้นหา Overlap ---
        # [IMPROVED] เพิ่ม window จาก 500 → 1500 เพื่อครอบคลุมมากขึ้น
        overlap_window = 1500
        prev_tail = prev_text[-overlap_window:]
        next_head = next_text[:overlap_window]

        # --- ขั้นตอนที่ 2: แบ่งข้อความในขอบเขตให้เป็นประโยค ---
        # [IMPROVED] ใช้ split_sentences_smart() แทน regex เดิม
        prev_sentences = split_sentences_smart(prev_tail)
        next_sentences = split_sentences_smart(next_head)

        # ถ้าส่วนหัวหรือท้ายไม่มีประโยคที่สมบูรณ์ ให้ข้ามการตรวจสอบและต่อข้อความไปเลย
        if not prev_sentences or not next_sentences:
            full_text += " " + next_text
            continue

        # --- ขั้นตอนที่ 3: ค้นหาประโยคที่ซ้ำกันเพื่อหาจุดตัด (Cut-off Point) ---
        # [IMPROVED] เช็คหลายประโยคแทนที่จะเช็คแค่ประโยคสุดท้าย
        check_count = min(3, len(prev_sentences))  # เช็ค 3 ประโยคสุดท้าย (หรือน้อยกว่าถ้ามีไม่ถึง)
        boundary_sentences = prev_sentences[-check_count:]  # ตัดเอาประโยคสุดท้ายมา
        
        best_match_index = -1  # ตำแหน่งประโยคที่ match ดีที่สุดใน next_sentences
        # ใช้ค่า threshold จาก config หรือใช้ default 0.7 (70% ความคล้าย)
        try:
            from config import DEDUP_MERGE_OVERLAP_THRESHOLD
            highest_ratio = DEDUP_MERGE_OVERLAP_THRESHOLD  # จาก config (default 70%)
        except ImportError:
            highest_ratio = 0.7  # Fallback ถ้าไม่มี config
        best_anchor_idx = -1  # ตำแหน่งประโยคที่ match ดีที่สุดใน boundary_sentences

        # เช็คแต่ละประโยคในขอบเขตท้ายของ chunk ก่อนหน้า กับหัวของ chunk ถัดไป
        for anchor_idx, anchor_sent in enumerate(boundary_sentences):
            for sentence_index, next_sent in enumerate(next_sentences):
                # ใช้ SequenceMatcher.ratio() เพื่อวัดความคล้ายของประโยค (0.0 - 1.0)
                # ratio = 1.0 หมายถึงเหมือนกันทุกตัวอักษร
                ratio = difflib.SequenceMatcher(None, anchor_sent, next_sent).ratio()
                if ratio > highest_ratio:
                    # เจอประโยคที่คล้ายกันมากกว่าเดิม → อัปเดต
                    highest_ratio = ratio
                    best_match_index = sentence_index
                    best_anchor_idx = anchor_idx

        # --- ขั้นตอนที่ 4: รวมข้อความโดยตัดส่วนที่ซ้ำซ้อนออก ---
        if best_match_index != -1 and best_match_index + 1 < len(next_sentences):
            # เจอประโยคที่ซ้ำแล้ว → เราจะถือว่าเนื้อหาใหม่ที่แท้จริง เริ่มต้นที่ "ประโยคถัดไป" จากจุดที่ซ้ำ
            # เพื่อไม่ให้เอาส่วนที่ซ้ำมาต่อซ้ำอีก
            start_of_new_content = next_sentences[best_match_index + 1]
            
            # ค้นหาตำแหน่งเริ่มต้นของประโยคใหม่นั้นใน `next_text` ทั้งหมด
            cutoff_pos = next_text.find(start_of_new_content)
            
            # ถ้าเจอตำแหน่ง ก็ตัดเอาเฉพาะส่วนที่เป็นเนื้อหาใหม่จริงๆ มาต่อท้าย (ข้ามส่วนที่ซ้ำ)
            if cutoff_pos != -1:
                full_text += " " + next_text[cutoff_pos:]
            else:
                # [Fallback] ถ้าหาตำแหน่งไม่เจอ (ไม่น่าจะเกิด แต่เผื่อ edge case)
                full_text += " " + next_text
        else:
            # [Fallback] ถ้าไม่เจอประโยคที่คล้ายกันเลย หรือประโยคที่ match คือประโยคสุดท้าย
            # → ไม่สามารถตัด overlap ได้ ต่อทั้งหมดไปเลย
            full_text += " " + next_text

    # --- ขั้นตอนสุดท้าย: ทำความสะอาดข้อความ ---
    return re.sub(r"\s+", " ", full_text).strip()

# -----------------------------------


# --- AUDIO PROCESSOR ---

class AudioProcessor:
    @staticmethod
    def preprocess_audio(file_path: str) -> AudioSegment | None:
        """
        โหลดและแปลงไฟล์เสียงให้อยู่ในรูปแบบมาตรฐานสำหรับ ASR

        - แปลงเป็น Mono Channel
        - ตั้งค่า Frame Rate เป็น 16kHz
        - ทำ Normalization เพื่อปรับระดับความดัง

        Args:
            file_path (str): ที่อยู่ของไฟล์เสียง

        Returns:
            AudioSegment | None: Object ของไฟล์เสียงที่ผ่านการประมวลผลแล้ว หรือ None หากเกิดข้อผิดพลาด
        """
        try:
            audio = AudioSegment.from_file(file_path)
            if audio.channels > 1:
                audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            audio = effects.normalize(audio)
            return audio
        except Exception:
            return None


def transcribe_chunk(client, chunk_data, chunk_index, prompt):
    """
    ส่งไฟล์เสียงส่วนย่อย (Chunk) ไปถอดความที่ ASR API

    มีกลไก Retry ในตัว หากการเรียก API ล้มเหลวจะพยายามใหม่สูงสุด 3 ครั้ง
    ก่อนจะยอมแพ้และคืนค่าเป็นสตริงว่าง

    Args:
        client: Instance ของ OpenAI client ที่เชื่อมต่อกับ ASR API
        chunk_data: ข้อมูล binary ของไฟล์เสียง (WAV format)
        chunk_index (int): ลำดับของ Chunk (สำหรับ Logging)
        prompt (str): Prompt ที่จะส่งให้ ASR เพื่อเป็นแนวทางในการถอดเสียง

    Returns:
        str: ข้อความที่ถอดเสียงได้ หรือสตริงว่างหากล้มเหลว
    """
    # ใช้ Retry แบบ Exponential Backoff สูงสุด 5 ครั้งเพื่อเพิ่มโอกาสสำเร็จ
    for attempt in range(5):
        try:
            # สร้าง BytesIO Object (ไฟล์เสมือนในหน่วยความจำ) จากข้อมูล binary
            file_buffer = io.BytesIO(chunk_data)
            
            # [FIX] กลับไปใช้ .wav เพื่อความชัวร์ (Stable Strategy)
            # แก้ปัญหา MIME type error ของ Typhoon (reject audio/mpeg)
            response = client.audio.transcriptions.create(
                model="typhoon-asr-realtime",
                file=("audio.wav", file_buffer),  # (filename, file_object)
                language="th",  # ภาษาไทย
                prompt=prompt,  # Prompt แนะนำคำศัพท์/บริบท
                temperature=0.2,  # ความ creative น้อย = ผลลัพธ์คงที่มากขึ้น
            )
            return response.text  # คืนข้อความที่ถอดเสียงได้
        except Exception as e:
            # Exponential Backoff: รอ 1s, 2s, 4s, 8s, 16s
            wait_time = 2 ** attempt
            print(f"\n   ⚠️ ASR Warning (Chunk {chunk_index}, Attempt {attempt+1}/5): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    print(f"   ❌ ASR Error: Chunk {chunk_index} failed after 5 retries.")
    return ""