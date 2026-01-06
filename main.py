# main.py

import os
import io
import argparse
import sys
import warnings
import httpx
import yt_dlp
from datetime import datetime
from tqdm import tqdm
import concurrent.futures

# --- 1. Path Setup (สำคัญมากสำหรับการหา Modules) ---
# เพิ่ม path ของโปรเจกต์ปัจจุบันเข้าไปใน system path เพื่อให้ import 'src' ได้ไม่ว่าจะรันจากไหน
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# --- Import Components ---
from config import *
from src.core.utils import (
    AudioProcessor, transcribe_chunk, merge_transcriptions, 
    get_file_size_mb, format_duration, sanitize_filename, 
    format_transcript_paragraphs, normalize_markdown_bullets, remove_markdown_bold
)
from src.core.data_managers import (
    StockContextManager, FinanceTermManager, SmartMarketResolver
)
from src.agents.llm_prompts import DynamicPromptBuilder
from src.agents.graph import build_workflow # ใช้ Graph

# --- Init ASR Client ---
try:
    from openai import OpenAI
except ImportError:
    print("❌ Error: 'openai' library not installed (required for ASR client).")
    sys.exit(1)




# --- Config Logic ---
# --- Config Logic ---
def get_adaptive_config(duration_sec: float) -> dict:
    """
    เลือกชุดการตั้งค่า (Config) ที่เหมาะสมตามความยาวของวิดีโอ (Adaptive 3-Tiers Strategy)
    
    [FINE-TUNED] ปรับจูนให้เหมาะสมกับข้อจำกัด 1MB ของ API (WAV Format)
    - Chunk Size: คงไว้ที่ 20s-25s เพื่อความปลอดภัย (ไม่ติด 413)
    - Max Workers: ปรับตามความยาวคลิปเพื่อไม่ให้ยิง request API จนติด Rate Limit (429)
    """
    if duration_sec <= 900: # Tier 1: คลิปสั้น (< 15 นาที)
        return {
            "CHUNK_DURATION": 20, "OVERLAP_DURATION": 5, "MAX_WORKERS": 10, # เร็วสุดๆ
            "TEXT_CHUNK_SIZE": 8000, "MODE_NAME": "Tier 1: Short-Form (Aggressive Speed)"
        }
    elif duration_sec <= 2700: # Tier 2: คลิปกลาง (15-45 นาที)
        return {
            "CHUNK_DURATION": 25, "OVERLAP_DURATION": 5, "MAX_WORKERS": 5, # สมดุล
            "TEXT_CHUNK_SIZE": 8000, "MODE_NAME": "Tier 2: Medium-Form (Balanced)"
        }
    else: # Tier 3: คลิปยาว (> 45 นาที)
        return {
            "CHUNK_DURATION": 25, "OVERLAP_DURATION": 5, "MAX_WORKERS": 3, # ปลอดภัย เน้นเสถียร
            "TEXT_CHUNK_SIZE": 12000, "MODE_NAME": "Tier 3: Long-Form (Conservative)"
        }


# --- Main Function ---
def main(target_url: str):
    """
    ฟังก์ชันหลักสำหรับดำเนินกระบวนการทั้งหมด (Orchestrator)
    """
    # 1. Initialization
    print("🔄 Initializing System...")
    ctx_mgr = StockContextManager()
    term_mgr = FinanceTermManager()
    resolver = SmartMarketResolver(ctx_mgr, term_mgr)
    prompt_builder = DynamicPromptBuilder(ctx_mgr, term_mgr)
    
    # [NEW] รวม Manager ทั้งหมดไว้ใน dict เพื่อส่งให้ Graph
    data_managers = {
        "stock_context": ctx_mgr, "finance_term": term_mgr, "resolver": resolver
    }

    try:
        http_client = httpx.Client(timeout=ASR_CLIENT_TIMEOUT_SECONDS)
        asr_client = OpenAI(
            base_url=TYPHOON_BASE_URL,
            api_key=TYPHOON_API_KEY,
            http_client=http_client,
        )
    except Exception as e:
        print(f"❌ Error initializing ASR Client: {e}")
        return

    if not os.path.exists(DOWNLOAD_DIR): os.makedirs(DOWNLOAD_DIR)
    if not os.path.exists(TRANSCRIPT_OUTPUT_DIR): os.makedirs(TRANSCRIPT_OUTPUT_DIR)

    # 2. Download & Metadata
    print(f"\n⬇️  [Step 1] Fetching Metadata & Audio from: {target_url}")
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # ดึงข้อมูลเมตาเดตาของวิดีโอมาก่อน
            info = ydl.extract_info(target_url, download=False)
            video_meta = {
                "title": info.get("title", "Unknown_Video"),
                "channel": info.get("uploader", "Unknown_Channel"),
                "duration": info.get("duration", 0)
            }
            
            # คำนวณ Config ที่เหมาะสมกับความยาววิดีโอ
            adaptive_cfg = get_adaptive_config(video_meta["duration"])
            
            print(f"    📄 Title: {video_meta['title']}")
            print(f"    📺 Channel: {video_meta['channel']}")
            print(f"    ⏱️ Duration: {format_duration(video_meta['duration'])}")
            print(f"\n⚙️  Adaptive Config: {adaptive_cfg['MODE_NAME']}")

            # กำหนดชื่อและที่อยู่ของไฟล์เสียงที่ต้องการจะบันทึกอย่างชัดเจน
            audio_filename = os.path.join(DOWNLOAD_DIR, f"{info['id']}.mp3")

            # ตรวจสอบว่ามีไฟล์เสียงที่ดาวน์โหลดไว้แล้วหรือไม่
            if os.path.exists(audio_filename):
                print(f"    🟢 Audio file already exists. Using cached file: {audio_filename}")
            else:
                print("    ⬇️  Downloading audio...")
                ydl.download([target_url])

    except KeyboardInterrupt:
        print("\n🛑 Download cancelled by user (Ctrl+C).")
        return
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return

    # Prepare Output Filenames
    safe_title = sanitize_filename(video_meta['title'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    raw_filename = f"{safe_title}_RAW_{timestamp}.txt"
    clean_filename = f"{safe_title}_CLEAN_{timestamp}.txt"
    markdown_filename = f"{safe_title}_SUMMARY_{timestamp}.md"

    # 3. ASR Transcription
    print("\n🔊 [Step 2] Preprocessing Audio...")
    audio = AudioProcessor.preprocess_audio(audio_filename)
    if not audio: 
        print("❌ Failed to process audio file.")
        return

    print("\n🚀 [Step 3] Transcribing with Typhoon (Stable WAV Mode)...")
    chunk_ms = adaptive_cfg["CHUNK_DURATION"] * 1000
    step = chunk_ms - (adaptive_cfg["OVERLAP_DURATION"] * 1000)
    
    chunks = []
    for chunk_index, start_ms in enumerate(range(0, len(audio), step)):
        buf = io.BytesIO()
        # [REVERT] กลับไปใช้ wav เพื่อความเข้ากันได้สูงสุด แต่ใช้ chunk เล็ก (20s)
        audio[start_ms : min(start_ms + chunk_ms, len(audio))].export(buf, format="wav")
        chunks.append({"data": buf.getvalue(), "index": chunk_index})

    results = {}
    dynamic_prompt = prompt_builder.build_prompt(video_meta)
    
    # [NEW] Graceful Exit Implementation
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=adaptive_cfg["MAX_WORKERS"])
    try:
        futures = {
            executor.submit(transcribe_chunk, asr_client, chunk_info["data"], chunk_info["index"], dynamic_prompt): chunk_info["index"]
            for chunk_info in chunks
        }
        
        # ใช้ tqdm เพื่อแสดง Progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="   Processing Chunks", unit="chunk"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"\n   ⚠️ Warning: ASR task for chunk {idx} failed unexpectedly: {e}")
                results[idx] = ""

    except KeyboardInterrupt:
        print("\n\n🛑 STOPPING: User pressed Ctrl+C. Cancelling all pending tasks...")
        executor.shutdown(wait=False, cancel_futures=True)
        print("   ✅ Tasks cancelled. Exiting...")
        sys.exit(0) # Exit ทันที
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        executor.shutdown(wait=False)
        return
    finally:
        # Ensure executor clean up
        executor.shutdown(wait=True)


    # ประกอบ Transcript พร้อมจัดการส่วนที่ล้มเหลว
    final_transcripts = []

    for chunk_index in sorted(results.keys()):
        text = results.get(chunk_index, "")
        final_transcripts.append(text if text else "[--- TRANSCRIPTION FAILED FOR THIS SEGMENT ---]")
    raw_text = merge_transcriptions(final_transcripts)
    
    # Save RAW Transcript
    with open(os.path.join(TRANSCRIPT_OUTPUT_DIR, raw_filename), "w", encoding="utf-8") as f:
        f.write(raw_text)
    
    
    # [NEW] Content Filter - Remove irrelevant content (social media CTAs, etc.)
    print("\n🧹 Filtering Irrelevant Content...")
    try:
        from src.utils.content_filter import preprocess_transcript
        raw_text = preprocess_transcript(raw_text, verbose=True)
        print("   ✅ Content filter applied")
    except Exception as e:
        print(f"   ⚠️ Content filter failed (continuing with original): {e}")
    
    # [NEW] Pre-processing Layer - Multi-layer defense Layer 1
    print("\n🔧 Pre-processing Layer (Pattern-based fixes)...")
    try:
        from src.utils.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        raw_text, preprocess_stats = preprocessor.preprocess(raw_text, verbose=True)
        print("   ✅ Pre-processing completed")
    except Exception as e:
        print(f"   ⚠️ Pre-processing failed (continuing with original): {e}")
        preprocess_stats = {}
    
    # [NEW] Context-Aware Entity Resolution - Multi-layer defense Layer 2
    print("\n🧠 Context-Aware Entity Resolution...")
    try:
        from src.utils.context_aware_resolver import ContextAwareResolver
        context_resolver = ContextAwareResolver(ctx_mgr, term_mgr)
        raw_text, context_fixes = context_resolver.fix_context_blindness(raw_text)
        if context_fixes:
            print(f"   ✅ Fixed {len(context_fixes)} context-aware entities")
            for fix in context_fixes[:3]:  # Show first 3
                print(f"      • {fix['original']} → {fix['resolved']} ({fix['type']})")
        else:
            print("   ℹ️ No context fixes needed")
    except Exception as e:
        print(f"   ⚠️ Context-aware resolution failed (continuing): {e}")
        
    # ตรวจสอบว่า Transcript ที่ได้มามีเนื้อหาที่ใช้งานได้หรือไม่
    # หากมีแต่ข้อความแสดงข้อผิดพลาด ให้หยุดการทำงานทันที
    if "[--- TRANSCRIPTION FAILED" in raw_text and len(raw_text) < 500:
        print("\n❌ CRITICAL ERROR: Transcription failed for all segments. Cannot proceed to Agentic Workflow.")
        print(f"   Please check ASR API status or network connection. Raw output saved to '{raw_filename}'.")
        return

    # 4. Agentic Workflow (LangGraph)
    print("\n🧠 [Step 4] Agentic Workflow (LangGraph)...")
    
    # [IMPROVED] Create RateLimiter with SMART budget calculation
    # Calculate budget based on ACTUAL TEXT SIZE, not video duration
    text_length = len(raw_text)
    chunk_size = adaptive_cfg.get("TEXT_CHUNK_SIZE", TEXT_CHUNK_SIZE)
    
    # Calculate actual number of chunks needed
    estimated_chunks = max(1, (text_length + chunk_size - 1) // chunk_size)
    
    # Conservative estimate: 
    # - Each chunk needs: correction (1) + verification (1) + possible retry (2) = 4 requests
    # - Plus global: summary (1) + buffer (3) = 4 requests
    # Total: chunks * 4 + 4
    base_requests = estimated_chunks * 4
    buffer_requests = 4
    calculated_budget = base_requests + buffer_requests
    
    # Apply safety limits
    min_budget = 8  # Minimum budget for any video
    max_budget = MAX_REQUESTS_PER_VIDEO  # Hard cap from config
    max_requests = max(min_budget, min(calculated_budget, max_budget))
    
    # Calculate efficiency metrics
    duration_minutes = video_meta["duration"] / 60
    old_budget = max(1, int(duration_minutes // 15)) * 3 + 5  # Old calculation for comparison
    
    print(f"\n📊 Budget Analysis:")
    print(f"   Video Duration: {duration_minutes:.1f} minutes")
    print(f"   Transcript Size: {text_length:,} characters")
    print(f"   Chunk Size: {chunk_size:,} characters")
    print(f"   Estimated Chunks: {estimated_chunks}")
    print(f"   Calculated Budget: {calculated_budget} requests")
    print(f"   Final Budget: {max_requests} requests (min={min_budget}, max={max_budget})")
    if old_budget != max_requests:
        savings = ((max_requests - old_budget) / old_budget * 100) if old_budget > 0 else 0
        direction = "increased" if savings > 0 else "optimized"
        print(f"   💡 Budget {direction} by {abs(savings):.1f}% vs old method (was {old_budget})")
    
    try:
        from rate_limiter import RateLimiter
        rate_limiter = RateLimiter(max_requests_per_video=max_requests)
        print(f"\n🛡️ Rate Limiter initialized with budget: {max_requests} requests")
    except ImportError:
        print("    ⚠️ Rate Limiter not available (running in legacy mode)")
        rate_limiter = None

    # === CONTEXT CACHE INITIALIZATION ===
    # Initialize cache BEFORE workflow to enable token savings
    try:
        from src.agents.llm_factory import ensure_cache_ready, get_cache_info
        print(f"\n📚 Context Cache Initialization...")
        
        cache_ready = ensure_cache_ready()
        
        if cache_ready:
            info = get_cache_info()
            print(f"   ✅ Cache ready - Token savings enabled!")
        else:
            print(f"   ℹ️ Cache not available - Using standard mode")
            
    except Exception as e:
        print(f"   ⚠️ Cache init failed: {e}")
        print(f"   → Continuing with standard mode")
    
    # Build Graph และส่ง data_managers เข้าไป
    app = build_workflow(data_managers=data_managers)
    
    # Initial State
    initial_state = {
        "raw_transcript": raw_text,
        "current_text": raw_text, # Start with raw text
        "channel_name": video_meta["channel"],
        "chunk_size": adaptive_cfg["TEXT_CHUNK_SIZE"],
        "feedback_msg": "",
        "iteration_count": 0,
        "unknown_tickers": [],
        "verified_mappings": [],
        "final_summary": "",
        "quality_score": 10, # Start with perfect score
        "data_managers": data_managers, # [NEW] ส่ง managers เข้า state
        "rate_limiter": rate_limiter,  # [NEW] ส่ง rate limiter เข้า state
        "video_duration_minutes": duration_minutes,  # [NEW] สำหรับ router
    }
    
    # Run Graph!
    final_state = app.invoke(initial_state)
    
    # [NEW] Process duplicate markers
    print("\n🧹 Processing duplicate markers...")
    from src.utils.dedup_marker_processor import remove_duplicates, verify_markers
    
    corrected_with_markers = final_state["current_text"]
    
    # Verify markers first
    marker_stats = verify_markers(corrected_with_markers)
    if marker_stats['has_dup_markers']:
        print(f"   📊 Found {marker_stats['dup_count']} duplicates marked by LLM")
        if marker_stats['invalid_markers']:
            print(f"   ⚠️  Warning: Found invalid markers: {marker_stats['invalid_markers']}")
    
    # Remove duplicates
    corrected_text, removed_count = remove_duplicates(corrected_with_markers, verbose=False)
    if removed_count > 0:
        print(f"   ✅ Removed {removed_count} duplicate lines")
    else:
        print(f"   ℹ️  No duplicates marked")
    
    # [NEW] Remove UNSURE markers for final CLEAN output
    print("   🔍 Stripping [UNSURE] confidence markers...")
    from src.utils.confidence_parser import parse_confidence_markers
    corrected_text, _ = parse_confidence_markers(corrected_text)

    # Extract Results and Format
    corrected_text = format_transcript_paragraphs(corrected_text)
    corrected_text = remove_markdown_bold(corrected_text)
    
    # [NEW] Post-processing Validation - Multi-layer defense Layer 3
    print("\n🔍 Post-processing Validation...")
    try:
        from src.utils.post_validator import PostProcessingValidator
        validator = PostProcessingValidator(ctx_mgr, term_mgr)
        validation_result = validator.validate(raw_text, corrected_text)
        
        if validation_result['total_issues'] > 0:
            print(f"   ⚠️ Found {validation_result['total_issues']} issues:")
            if validation_result['high_severity'] > 0:
                print(f"      🚨 {validation_result['high_severity']} high severity issues")
            if validation_result['medium_severity'] > 0:
                print(f"      ⚠️ {validation_result['medium_severity']} medium severity issues")
            
            # Show first 3 high severity issues
            high_issues = [i for i in validation_result['issues'] if i.get('severity') == 'high']
            for issue in high_issues[:3]:
                print(f"      • {issue.get('description', 'Unknown issue')}")
        else:
            print("   ✅ No validation issues found")
    except Exception as e:
        print(f"   ⚠️ Post-validation failed (continuing): {e}")
    
    # Save CLEAN Transcript
    with open(os.path.join(TRANSCRIPT_OUTPUT_DIR, clean_filename), "w", encoding="utf-8") as f:
        f.write(corrected_text)
    
    # 🤖 Auto Error Detection & Learning (NEW!)
    try:
        # Check if auto_error_detector exists, if not use asr_error_logger logic
        try:
            from auto_error_detector import get_detector
            detector = get_detector()
            print(f"\n🔍 Auto-detecting ASR errors...")
            detected_errors = detector.detect_and_log_errors(
                raw_text=raw_text,
                clean_text=corrected_text,
                video_id=video_id,
                auto_log=True
            )
        except (ImportError, ModuleNotFoundError):
            # Fallback to existing asr_error_logger if manual detector missing
            from asr_error_logger import get_logger
            logger = get_logger()
            # The auto-learning node in graph already does most of this
            # but we can add a manual log check here if needed.
            detected_errors = [] 
            
        if detected_errors:
            print(f"✅ Detected {len(detected_errors)} error patterns")
            
    except Exception as e:
        pass # Silence internal logging errors

    # 5. Output Summary
    final_md = normalize_markdown_bullets(final_state["final_summary"])
    final_md = remove_markdown_bold(final_md)
    
    # [NEW] Strip UNSURE markers from summary as well
    from src.utils.confidence_parser import parse_confidence_markers
    final_md, _ = parse_confidence_markers(final_md)
    
    # Save SUMMARY Markdown
    with open(os.path.join(TRANSCRIPT_OUTPUT_DIR, markdown_filename), "w", encoding="utf-8") as f:
        f.write(final_md)

    print(f"\n✅ SUCCESS! All files saved in '{TRANSCRIPT_OUTPUT_DIR}':")
    print(f"   1. {raw_filename}")
    print(f"   2. {clean_filename}")
    print(f"   3. {markdown_filename}")
    
    print("\n" + "="*20 + " SUMMARY PREVIEW " + "="*20 + "\n")
    print(final_md)
    print("\n" + "="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Agentic ASR Workflow (LangGraph)")
    parser.add_argument("--url", type=str, help="Youtube URL to transcribe")
    
    args = parser.parse_args()
    
    # ใช้ URL ที่รับมา หรือใช้ Default ถ้าไม่ได้ใส่
    target = args.url if args.url else DEFAULT_YOUTUBE_URL
    
    main(target)