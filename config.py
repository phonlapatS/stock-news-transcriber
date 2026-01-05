# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==========================================
# 🔑 API KEYS & ENDPOINTS
# ==========================================
# Typhoon ASR: สำหรับถอดเสียงภาษาไทย
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_API_KEY = "sk-vCE2QnUydpGnzic35kI3IcoTsAeWzb2X3jYCCAXDPmfT2JnN" 

# Google Gemini API: สมองหลักของ Agent (Gemini 2.5 Flash - with Key Rotation)
GROQ_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"  
GROQ_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Primary key from .env (will be rotated)

# --- LLM Settings ---
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash-lite")

# Retry and delay settings
MAX_RETRIES = 5  # ลดจาก 10 เป็น 5 เพื่อลด retry loops (เพียงพอสำหรับ key rotation)
LLM_REQUEST_DELAY_SECONDS = 7  # เพิ่มจาก 3 เป็น 7 วินาที เพื่อ respect RPM limit (10/min = 6s minimum)
ASR_CLIENT_TIMEOUT_SECONDS = 120.0  # Timeout สำหรับ ASR Client

# ==========================================
# 📂 PATHS & FILES
# ==========================================
DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=ks2e22C2zGA"
DOWNLOAD_DIR = "downloads"
TRANSCRIPT_OUTPUT_DIR = "transcripts_output"

# Knowledge Base Files
MASTER_KB_FILE = "knowledge_base.json"
CACHE_FILE = "ticker_cache.json"
FINANCE_TERM_FILE = "finance_terms.json"

# ==========================================
# ⚙️ SYSTEM SETTINGS
# ==========================================
# Rate Limiting Configuration (Google Gemini Free Tier Safe Mode)
LLM_REQUEST_DELAY_SECONDS = 3.0  # Increased to safe zone (Max ~20 RPM)
MAX_RETRIES = 1                   # FREE TIER: Reduced from 10 to save requests
MAX_REQUESTS_PER_VIDEO = 50       # Hard cap per video run (increased for text-based calculation)
ENABLE_SHORT_VIDEO_OPTIMIZATION = True # Merge chunks for short videos

# Default Text Processing
TEXT_CHUNK_SIZE = 8000           # Default chunk size if not specified

# ==========================================
# 🧹 DEDUPLICATION SETTINGS
# ==========================================
# Deduplication Configuration (UPDATED: More conservative to preserve content)
# Lower = more aggressive (catches more duplicates, may remove valid content)
# Higher = more conservative (safer, may miss some duplicates)
DEDUP_CONFIG = {
    # boundary: For chunk boundary duplicates (exact overlap) - very strict
    # general: For other duplicates - strict
    "default": {
        "boundary": 0.98,  # 98% similarity (was 0.8 - too aggressive!)
        "general": 0.95    # 95% similarity (was 0.75 - too aggressive!)
    }
}

# Overlap detection during chunk merging
DEDUP_MERGE_OVERLAP_THRESHOLD = float(os.getenv("DEDUP_MERGE_OVERLAP_THRESHOLD", "0.70"))  # 70%

# Sentence splitting strategy
# "newline" = prefer newline splitting (best for Thai ASR)
# "punctuation" = prefer punctuation splitting (best for English/formatted text)
# "auto" = auto-detect based on content
SENTENCE_SPLIT_STRATEGY = os.getenv("SENTENCE_SPLIT_STRATEGY", "newline")

# ==========================================
# 🧠 CONTEXT CACHING (Gemini Native API)
# ==========================================
# Enable Context Caching to reduce token usage by 30-40%
# Set to False for emergency rollback to OpenAI-compatible mode
USE_CONTEXT_CACHING = os.getenv("USE_CONTEXT_CACHING", "False").lower() == "true"

# Cache configuration
CONTEXT_CACHE_TTL_HOURS = 1      # Gemini limit: 1 hour max
CONTEXT_CACHE_REFRESH_MINUTES = 55  # Proactive refresh before expiry

# IMPORTANT: Keep GROQ_BASE_URL for backward compatibility
# Not used when USE_CONTEXT_CACHING=True, but kept for rollback

