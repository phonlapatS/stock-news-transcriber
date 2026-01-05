# key_rotation.py
"""
Google Gemini API Key Rotation System
Rotates between multiple API keys to bypass free tier limits (20 RPD per key)
"""

import os
from typing import List, Optional
from config import LLM_MODEL_NAME

# ===========================================
# 🔑 GEMINI API KEYS (Add your keys here)
# ===========================================
GEMINI_API_KEYS: List[str] = [
    os.getenv("GOOGLE_API_KEY_1", ""),
    os.getenv("GOOGLE_API_KEY_2", ""),
    os.getenv("GOOGLE_API_KEY_3", ""),
    os.getenv("GOOGLE_API_KEY_4", ""),
    os.getenv("GOOGLE_API_KEY_5", ""),
    os.getenv("GOOGLE_API_KEY_6", ""),
    os.getenv("GOOGLE_API_KEY_7", ""),
    os.getenv("GOOGLE_API_KEY_8", ""),
    os.getenv("GOOGLE_API_KEY_9", ""),
    os.getenv("GOOGLE_API_KEY_10", ""),
]


# Filter out empty keys
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key]

# Current key index
_current_key_index = 0
_keys_tried_this_session = set()  # Track which keys we've tried

def get_current_key() -> str:
    """Get the current active API key"""
    global _current_key_index
    if not GEMINI_API_KEYS:
        raise ValueError("No Gemini API keys configured! Please set GEMINI_API_KEY in .env")
    return GEMINI_API_KEYS[_current_key_index]

def rotate_key() -> bool:
    """
    Rotate to the next available API key
    Returns True if rotation successful, False if no more keys available
    """
    global _current_key_index, _keys_tried_this_session
    
    if len(GEMINI_API_KEYS) <= 1:
        print(f"      ⚠️ Only 1 API key configured - cannot rotate!")
        return False
    
    # Move to next key FIRST
    _current_key_index = (_current_key_index + 1) % len(GEMINI_API_KEYS)
    
    # THEN mark it as tried (after rotation, not before!)
    _keys_tried_this_session.add(_current_key_index)
    
    # Check if we've tried ALL keys
    if len(_keys_tried_this_session) >= len(GEMINI_API_KEYS):
        print(f"      ❌ All {len(GEMINI_API_KEYS)} API keys exhausted!")
        _keys_tried_this_session.clear()  # Reset for next request
        return False
    
    print(f"      ✅ Rotated to API key #{_current_key_index + 1}/{len(GEMINI_API_KEYS)}")
    return True

def reset_rotation():
    """Reset to the first API key and clear tried keys"""
    global _current_key_index, _keys_tried_this_session
    _current_key_index = 0
    _keys_tried_this_session.clear()
    print(f"      🔄 Reset to first API key")

def get_keys_status() -> dict:
    """Get status of all configured keys"""
    return {
        "total_keys": len(GEMINI_API_KEYS),
        "current_index": _current_key_index,
        "keys_tried": len(_keys_tried_this_session),
        "current_key_preview": get_current_key()[:20] + "..." if GEMINI_API_KEYS else None,
        "model": LLM_MODEL_NAME
    }

# Print status on import
if __name__ == "__main__":
    status = get_keys_status()
    print(f"📊 Gemini API Key Rotation Status:")
    print(f"   Total Keys: {status['total_keys']}")
    print(f"   Current: #{status['current_index'] + 1}")
    print(f"   Model: {status['model']}")
