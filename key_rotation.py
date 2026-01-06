# key_rotation.py
"""
Simple API Key Manager for Gemini API

Provides get_current_key() function to retrieve Gemini API key
from environment variables loaded by config.py
"""

import os
from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv()

# Key rotation state
_current_key_index = 0
_available_keys = []


def _load_keys_from_env():
    """
    Load all available Gemini API keys from environment
    
    Supports multiple keys and both naming conventions:
    - GEMINI_API_KEY / GOOGLE_API_KEY (primary)
    - GEMINI_API_KEY_2 / GOOGLE_API_KEY_2 (secondary)
    - GEMINI_API_KEY_3 / GOOGLE_API_KEY_3 (tertiary)
    - ... etc
    """
    global _available_keys
    
    if _available_keys:  # Already loaded
        return
    
    keys = []
    
    # Load primary key - try both GEMINI_API_KEY and GOOGLE_API_KEY
    primary = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if primary and primary.strip() and not primary.startswith("YOUR_"):
        keys.append(primary.strip())
    
    # Load additional keys (try both naming conventions)
    for i in range(2, 11):  # Support up to 10 keys total
        key = os.getenv(f"GEMINI_API_KEY_{i}") or os.getenv(f"GOOGLE_API_KEY_{i}")
        if key and key.strip() and not key.startswith("YOUR_"):
            keys.append(key.strip())
    
    if not keys:
        raise ValueError(
            "❌ No valid Gemini API keys found in .env file!\n"
            "   Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file"
        )
    
    _available_keys = keys
    
    # Only print on first load
    if len(keys) == 1:
        print(f"🔑 Using 1 Gemini API key")
    else:
        print(f"🔑 Loaded {len(keys)} Gemini API keys for rotation")


def get_current_key() -> str:
    """
    Get current Gemini API key
    
    Returns:
        str: Current Gemini API key
        
    Raises:
        ValueError: If no valid keys are found
    """
    _load_keys_from_env()
    
    if not _available_keys:
        raise ValueError("No Gemini API keys available")
    
    return _available_keys[_current_key_index]


def rotate_to_next_key():
    """
    Rotate to next available API key (if multiple keys exist)
    
    Usage:
        Called when hitting rate limit to switch to next key
    """
    global _current_key_index
    
    _load_keys_from_env()
    
    if len(_available_keys) <= 1:
        return  # No rotation needed with single key
    
    _current_key_index = (_current_key_index + 1) % len(_available_keys)
    print(f"🔄 Rotated to API key #{_current_key_index + 1}/{len(_available_keys)}")


def get_available_key_count() -> int:
    """
    Get total number of available API keys
    
    Returns:
        int: Number of keys loaded
    """
    _load_keys_from_env()
    return len(_available_keys)
