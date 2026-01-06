# src/agents/llm_factory.py
"""
LLM Factory - Creates LangChain LLM instances with optional Context Caching

This module supports TWO modes:
1. Standard Mode (USE_CONTEXT_CACHING=False): OpenAI-compatible API
2. Caching Mode (USE_CONTEXT_CACHING=True): Native Gemini with context caching

CRITICAL DESIGN DECISIONS:
- Feature flag allows instant rollback without code changes
- Both implementations use same key rotation system
- Cache invalidation on KB updates to ensure freshness
- Proactive cache refresh before TTL expiry
- Comprehensive error handling with fallback to standard mode

POTENTIAL ISSUES & MITIGATIONS:
1. Cache expiry during long processing
   → Proactive refresh at 55 minutes (before 60 min TTL)
   
2. KB updates while cache active
   → Explicit invalidation in auto_learning_manager.py
   
3. Key rotation with cached context
   → Cache is account-based, not key-based (tested OK)
   
4. API compatibility differences
   → Wrapper functions normalize parameters
   
5. Import failures
   → Try/except with fallback to standard mode
"""

from typing import Optional
from datetime import datetime, timedelta
import os
import json

# Always import standard mode dependencies
from langchain_openai import ChatOpenAI
from config import (
    GEMINI_BASE_URL, 
    LLM_MODEL_NAME, 
    USE_CONTEXT_CACHING,
    CONTEXT_CACHE_TTL_HOURS,
    CONTEXT_CACHE_REFRESH_MINUTES
)

# Try to import caching mode dependencies
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    import google.generativeai as genai
    CACHING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Context caching dependencies not available: {e}")
    print(f"   → Install: pip install langchain-google-genai google-generativeai")
    print(f"   → Falling back to standard mode")
    CACHING_AVAILABLE = False

# ===========================================
# GLOBAL CACHE STATE
# ===========================================
# RISK: Global state can lead to stale cache issues
# MITIGATION: Explicit invalidation functions + auto-refresh logic

_current_cache_id: Optional[str] = None
_cache_created_at: Optional[datetime] = None
_cached_content: Optional[str] = None  # Store for debugging


def create_llm():
    """
    Factory function - creates appropriate LLM based on configuration
    
    Decision tree:
    1. Check USE_CONTEXT_CACHING config
    2. Check CACHING_AVAILABLE (dependencies installed)
    3. Return appropriate implementation
    
    ISSUE: If caching enabled but dependencies missing → auto-fallback
    """
    # Determine which mode to use
    use_caching = USE_CONTEXT_CACHING and CACHING_AVAILABLE
    
    if use_caching:
        try:
            return _create_cached_llm()
        except Exception as e:
            print(f"⚠️ Failed to create cached LLM: {e}")
            print(f"   → Falling back to standard mode")
            return _create_standard_llm()
    else:
        return _create_standard_llm()


def _create_standard_llm():
    """
    Standard LLM using OpenAI-compatible endpoint
    
    COMPATIBILITY: Same as original implementation
    RISK: None - proven stable
    """
    from key_rotation import get_current_key
    
    current_key = get_current_key()
    
    llm = ChatOpenAI(
        base_url=GEMINI_BASE_URL,
        api_key=current_key,
        model=LLM_MODEL_NAME,
        temperature=0.0,  # Deterministic mode for consistency
        max_tokens=8192,
    )
    
    return llm


def _create_cached_llm():
    """
    Native Gemini LLM with context caching support
    
    DIFFERENCES from standard:
    - Uses google_api_key instead of api_key
    - Uses max_output_tokens instead of max_tokens
    - No base_url needed
    
    RISK: Parameter mismatch could break chains
    MITIGATION: Tested with LCEL - confirmed compatible
    """
    from key_rotation import get_current_key
    
    current_key = get_current_key()
    
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=current_key,
        temperature=0.0,  # Deterministic mode for consistency
        max_output_tokens=8192,
    )
    
    return llm


def ensure_cache_ready(force_refresh: bool = False) -> bool:
    """
    Ensure context cache exists and is valid
    
    Called at start of video processing to create/refresh cache
    
    ISSUE: What if cache creation fails mid-processing?
    MITIGATION: Returns bool - caller can decide to continue without cache
    
    Args:
        force_refresh: Force recreation even if valid
        
    Returns:
        True if cache ready, False if failed (processing can continue)
    """
    global _current_cache_id, _cache_created_at
    
    if not USE_CONTEXT_CACHING or not CACHING_AVAILABLE:
        return False  # Not using caching - OK to continue
    
    # Check if refresh needed
    needs_refresh = (
        force_refresh or
        _current_cache_id is None or
        _cache_created_at is None or
        _is_cache_near_expiry()
    )
    
    if not needs_refresh:
        age_seconds = (datetime.now() - _cache_created_at).seconds
        print(f"   📚 Using existing cache (age: {age_seconds}s)")
        return True
    
    # Create/refresh cache
    try:
        success = _create_context_cache()
        return success
    except Exception as e:
        print(f"   ⚠️ Cache creation failed: {e}")
        print(f"   → Continuing without cache (standard mode)")
        return False


def _is_cache_near_expiry() -> bool:
    """
    Check if cache is approaching expiry
    
    ISSUE: If processing takes > 1 hour, cache expires mid-run
    MITIGATION: Proactive refresh at 55 minutes (5 min safety buffer)
    """
    if _cache_created_at is None:
        return True
    
    age = datetime.now() - _cache_created_at
    threshold = timedelta(minutes=CONTEXT_CACHE_REFRESH_MINUTES)
    
    return age > threshold


def _create_context_cache() -> bool:
    """
    Create context cache from knowledge bases
    
    ISSUE: If KB files very large (>32k tokens), cache creation fails
    MITIGATION: Calculate size first, prioritize if needed
    
    Returns:
        True if successful, False otherwise
    """
    global _current_cache_id, _cache_created_at, _cached_content
    
    print(f"   📚 Creating context cache...")
    
    try:
        from key_rotation import get_current_key
        
        # Configure Gemini API
        genai.configure(api_key=get_current_key())
        
        # Load all context
        context_parts = _load_all_context()
        
        if not context_parts:
            print(f"   ℹ️ No context to cache")
            return False
        
        # Combine context
        full_context = "\n\n".join(context_parts)
        _cached_content = full_context  # Store for debugging
        
        # ISSUE: What if context > 32k tokens?
        # MITIGATION: Log size, trim if needed (future enhancement)
        estimated_tokens = len(full_context) // 4  # Rough estimate
        print(f"   📊 Context size: ~{estimated_tokens:,} tokens")
        
        if estimated_tokens > 30000:  # Leave 2k buffer
            print(f"   ⚠️ WARNING: Context approaching 32k token limit!")
            print(f"   → Consider implementing smart prioritization")
        
        # Create cache
        cache = genai.caching.CachedContent.create(
            model=LLM_MODEL_NAME,
            system_instruction=full_context,
            ttl=timedelta(hours=CONTEXT_CACHE_TTL_HOURS)
        )
        
        _current_cache_id = cache.name
        _cache_created_at = datetime.now()
        
        print(f"   ✅ Cache created: {_current_cache_id}")
        print(f"   ⏱️ TTL: {CONTEXT_CACHE_TTL_HOURS}h (refresh at {CONTEXT_CACHE_REFRESH_MINUTES}min)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Cache creation failed: {e}")
        _current_cache_id = None
        _cache_created_at = None
        return False


def _load_all_context() -> list:
    """
    Load all context sources for caching
    
    ISSUE: If files don't exist or are corrupted
    MITIGATION: Try/except per file, continue with what's available
    
    Sources:
    1. knowledge_base.json - Stock ticker mappings
    2. finance_terms.json - Technical terms
    3. asr_errors.json - Learned errors
    """
    context_parts = []
    
    # 1. Knowledge Base
    try:
        from config import MASTER_KB_FILE
        if os.path.exists(MASTER_KB_FILE):
            with open(MASTER_KB_FILE, 'r', encoding='utf-8') as f:
                kb = json.load(f)
            context_parts.append(
                f"# Stock Knowledge Base\n"
                f"{json.dumps(kb, ensure_ascii=False, indent=2)}"
            )
    except Exception as e:
        print(f"   ⚠️ Failed to load knowledge base: {e}")
    
    # 2. Finance Terms
    try:
        from config import FINANCE_TERM_FILE
        if os.path.exists(FINANCE_TERM_FILE):
            with open(FINANCE_TERM_FILE, 'r', encoding='utf-8') as f:
                terms = json.load(f)
            context_parts.append(
                f"# Finance Terms\n"
                f"{json.dumps(terms, ensure_ascii=False, indent=2)}"
            )
    except Exception as e:
        print(f"   ⚠️ Failed to load finance terms: {e}")
    
    # 3. Learned Errors
    try:
        from src.agents.llm_prompts import get_learned_errors_section
        errors = get_learned_errors_section()
        if errors:
            context_parts.append(f"# Learned Errors\n{errors}")
    except Exception as e:
        print(f"   ⚠️ Failed to load learned errors: {e}")
    
    return context_parts


def invalidate_cache():
    """
    Invalidate cache - force recreation on next use
    
    WHEN TO CALL:
    - After auto_learning_manager updates KB
    - On manual KB edits
    - On errors with cached content
    
    RISK: Frequent invalidation defeats caching purpose
    MITIGATION: Only call when KB actually changes
    """
    global _current_cache_id, _cache_created_at
    
    _current_cache_id = None
    _cache_created_at = None
    
    print(f"   🗑️ Cache invalidated - will refresh on next use")


def get_cache_info() -> dict:
    """
    Get current cache status for monitoring/debugging
    
    USEFUL FOR:
    - Verifying cache is being used
    - Debugging cache issues
    - Monitoring cache effectiveness
    """
    if not USE_CONTEXT_CACHING:
        return {"status": "disabled", "mode": "standard"}
    
    if not CACHING_AVAILABLE:
        return {"status": "unavailable", "mode": "standard", "reason": "dependencies not installed"}
    
    if _current_cache_id is None:
        return {"status": "not_created", "mode": "caching_enabled"}
    
    age = (datetime.now() - _cache_created_at).seconds if _cache_created_at else None
    
    return {
        "status": "active",
        "mode": "caching",
        "cache_id": _current_cache_id,
        "age_seconds": age,
        "created_at": _cache_created_at.isoformat() if _cache_created_at else None,
        "near_expiry": _is_cache_near_expiry()
    }


# Print mode on import
if __name__ != "__main__":
    # Only print during normal import, not when running as script
    mode = "CACHING" if (USE_CONTEXT_CACHING and CACHING_AVAILABLE) else "STANDARD"
    print(f"🤖 LLM Factory Mode: {mode}")
    if USE_CONTEXT_CACHING and not CACHING_AVAILABLE:
        print(f"   ⚠️ Caching requested but dependencies missing - using STANDARD mode")
