"""
Safe Deduplication Module
Removes only exact duplicate sentences with safety checks
"""

def safe_deduplicate(text: str, max_removal_percent: float = 0.10) -> tuple:
    """
    Safely remove exact duplicate bullets and sentences
    
    Handles:
    - Markdown bullets: -   content
    - Regular lines
    - Consecutive repetitions (detects patterns)
    
    Args:
        text: Input text
        max_removal_percent: Maximum % of content allowed to remove (default 10%)
    
    Returns:
        (cleaned_text, stats_dict)
    """
    import re
    
    lines = text.split('\n')
    seen = set()
    result = []
    removed_count = 0
    
    # Track consecutive identical lines to detect bulk repetition
    prev_content = None
    consecutive_count = 0
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip empty lines (keep them)
        if not line_stripped:
            result.append(line)
            prev_content = None
            consecutive_count = 0
            continue
        
        # ===== NEW: Extract bullet content =====
        bullet_match = re.match(r'^-\s+(.+)$', line_stripped)
        if bullet_match:
            # It's a bullet, extract content
            content = bullet_match.group(1).strip()
        else:
            # It's a regular line
            content = line_stripped
        
        # Check consecutive repetition (pattern detection)
        if content == prev_content:
            consecutive_count += 1
            # If same content appears 3+ times in a row, it's suspicious
            if consecutive_count >= 2:  # 3rd occurrence onwards
                removed_count += 1
                if consecutive_count == 2:  # First time detecting repetition
                    print(f"   🚨 Detected repetition pattern: {content[:60]}...")
                continue
        else:
            consecutive_count = 0
        
        # Check if exact duplicate (anywhere in text)
        if content in seen:
            removed_count += 1
            print(f"   🗑️  Removed duplicate: {content[:60]}...")
            continue
        
        # Keep this line
        seen.add(content)
        result.append(line)
        prev_content = content
    
    cleaned_text = '\n'.join(result)
    
    # SAFETY CHECK
    original_words = len(text.split())
    cleaned_words = len(cleaned_text.split())
    removal_percent = (original_words - cleaned_words) / original_words if original_words > 0 else 0
    
    stats = {
        "removed_sentences": removed_count,
        "original_words": original_words,
        "cleaned_words": cleaned_words,
        "removal_percent": removal_percent,
        "safe": removal_percent <= max_removal_percent
    }
    
    # If too much removed, return original
    if not stats["safe"]:
        print(f"   ⚠️  SAFETY CHECK FAILED: {removal_percent:.1%} removed (max: {max_removal_percent:.1%})")
        print(f"   ↩️  Keeping original text")
        return text, {**stats, "applied": False}
    
    if removed_count > 0:
        print(f"   ✅ Safe dedup: removed {removed_count} duplicates ({removal_percent:.1%} of content)")
    
    return cleaned_text, {**stats, "applied": True}



def remove_exact_duplicates_simple(text: str) -> str:
    """Simple wrapper for exact duplicate removal"""
    cleaned, stats = safe_deduplicate(text)
    return cleaned
