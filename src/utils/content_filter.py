#!/usr/bin/env python3
"""
Pre-filter irrelevant content from ASR transcripts before LLM processing
This saves LLM quota by removing non-investment related text
"""

import re
from typing import Tuple


# Patterns for irrelevant content
IRRELEVANT_PATTERNS = [
    # Social media calls-to-action
    r'อย่าลืมกด.*?(subscribe|ไลค์|like|share|แชร์|กระดิ่ง|แจ้งเตือน)',
    r'(subscribe|ไลค์|like|share|แชร์).*?(ช่อง|คลิป|วิดีโอ)',
    r'กด.*?(ติดตาม|follow)',
    r'คอมเม้นท์.*?see first',
    r'อย่าลืมกด.*?กระดิ่ง',
    
    # Platform mentions (non-essential)
    r'ที่รับชมผ่านทาง.*?(youtube|facebook|line|twitter)',
    r'ติดตาม.*?(facebook|youtube|line|twitter)',
    
    # Generic sign-offs (keep if has investment content, remove if standalone)
    r'^สวัสดีครับ\s*$',
    r'^ขอบคุณที่รับชม\s*$',
    r'^แล้วพบกันใหม่\s*$',
]


def is_irrelevant_sentence(sentence: str) -> bool:
    """
    Check if a sentence is irrelevant (non-investment content)
    
    Args:
        sentence: Input sentence
        
    Returns:
        True if irrelevant, False otherwise
    """
    sentence_lower = sentence.lower()
    
    # Check against patterns
    for pattern in IRRELEVANT_PATTERNS:
        if re.search(pattern, sentence_lower, re.IGNORECASE):
            return True
    
    return False


def contains_investment_keywords(text: str) -> bool:
    """
    Check if text contains investment-related keywords
    
    Args:
        text: Input text
        
    Returns:
        True if contains investment keywords
    """
    investment_keywords = [
        'กำไร', 'ขาดทุน', 'รายได้', 'ราคา', 'หุ้น', 'ผลประกอบการ',
        'ไตรมาส', 'qoq', 'yoy', 'margin', 'บาท', 'ล้าน', 'พัน',
        'เติบโต', 'ลดลง', 'เพิ่มขึ้น', 'ปันผล', 'dividend',
        'ซื้อ', 'ขาย', 'trading', 'target', 'ราคาเป้าหมาย',
        'วิเคราะห์', 'คาดการณ์', 'ประมาณการ'
    ]
    
    text_lower = text.lower()
    
    for keyword in investment_keywords:
        if keyword in text_lower:
            return True
    
    return False


def remove_irrelevant_content(text: str) -> Tuple[str, int]:
    """
    Remove irrelevant content from transcript
    
    Args:
        text: Raw transcript text
        
    Returns:
        Tuple of (cleaned_text, num_sentences_removed)
    """
    # FIXED: Use newline-based splitting for Thai language
    # Thai doesn't always use punctuation, so splitting by . ? ! is too aggressive
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    # Filter lines
    kept_lines = []
    removed_count = 0
    
    for line in lines:
        # CONSERVATIVE approach: Only remove if CLEARLY irrelevant
        # AND does NOT contain investment keywords
        is_clearly_irrelevant = is_irrelevant_sentence(line)
        has_investment_content = contains_investment_keywords(line)
        
        # Keep if:
        # 1. Not irrelevant at all, OR
        # 2. Has investment keywords (even if matched irrelevant pattern)
        if not is_clearly_irrelevant or has_investment_content:
            kept_lines.append(line)
        else:
            # Only remove if clearly irrelevant AND no investment content
            removed_count += 1
    
    # SAFETY CHECK: Never remove more than 30% of content
    removal_percentage = (removed_count / len(lines) * 100) if lines else 0
    
    if removal_percentage > 30:
        # Too aggressive! Return original text
        print(f"   ⚠️ Filter too aggressive ({removal_percentage:.1f}% removal). Keeping original.")
        return text, 0
    
    # Reconstruct text (preserve original structure)
    cleaned_text = '\n'.join(kept_lines)
    
    # SAFETY CHECK: Never return empty string
    if not cleaned_text.strip() and text.strip():
        print(f"   ⚠️ Filter removed everything! Keeping original.")
        return text, 0
    
    return cleaned_text, removed_count


def filter_transcript_ends(text: str) -> Tuple[str, bool]:
    """
    Remove common beginning/ending fluff from transcripts
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (cleaned_text, was_modified)
    """
    original_text = text
    lines = text.split('\n')
    
    if len(lines) < 10:
        # Too short to safely filter
        return text, False
    
    # Remove trailing social media CTAs (last 5 lines only)
    last_lines = lines[-5:]
    filtered_last = []
    
    for line in last_lines:
        # Only remove if CLEARLY irrelevant AND no investment content
        if not (is_irrelevant_sentence(line) and not contains_investment_keywords(line)):
            filtered_last.append(line)
    
    # Reconstruct
    if len(filtered_last) < len(last_lines):
        lines = lines[:-5] + filtered_last
        text = '\n'.join(lines)
    
    was_modified = (text != original_text)
    
    return text, was_modified


def preprocess_transcript(text: str, verbose: bool = True) -> str:
    """
    Main preprocessing function - removes all irrelevant content
    
    Args:
        text: Raw transcript
        verbose: Print stats
        
    Returns:
        Cleaned transcript
    """
    original_length = len(text)
    
    # SAFETY CHECK: Don't process if too short
    if original_length < 100:
        if verbose:
            print(f"   ⚠️ Transcript too short ({original_length} chars). Skipping filter.")
        return text
    
    # Step 1: Filter ends
    text, ends_modified = filter_transcript_ends(text)
    
    # Step 2: Remove irrelevant sentences
    text, num_removed = remove_irrelevant_content(text)
    
    cleaned_length = len(text)
    saved_chars = original_length - cleaned_length
    saved_pct = (saved_chars / original_length * 100) if original_length > 0 else 0
    
    if verbose and (num_removed > 0 or ends_modified):
        print(f"🧹 Content Filter:")
        print(f"   - Removed {num_removed} irrelevant sentence(s)")
        print(f"   - Saved {saved_chars} characters ({saved_pct:.1f}% reduction)")
    
    return text
