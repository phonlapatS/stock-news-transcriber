#!/usr/bin/env python3
"""
Utility functions for text processing and duplicate detection
"""

from difflib import SequenceMatcher
from typing import List, Set, Tuple, Dict
import re
import math
from collections import Counter

try:
    from pythainlp.tokenize import sent_tokenize as thai_sent_tokenize
    HAS_PYTHAINLP = True
except ImportError:
    HAS_PYTHAINLP = False


def calculate_sentence_similarity(sent1: str, sent2: str) -> float:
    """
    Calculate similarity between two sentences using SequenceMatcher
    
    Args:
        sent1: First sentence
        sent2: Second sentence
        
    Returns:
        Similarity score between 0 and 1
    """
    # Normalize sentences
    s1 = sent1.strip().lower()
    s2 = sent2.strip().lower()
    
    # Calculate similarity
    return SequenceMatcher(None, s1, s2).ratio()


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences (Thai-aware) using pythainlp if available.
    """
    if not text:
        return []

    # 1. Use PyThaiNLP for linguistic splitting if available
    sentences = []
    if HAS_PYTHAINLP:
        try:
            sentences = thai_sent_tokenize(text)
        except Exception as e:
            # Fallback if internal dependencies like pycrfsuite are missing
            print(f"   ⚠️ PyThaiNLP sent_tokenize failed: {e}. Falling back to regex.")
            sentences = re.split(r'[\.!?]\s+|\n\n+', text)
    else:
        # Fallback to regex-based splitting
        # Split by common Thai sentence endings or double newlines
        sentences = re.split(r'[\.!?]\s+|\n\n+', text)
    
    # 2. Refinement: Even after sent_tokenize, ASR blocks can be huge without punctuation
    refined_sentences = []
    for s in sentences:
        s = s.strip()
        if not s: continue
        
        # If any sentence is still very long (>150 chars), try splitting by multiple spaces or newlines
        if len(s) > 150:
            parts = re.split(r'\s{2,}|\n', s)
            for p in parts:
                p = p.strip()
                if not p: continue
                # If STILL too long and contains many words (spaces), force a split at ~100 chars at a space boundary
                if len(p) > 200:
                    # Very crude split for extremely long ASR blocks
                    words = p.split(' ')
                    temp_sent = []
                    current_len = 0
                    for w in words:
                        temp_sent.append(w)
                        current_len += len(w) + 1
                        if current_len > 120:
                            refined_sentences.append(' '.join(temp_sent).strip())
                            temp_sent = []
                            current_len = 0
                    if temp_sent:
                        refined_sentences.append(' '.join(temp_sent).strip())
                else:
                    refined_sentences.append(p)
        else:
            refined_sentences.append(s)
    
    # Clean and filter
    result = [s for s in refined_sentences if s and len(s) > 2]
    return result


def remove_duplicate_sentences(text: str, similarity_threshold: float = 0.75) -> str:
    """
    Remove duplicate/similar sentences from text
    
    Args:
        text: Input text with potential duplicates
        similarity_threshold: Similarity threshold (0.7-0.8 recommended)
        
    Returns:
        Text with duplicates removed
    """
    sentences = split_into_sentences(text)
    
    if len(sentences) <= 1:
        return text
    
    # Track which sentences to keep
    keep_indices = []
    seen_groups = []  # Groups of similar sentences
    
    for i, sent1 in enumerate(sentences):
        is_duplicate = False
        
        # Check against already kept sentences
        for group in seen_groups:
            representative_idx = group[0]
            representative_sent = sentences[representative_idx]
            
            similarity = calculate_sentence_similarity(sent1, representative_sent)
            
            if similarity >= similarity_threshold:
                # Found a duplicate - add to group
                group.append(i)
                is_duplicate = True
                break
        
        if not is_duplicate:
            # New unique sentence - create new group
            keep_indices.append(i)
            seen_groups.append([i])
    
    # Reconstruct text with only kept sentences
    result_sentences = [sentences[i] for i in keep_indices]
    
    # Join sentences back
    return '\n\n'.join(result_sentences)


def select_best_sentence(sentences: List[str]) -> str:
    """
    Select the best sentence from a list of similar sentences
    
    Criteria:
    - Longer is generally better (more complete)
    - Has proper punctuation
    - No obvious errors
    
    Args:
        sentences: List of similar sentences
        
    Returns:
        Best sentence
    """
    if len(sentences) == 1:
        return sentences[0]
    
    # Score each sentence
    scored = []
    
    for sent in sentences:
        score = 0
        
        # Length bonus (longer = more complete)
        score += len(sent) / 10
        
        # Proper ending bonus
        if sent.strip()[-1] in '.!?':
            score += 5
        
        # No obvious errors (very basic check)
        if not '...' in sent and not '???' in sent:
            score += 2
        
        scored.append((score, sent))
    
    # Return highest scoring sentence
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]


def remove_duplicate_sentences_smart(text: str, similarity_threshold: float = 0.75) -> Tuple[str, int]:
    """
    Smart duplicate removal - keeps best version of each sentence
    
    Args:
        text: Input text
        similarity_threshold: Similarity threshold (default 0.75 = 75%)
        
    Returns:
        Tuple of (cleaned_text, num_duplicates_removed)
    """
    sentences = split_into_sentences(text)
    
    if len(sentences) <= 1:
        return text, 0
    
    # Group similar sentences
    groups = []
    used = set()
    
    for i, sent1 in enumerate(sentences):
        if i in used:
            continue
        
        group = [sent1]
        used.add(i)
        
        # Find all similar sentences
        for j, sent2 in enumerate(sentences[i+1:], start=i+1):
            if j in used:
                continue
            
            similarity = calculate_sentence_similarity(sent1, sent2)
            
            if similarity >= similarity_threshold:
                group.append(sent2)
                used.add(j)
        
        groups.append(group)
    
    # Select best from each group
    best_sentences = [select_best_sentence(group) for group in groups]
    
    num_removed = len(sentences) - len(best_sentences)
    
    # Reconstruct text
    cleaned_text = '\n\n'.join(best_sentences)
    
    return cleaned_text, num_removed


def split_sentences_smart(text: str, strategy: str = None) -> List[str]:
    """
    Smart sentence splitting with configurable strategy
    
    Args:
        text: Input text
        strategy: Splitting strategy - "newline", "punctuation", or "auto"
                 If None, uses config.SENTENCE_SPLIT_STRATEGY
    
    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []
    
    # Import config
    if strategy is None:
        try:
            from config import SENTENCE_SPLIT_STRATEGY
            strategy = SENTENCE_SPLIT_STRATEGY
        except ImportError:
            strategy = "newline"  # Fallback default
    
    sentences = []
    
    if strategy == "newline":
        # PRIMARY: Newline-based
        raw_lines = [s.strip() for s in text.split('\n') if s.strip()]
        for line in raw_lines:
            # For each line, further split using pythainlp if it's long
            if HAS_PYTHAINLP and len(line) > 100:
                try:
                    sentences.extend(thai_sent_tokenize(line))
                except Exception as e:
                    # Fallback to double space split if pythainlp fails
                    parts = re.split(r'\s{2,}', line)
                    sentences.extend([p.strip() for p in parts if p.strip()])
            elif len(line) > 200:
                # Fallback to double space split if no pythainlp
                parts = re.split(r'\s{2,}', line)
                sentences.extend([p.strip() for p in parts if p.strip()])
            else:
                sentences.append(line)
        
    elif strategy == "punctuation":
        # Punctuation-based
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
    elif strategy == "auto":
        # AUTO: Use pythainlp if available for best results
        if HAS_PYTHAINLP:
            try:
                sentences = thai_sent_tokenize(text)
            except Exception as e:
                print(f"   ⚠️ PyThaiNLP sent_tokenize failed: {e}. Falling back to regex.")
                # Detect simple strategy
                newline_ratio = text.count('\n') / max(len(text), 1)
                if newline_ratio > 0.05:
                    sentences = [s.strip() for s in text.split('\n') if s.strip()]
                else:
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        else:
            # Detect simple strategy
            newline_ratio = text.count('\n') / max(len(text), 1)
            if newline_ratio > 0.05:
                sentences = [s.strip() for s in text.split('\n') if s.strip()]
            else:
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    
    else:
        # FALLBACK: Default to newline
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
    
    return sentences


def score_sentence_quality(sent: str) -> float:
    """
    Score sentence quality for choosing best version
    
    Criteria:
    - Length (longer = more complete)
    - Proper ending punctuation
    - Unique words ratio (no repetition)
    - No obvious errors
    
    Args:
        sent: Sentence to score
        
    Returns:
        Quality score (higher = better)
    """
    if not sent:
        return 0.0
    
    score = 0.0
    
    # 1. Length bonus (longer = more complete)
    score += len(sent) * 0.1
    
    # 2. Proper ending bonus
    if sent.strip() and sent.strip()[-1] in '.!?':
        score += 20.0
    
    # 3. Word uniqueness (avoid repetition)
    words = sent.split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio > 0.8:  # 80%+ unique words
            score += 10.0
    
    # 4. No obvious errors
    if '...' not in sent and '???' not in sent:
        score += 5.0
    
    # 5. Has numbers/financial terms (relevant content)
    if any(char.isdigit() for char in sent):
        score += 3.0
    
    return score


def remove_chunk_boundary_duplicates(text: str, similarity_threshold: float = 0.85) -> Tuple[str, int]:
    """
    Remove duplicates specifically at chunk boundaries
    
    Uses higher threshold (0.85) as these are likely exact duplicates
    from chunk overlap during processing.
    
    Args:
        text: Merged text that may have boundary duplicates
        similarity_threshold: Similarity threshold (default 0.85 = 85%)
        
    Returns:
        Tuple of (cleaned_text, num_removed)
    """
    sentences = split_sentences_smart(text)
    
    if len(sentences) <= 1:
        return text, 0
    
    cleaned_sentences = []
    skip_indices = set()
    removed_count = 0
    
    for i in range(len(sentences)):
        if i in skip_indices:
            continue
        
        current_sent = sentences[i]
        
        # Check similarity with next few sentences (boundary duplicates might be shifted)
        found_duplicate = False
        look_ahead = 5 # Check next 5 sentences
        
        for j in range(1, look_ahead + 1):
            if i + j < len(sentences) and (i + j) not in skip_indices:
                next_sent = sentences[i + j]
                similarity = calculate_sentence_similarity(current_sent, next_sent)
                
                if similarity >= similarity_threshold:
                    # Found duplicate! Choose better one and skip the other
                    current_quality = score_sentence_quality(current_sent)
                    next_quality = score_sentence_quality(next_sent)
                    
                    if current_quality >= next_quality:
                        # Keep current, skip next
                        skip_indices.add(i + j)
                    else:
                        # Keep next (in future iteration), skip current now
                        found_duplicate = True
                    
                    removed_count += 1
                    break
        
        if not found_duplicate:
            cleaned_sentences.append(current_sent)
    
    cleaned_text = '\n\n'.join(cleaned_sentences)
    return cleaned_text, removed_count
