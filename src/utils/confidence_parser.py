# utils/confidence_parser.py
"""
Confidence-based correction utilities

Parse [UNSURE:term:confidence%] markers from LLM output
Extract low-confidence terms for targeted re-correction
"""

import re
from typing import List, Dict, Tuple


def parse_confidence_markers(text: str) -> Tuple[str, List[Dict]]:
    """
    Parse [UNSURE:term:confidence%] markers from corrected text
    
    Args:
        text: Corrected text with confidence markers
        
    Returns:
        Tuple of (clean_text, uncertain_terms)
        - clean_text: Text with markers removed
        - uncertain_terms: List of {term, confidence, position}
    
    Example:
        Input: "หุ้น [UNSURE:Dow Jones:80%] ปรับตัวลง"
        Output: ("หุ้น Dow Jones ปรับตัวลง", 
                [{"term": "Dow Jones", "confidence": 80, "position": 4}])
    """
    pattern = r'\[UNSURE:([^:]+):(\d+)%\]'
    
    uncertain_terms = []
    position = 0
    
    # Find all markers
    for match in re.finditer(pattern, text):
        term = match.group(1)
        confidence = int(match.group(2))
        
        # Calculate position in clean text
        start_pos = match.start() - position
        
        uncertain_terms.append({
            "term": term,
            "confidence": confidence,
            "position": start_pos,
            "raw_match": match.group(0)
        })
        
        # Adjust position offset for next matches
        position += len(match.group(0)) - len(term)
    
    # Remove markers from text
    clean_text = re.sub(pattern, r'\1', text)
    
    return clean_text, uncertain_terms


def get_low_confidence_chunks(text: str, uncertain_terms: List[Dict], 
                               threshold: int = 80, context_window: int = 100) -> List[Dict]:
    """
    Extract chunks containing low-confidence terms for targeted correction
    
    Args:
        text: Clean text (after removing markers)
        uncertain_terms: List from parse_confidence_markers()
        threshold: Confidence threshold (default 80%)
        context_window: Characters around term to include (default 100)
        
    Returns:
        List of chunks needing re-correction with context
    """
    low_confidence = [t for t in uncertain_terms if t["confidence"] < threshold]
    
    chunks = []
    for term_info in low_confidence:
        pos = term_info["position"]
        
        # Extract context window
        start = max(0, pos - context_window)
        end = min(len(text), pos + len(term_info["term"]) + context_window)
        
        chunk = text[start:end]
        
        chunks.append({
            "term": term_info["term"],
            "confidence": term_info["confidence"],
            "chunk": chunk,
            "start_pos": start,
            "end_pos": end,
            "context_before": text[start:pos],
            "context_after": text[pos + len(term_info["term"]):end]
        })
    
    return chunks


def format_for_targeted_correction(chunk_info: Dict) -> str:
    """
    Format chunk for targeted correction prompt
    
    Args:
        chunk_info: Chunk from get_low_confidence_chunks()
        
    Returns:
        Formatted prompt text
    """
    return f"""
พบคำที่ไม่แน่ใจ: "{chunk_info['term']}" (confidence: {chunk_info['confidence']}%)

บริบทรอบข้าง:
\"\"\"
{chunk_info['chunk']}
\"\"\"

คำที่สงสัย: {chunk_info['term']}
ก่อนหน้า: ...{chunk_info['context_before'][-50:]}
หลังจาก: {chunk_info['context_after'][:50]}...

โปรดตรวจสอบอีกครั้งว่า "{chunk_info['term']}" ถูกต้องหรือไม่?
ถ้าแก้ไข ให้ระบุ confidence ใหม่ด้วย
"""


def merge_corrections(original: str, targeted_corrections: Dict[str, str]) -> str:
    """
    Merge targeted corrections back into original text
    
    Args:
        original: Original corrected text
        targeted_corrections: Dict {old_term: new_term}
        
    Returns:
        Text with targeted corrections applied
    """
    result = original
    
    for old_term, new_term in targeted_corrections.items():
        result = result.replace(old_term, new_term)
    
    return result


def calculate_overall_confidence(uncertain_terms: List[Dict]) -> float:
    """
    Calculate overall confidence score for the correction
    
    Args:
        uncertain_terms: List from parse_confidence_markers()
        
    Returns:
        Overall confidence (0-100)
    """
    if not uncertain_terms:
        return 100.0  # No uncertainties = high confidence
    
    # Weighted average by number of uncertain terms
    total_weight = len(uncertain_terms)
    weighted_sum = sum(t["confidence"] for t in uncertain_terms)
    
    return weighted_sum / total_weight if total_weight > 0 else 100.0


# Example usage
if __name__ == "__main__":
    # Test
    test_text = "หุ้น [UNSURE:Dow Jones:80%] และ [UNSURE:Nasdaq:90%] ปรับตัวลง AOT เจรจากับ [UNSURE:KingPower:60%]"
    
    clean, uncertain = parse_confidence_markers(test_text)
    print(f"Clean text: {clean}")
    print(f"Uncertain terms: {uncertain}")
    
    low_conf = get_low_confidence_chunks(clean, uncertain, threshold=85)
    print(f"\nLow confidence chunks: {len(low_conf)}")
    for chunk in low_conf:
        print(f"  - {chunk['term']} ({chunk['confidence']}%)")
    
    overall = calculate_overall_confidence(uncertain)
    print(f"\nOverall confidence: {overall:.1f}%")
