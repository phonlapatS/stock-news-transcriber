"""
Deduplication Marker Processor
ประมวลผล [DUP] markers จาก LLM output และลบประโยคซ้ำ
"""

import re


def remove_duplicates(text: str, verbose: bool = False) -> tuple:
    """
    ลบประโยคที่มี [DUP] marker ออกจาก text
    
    Args:
        text: ข้อความที่มี [DUP] markers
        verbose: แสดงข้อความ debug หรือไม่
        
    Returns:
        tuple (cleaned_text, removed_count)
        - cleaned_text: ข้อความที่ลบประโยคซ้ำออกแล้ว
        - removed_count: จำนวนประโยคที่ลบออก
    """
    if not text:
        return text, 0
    
    # Pattern: ประโยคที่มี [DUP] ที่จุดเริ่มต้นหรือตรงกลาง
    # จับทั้ง [DUP] และประโยคนั้นๆ
    lines = text.split('\n')
    cleaned_lines = []
    removed_count = 0
    
    for line in lines:
        # ถ้าบรรทัดมี [DUP] → ข้ามไป
        if '[DUP]' in line:
            removed_count += 1
            if verbose:
                print(f"   🗑️ Removed: {line[:80]}...")
            continue
        cleaned_lines.append(line)
    
    if verbose and removed_count > 0:
        print(f"   ✅ Total removed: {removed_count} duplicate lines")
    
    return '\n'.join(cleaned_lines), removed_count



def verify_markers(text: str) -> dict:
    """
    ตรวจสอบและนับจำนวน [DUP] markers ในข้อความ
    
    Args:
        text: ข้อความที่ต้องการตรวจสอบ
        
    Returns:
        dict ที่มีข้อมูล:
        - has_dup_markers: bool - มี [DUP] markers หรือไม่
        - dup_count: int - จำนวน [DUP] markers ทั้งหมด
        - marked_lines: list - บรรทัดที่มี [DUP]
        - examples: list - ตัวอย่างบรรทัดที่มี [DUP] (สูงสุด 5 บรรทัด)
        - invalid_markers: list - markers ที่ผิดรูปแบบ (ถ้ามี)
    """
    if not text:
        return {
            'has_dup_markers': False,
            'dup_count': 0,
            'marked_lines': [],
            'examples': [],
            'invalid_markers': []
        }
    
    lines = text.split('\n')
    marked_lines = []
    invalid_markers = []
    
    for i, line in enumerate(lines, 1):
        if '[DUP]' in line:
            marked_lines.append({
                'line_number': i,
                'content': line.strip()
            })
        # เช็คว่ามี marker แบบอื่นที่ไม่ถูกต้อง
        elif '[dup]' in line.lower() and '[DUP]' not in line:
            invalid_markers.append(line.strip())
    
    return {
        'has_dup_markers': len(marked_lines) > 0,
        'dup_count': len(marked_lines),
        'marked_lines': marked_lines,
        'examples': [m['content'] for m in marked_lines[:5]],  # แสดงแค่ 5 ตัวอย่างแรก
        'invalid_markers': invalid_markers
    }



def process_dedup_markers(text: str, remove: bool = True) -> tuple:
    """
    ประมวลผล [DUP] markers แบบครบวงจร
    
    Args:
        text: ข้อความที่จะประมวลผล
        remove: ถ้า True จะลบประโยคที่มี [DUP] ออก, ถ้า False จะแค่ verify
        
    Returns:
        tuple (cleaned_text, stats)
        - cleaned_text: ข้อความที่ประมวลผลแล้ว
        - stats: สถิติการลบ
    """
    # ตรวจสอบก่อน
    before_stats = verify_markers(text)
    
    if remove:
        cleaned_text, removed_count = remove_duplicates(text)
        after_stats = verify_markers(cleaned_text)
        
        stats = {
            'before': before_stats['dup_count'],
            'after': after_stats['dup_count'],
            'removed': removed_count
        }
    else:
        cleaned_text = text
        stats = {
            'before': before_stats['dup_count'],
            'after': before_stats['dup_count'],
            'removed': 0
        }
    
    return cleaned_text, stats
