"""
Calculate WER/CER for specific groundtruth files
Compares RAW ASR output against user-corrected groundtruth
"""

import re
from pathlib import Path
import glob


def tokenize_thai(text):
    """Tokenize Thai text into words"""
    text = re.sub(r'\s+', ' ', text.strip())
    return text.split()


def levenshtein_distance(ref, hyp):
    """Calculate edit distance"""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    return dp[m][n]


def calculate_wer_cer(reference, hypothesis):
    """Calculate WER and CER"""
    # Word-level
    ref_words = tokenize_thai(reference)
    hyp_words = tokenize_thai(hypothesis)
    word_distance = levenshtein_distance(ref_words, hyp_words)
    wer = (word_distance / len(ref_words) * 100) if len(ref_words) > 0 else 0
    
    # Character-level
    ref_chars = reference.replace(' ', '').replace('\r', '').replace('\n', '')
    hyp_chars = hypothesis.replace(' ', '').replace('\r', '').replace('\n', '')
    char_distance = levenshtein_distance(list(ref_chars), list(hyp_chars))
    cer = (char_distance / len(ref_chars) * 100) if len(ref_chars) > 0 else 0
    
    return {
        'wer': round(wer, 2),
        'cer': round(cer, 2),
        'word_errors': word_distance,
        'total_words': len(ref_words),
        'char_errors': char_distance,
        'total_chars': len(ref_chars)
    }


def find_clean_file(groundtruth_file):
    """Find corresponding CLEAN file by replacing _groundtruth_ with _CLEAN_"""
    clean_file = Path(groundtruth_file.replace('_groundtruth_', '_CLEAN_'))
    return clean_file if clean_file.exists() else None


def main():
    # Files to check
    groundtruth_files = [
        r"ตรวจงาน\[Live] Coffee Break เช็คหุ้นบ่าย สายเทคนิค ประจำวันที่ 23 ธ.ค. 2568_groundtruth_20251228_1013.txt",
        r"ตรวจงาน\วิเคราะห์หุ้นรายวัน__06052568_groundtruth_20251228_1116.txt",
        r"ตรวจงาน\วิเคราะห์หุ้นรายตัว__16062568_groundtruth_20251228_1121.txt",
        r"ตรวจงาน\วิเคราะห์หุ้นรายวัน__22042568_groundtruth_20251227_1636.txt"
    ]
    
    print("\n📊 WER/CER Calculation Results\n")
    print("="*80)
    
    results = []
    
    for gt_file in groundtruth_files:
        filename = Path(gt_file).name
        print(f"\n📄 {filename}")
        
        # Find CLEAN file
        clean_file = find_clean_file(gt_file)
        
        if not clean_file:
            print(f"   ❌ CLEAN file not found")
            continue
        
        print(f"   Comparing: {clean_file.name}")
        
        try:
            # Read files
            with open(gt_file, 'r', encoding='utf-8') as f:
                reference = f.read()
            
            with open(clean_file, 'r', encoding='utf-8') as f:
                hypothesis = f.read()
            
            # Calculate metrics
            metrics = calculate_wer_cer(reference, hypothesis)
            results.append({
                'file': filename,
                'metrics': metrics
            })
            
            # Display
            print(f"   WER: {metrics['wer']:.2f}% ({metrics['word_errors']}/{metrics['total_words']} words)")
            print(f"   CER: {metrics['cer']:.2f}% ({metrics['char_errors']}/{metrics['total_chars']} chars)")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    if results:
        print("\n" + "="*80)
        print("\n📈 SUMMARY\n")
        
        avg_wer = sum(r['metrics']['wer'] for r in results) / len(results)
        avg_cer = sum(r['metrics']['cer'] for r in results) / len(results)
        
        print(f"Files evaluated: {len(results)}")
        print(f"Average WER: {avg_wer:.2f}%")
        print(f"Average CER: {avg_cer:.2f}%")
        
        # Quality assessment
        print(f"\nQuality Assessment:")
        if avg_wer < 10:
            print(f"  WER: Excellent (< 10%)")
        elif avg_wer < 20:
            print(f"  WER: Good (10-20%)")
        else:
            print(f"  WER: Needs Improvement (> 20%)")
        
        if avg_cer < 5:
            print(f"  CER: Excellent (< 5%)")
        elif avg_cer < 10:
            print(f"  CER: Good (5-10%)")
        else:
            print(f"  CER: Needs Improvement (> 10%)")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()
