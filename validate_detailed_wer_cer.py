#!/usr/bin/env python3
"""
Comprehensive WER/CER Validation Report
Calculate detailed error metrics for multiple transcripts
"""

import re
from pathlib import Path
from typing import Tuple, List, Dict
import json

def tokenize_thai(text: str) -> List[str]:
    """Tokenize Thai text into words"""
    text = re.sub(r'\s+', ' ', text.strip())
    return text.split()

def calculate_levenshtein_detailed(ref: List[str], hyp: List[str]) -> Tuple[int, Dict]:
    """
    Calculate Levenshtein distance with detailed error breakdown
    Returns: (total_distance, error_breakdown)
    """
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ops = [[None] * (n + 1) for _ in range(m + 1)]
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
        ops[i][0] = 'del'
    for j in range(n + 1):
        dp[0][j] = j
        ops[0][j] = 'ins'
    ops[0][0] = None
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
                ops[i][j] = 'match'
            else:
                costs = {
                    'del': dp[i-1][j],
                    'ins': dp[i][j-1],
                    'sub': dp[i-1][j-1]
                }
                min_op = min(costs, key=costs.get)
                dp[i][j] = 1 + costs[min_op]
                ops[i][j] = min_op
    
    # Backtrack to count error types
    i, j = m, n
    errors = {'substitution': 0, 'deletion': 0, 'insertion': 0}
    
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == 'match':
            i -= 1
            j -= 1
        elif op == 'sub':
            errors['substitution'] += 1
            i -= 1
            j -= 1
        elif op == 'del':
            errors['deletion'] += 1
            i -= 1
        elif op == 'ins':
            errors['insertion'] += 1
            j -= 1
        else:
            break
    
    return dp[m][n], errors

def analyze_errors(reference: str, hypothesis: str) -> Dict:
    """Comprehensive error analysis"""
    ref_words = tokenize_thai(reference)
    hyp_words = tokenize_thai(hypothesis)
    
    # Word-level analysis
    distance, word_errors = calculate_levenshtein_detailed(ref_words, hyp_words)
    wer = (distance / len(ref_words) * 100) if len(ref_words) > 0 else 0.0
    
    # Character-level analysis
    ref_chars = list(reference.replace(' ', '').replace('\r', '').replace('\n', ''))
    hyp_chars = list(hypothesis.replace(' ', '').replace('\r', '').replace('\n', ''))
    char_dist, char_errors = calculate_levenshtein_detailed(ref_chars, hyp_chars)
    cer = (char_dist / len(ref_chars) * 100) if len(ref_chars) > 0 else 0.0
    
    return {
        'wer': round(wer, 2),
        'cer': round(cer, 2),
        'word_errors': word_errors,
        'char_errors': char_errors,
        'ref_words': len(ref_words),
        'hyp_words': len(hyp_words),
        'ref_chars': len(ref_chars),
        'hyp_chars': len(hyp_chars),
        'total_word_distance': distance,
        'total_char_distance': char_dist
    }

def diagnose_errors(metrics: Dict) -> Dict:
    """Diagnose root causes of errors"""
    diagnoses = []
    
    # Word errors
    total_word_errors = sum(metrics['word_errors'].values())
    
    if metrics['word_errors']['substitution'] > total_word_errors * 0.5:
        diagnoses.append("ASR confusion: Many word substitutions (เสียงคล้ายกัน)")
    
    if metrics['word_errors']['deletion'] > total_word_errors * 0.3:
        diagnoses.append("Missing words: High deletion rate (ASR ไม่จับเสียง)")
    
    if metrics['word_errors']['insertion'] > total_word_errors * 0.3:
        diagnoses.append("Extra words: High insertion rate (ASR จับเสียงผิด)")
    
    # Overall quality
    if metrics['wer'] < 5:
        quality = "Excellent"
    elif metrics['wer'] < 10:
        quality = "Good"
    elif metrics['wer'] < 20:
        quality = "Fair"
    else:
        quality = "Needs Improvement"
        diagnoses.append("High error rate: Requires manual review")
    
    return {
        'quality': quality,
        'diagnoses': diagnoses if diagnoses else ["Good quality - minimal errors"],
        'error_distribution': {
            'substitution': round(metrics['word_errors']['substitution'] / max(1, total_word_errors) * 100, 1),
            'deletion': round(metrics['word_errors']['deletion'] / max(1, total_word_errors) * 100, 1),
            'insertion': round(metrics['word_errors']['insertion'] / max(1, total_word_errors) * 100, 1)
        }
    }

def main():
    base_dir = Path(__file__).parent / "transcripts_output"
    
    # Target files from user request
    target_dates = [
        "01122568", "30092568", "27102568", "12092568", "21102568",
        "06112568", "27112568", "24112568", "26112568", "16092568"
    ]
    
    results = []
    
    print("📊 Comprehensive WER/CER Validation Report")
    print("=" * 80)
    
    for date in target_dates:
        # Find matching files - search for both correct and typo spellings!
        groundtruth_files = []
        for pattern in [f"วิเคราะห์หุ้นรายวัน__{date}_groundtruth_*.txt", 
                       f"วิเคราะห์หุ้นรายวัน__{date}_groudtruth_*.txt"]:  # Handle typo!
            groundtruth_files.extend(list(base_dir.glob(pattern)))
        
        clean_pattern = f"วิเคราะห์หุ้นรายวัน__{date}_CLEAN_*.txt"
        clean_files = list(base_dir.glob(clean_pattern))
        
        if not groundtruth_files or not clean_files:
            print(f"\n⚠️  {date}: Files not found")
            continue
        
        gt_file = groundtruth_files[0]
        clean_file = clean_files[0]
        
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                groundtruth = f.read()
            with open(clean_file, 'r', encoding='utf-8') as f:
                clean_text = f.read()
            
            # Analyze
            metrics = analyze_errors(groundtruth, clean_text)
            diagnosis = diagnose_errors(metrics)
            
            results.append({
                'date': date,
                'metrics': metrics,
                'diagnosis': diagnosis
            })
            
            print(f"\n✅ {date}: WER={metrics['wer']}%, CER={metrics['cer']}%")
            
        except Exception as e:
            print(f"\n❌ {date}: Error - {e}")
    
    # Generate report
    print("\n" + "=" * 80)
    print("📄 Generating detailed report...")
    
    report_lines = []
    report_lines.append("# WER/CER Comprehensive Validation Report\n")
    report_lines.append(f"**Date:** {Path(__file__).parent.name}\n")
    report_lines.append(f"**Total Files Analyzed:** {len(results)}\n\n")
    
    # Summary table
    report_lines.append("## Summary Table\n\n")
    report_lines.append("| Date | WER (%) | CER (%) | Quality | Words (Ref) | Errors |\n")
    report_lines.append("|------|---------|---------|---------|-------------|--------|\n")
    
    for r in results:
        date = r['date']
        m = r['metrics']
        d = r['diagnosis']
        total_errors = sum(m['word_errors'].values())
        report_lines.append(
            f"| {date} | {m['wer']} | {m['cer']} | {d['quality']} | {m['ref_words']} | {total_errors} |\n"
        )
    
    # Detailed analysis
    report_lines.append("\n## Detailed Error Analysis\n\n")
    
    for r in results:
        date = r['date']
        m = r['metrics']
        d = r['diagnosis']
        
        report_lines.append(f"### วิเคราะห์หุ้นรายวัน: {date}\n\n")
        
        # Metrics
        report_lines.append(f"**Word Error Rate (WER):** {m['wer']}%\n")
        report_lines.append(f"**Character Error Rate (CER):** {m['cer']}%\n")
        report_lines.append(f"**Quality Assessment:** {d['quality']}\n\n")
        
        # Error breakdown
        report_lines.append("**Error Breakdown:**\n")
        report_lines.append(f"- Substitutions: {m['word_errors']['substitution']} ({d['error_distribution']['substitution']}%)\n")
        report_lines.append(f"- Deletions: {m['word_errors']['deletion']} ({d['error_distribution']['deletion']}%)\n")
        report_lines.append(f"- Insertions: {m['word_errors']['insertion']} ({d['error_distribution']['insertion']}%)\n\n")
        
        # Diagnosis
        report_lines.append("**Root Causes:**\n")
        for diag in d['diagnoses']:
            report_lines.append(f"- {diag}\n")
        report_lines.append("\n")
    
    # Write report
    report_path = Path(__file__).parent / ".gemini" / "antigravity" / "brain" / "bfa385ed-605e-463e-a862-609678a79a02" / "wer_cer_detailed_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print(f"✅ Report saved to: {report_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
