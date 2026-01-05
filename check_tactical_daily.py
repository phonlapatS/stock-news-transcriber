"""Quick WER/CER for Tactical Daily SPDR Gold"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from validate_detailed_wer_cer import analyze_errors, diagnose_errors

# Read files
gt_file = r'transcripts_output/ตรวจงาน/Tactical_Daily_28_พย_68_LIVE_SPDR_Gold_กองทุนทองคำ_groundtruth_20251224_2157.txt'
clean_file = r'transcripts_output/ตรวจงาน/Tactical_Daily_28_พย_68_LIVE_SPDR_Gold_กองทุนทองคำ_CLEAN_20251224_2157.txt'

with open(gt_file, 'r', encoding='utf-8') as f:
    groundtruth = f.read()

with open(clean_file, 'r', encoding='utf-8') as f:
    clean = f.read()

print("Analyzing Tactical Daily (SPDR Gold)...")
metrics = analyze_errors(groundtruth, clean)
diagnosis = diagnose_errors(metrics)

print('='*70)
print('WER/CER Analysis: Tactical Daily - SPDR Gold (28 Nov 2568)')
print('='*70)
print(f'WER (Word Error Rate): {metrics["wer"]:.2f}%')
print(f'CER (Character Error Rate): {metrics["cer"]:.2f}%')
print(f'Quality Assessment: {diagnosis["quality"]}')
print()
print('Word-level Errors:')
print(f'  - Substitutions: {metrics["word_errors"]["substitution"]}')
print(f'  - Deletions: {metrics["word_errors"]["deletion"]}')
print(f'  - Insertions: {metrics["word_errors"]["insertion"]}')
print()
print(f'Reference Text: {metrics["ref_words"]} words, {metrics["ref_chars"]} chars')
print(f'Hypothesis Text: {metrics["hyp_words"]} words, {metrics["hyp_chars"]} chars')
print(f'Word difference: {metrics["hyp_words"] - metrics["ref_words"]:+d}')
print()
if diagnosis['diagnoses']:
    print('Diagnosis:')
    for d in diagnosis['diagnoses']:
        print(f'  • {d}')
print('='*70)
