# asr_error_logger.py
"""
ASR Error Logger - Continuous Learning System
Logs ASR errors found in transcripts and builds knowledge base for prompt improvement
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

class ErrorPattern:
    """Represents a single ASR error pattern"""
    def __init__(self, raw: str, corrected: str, context: str = "", 
                 error_id: str = None, frequency: int = 1, video_ids: List[str] = None):
        self.error_id = error_id or f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.raw = raw
        self.corrected = corrected
        self.context = context
        self.frequency = frequency
        self.video_ids = video_ids or []
        self.added_date = datetime.now().isoformat()
        self.last_seen = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "id": self.error_id,
            "raw": self.raw,
            "corrected": self.corrected,
            "context": self.context,
            "frequency": self.frequency,
            "video_ids": self.video_ids,
            "added_date": self.added_date,
            "last_seen": self.last_seen
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'ErrorPattern':
        ep = ErrorPattern(
            raw=data["raw"],
            corrected=data["corrected"],
            context=data.get("context", ""),
            error_id=data.get("id"),
            frequency=data.get("frequency", 1),
            video_ids=data.get("video_ids", [])
        )
        ep.added_date = data.get("added_date", ep.added_date)
        ep.last_seen = data.get("last_seen", ep.last_seen)
        return ep

class ASRErrorLogger:
    """Manages ASR error logging and retrieval"""
    
    def __init__(self, log_file: str = "asr_errors.json"):
        self.log_file = Path(log_file)
        self.errors: List[ErrorPattern] = []
        self.load()
    
    def load(self):
        """Load existing errors from file"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.errors = [ErrorPattern.from_dict(e) for e in data.get("errors", [])]
                print(f"✅ Loaded {len(self.errors)} error patterns from {self.log_file}")
            except Exception as e:
                print(f"⚠️ Error loading {self.log_file}: {e}")
                self.errors = []
        else:
            print(f"📝 Creating new error log: {self.log_file}")
            self.errors = []
    
    def save(self):
        """Save errors to file"""
        try:
            data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "total_errors": len(self.errors),
                "errors": [e.to_dict() for e in self.errors]
            }
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Saved {len(self.errors)} errors to {self.log_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving to {self.log_file}: {e}")
            return False
    
    def log_error(self, raw: str, corrected: str, context: str = "", video_id: str = "") -> bool:
        """
        Log a new error or update existing one
        Returns True if new error added, False if existing error updated
        """
        # Check if error already exists (same raw text)
        existing = self.find_error(raw)
        
        if existing:
            # Update frequency and video_ids
            existing.frequency += 1
            existing.last_seen = datetime.now().isoformat()
            if video_id and video_id not in existing.video_ids:
                existing.video_ids.append(video_id)
            
            # Update corrected text if different (keep latest)
            if existing.corrected != corrected:
                print(f"⚠️ Updating correction: '{raw}' from '{existing.corrected}' → '{corrected}'")
                existing.corrected = corrected
            
            self.save()
            print(f"📊 Updated existing error: '{raw}' → '{corrected}' (frequency: {existing.frequency})")
            return False
        else:
            # Add new error
            new_error = ErrorPattern(
                raw=raw,
                corrected=corrected,
                context=context,
                video_ids=[video_id] if video_id else []
            )
            self.errors.append(new_error)
            self.save()
            print(f"✨ New error logged: '{raw}' → '{corrected}'")
            return True
    
    def find_error(self, raw: str) -> Optional[ErrorPattern]:
        """Find error by raw text"""
        for error in self.errors:
            if error.raw == raw:
                return error
        return None
    
    def get_all_errors(self) -> List[ErrorPattern]:
        """Get all errors"""
        return self.errors
    
    def get_top_errors(self, limit: int = 20, min_frequency: int = 1) -> List[ErrorPattern]:
        """Get top N most frequent errors"""
        filtered = [e for e in self.errors if e.frequency >= min_frequency]
        sorted_errors = sorted(filtered, key=lambda x: x.frequency, reverse=True)
        return sorted_errors[:limit]
    
    def get_errors_by_context(self, context: str) -> List[ErrorPattern]:
        """Get errors filtered by context"""
        return [e for e in self.errors if context.lower() in e.context.lower()]
    
    def get_error_examples_for_prompt(self, limit: int = 15, format: str = "bullet") -> str:
        """
        Get formatted error examples for prompt injection
        
        Args:
            limit: Max number of examples
            format: "bullet" or "table"
        """
        top_errors = self.get_top_errors(limit=limit)
        
        if not top_errors:
            return ""
        
        if format == "bullet":
            lines = []
            for err in top_errors:
                freq_str = f" (seen {err.frequency}x)" if err.frequency > 1 else ""
                lines.append(f"- {err.raw} → {err.corrected}{freq_str}")
            return "\n".join(lines)
        
        elif format == "table":
            lines = ["| Raw | Corrected | Frequency |", "|-----|-----------|-----------|"]
            for err in top_errors:
                lines.append(f"| {err.raw} | {err.corrected} | {err.frequency} |")
            return "\n".join(lines)
        
        return ""
    
    def get_stats(self) -> dict:
        """Get statistics about logged errors"""
        if not self.errors:
            return {"total": 0}
        
        total_frequency = sum(e.frequency for e in self.errors)
        
        return {
            "total_unique_errors": len(self.errors),
            "total_occurrences": total_frequency,
            "avg_frequency": total_frequency / len(self.errors),
            "most_common": self.get_top_errors(limit=5),
            "contexts": list(set(e.context for e in self.errors if e.context))
        }

# Global instance
_logger = None

def get_logger() -> ASRErrorLogger:
    """Get global logger instance"""
    global _logger
    if _logger is None:
        _logger = ASRErrorLogger()
    return _logger

if __name__ == "__main__":
    # Test
    logger = ASRErrorLogger()
    
    # Test logging
    logger.log_error("โซดี", "สวัสดี", "greeting")
    logger.log_error("คิวโต", "QoQ", "financial")
    logger.log_error("ยูทูปเวษา", "ยูโซเฟสสาม", "financial")
    
    # Test retrieval
    print("\n📊 Stats:", logger.get_stats())
    print("\n📝 Examples for prompt:")
    print(logger.get_error_examples_for_prompt())
