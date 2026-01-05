"""
Dynamic threshold management based on content characteristics
Adapts deduplication thresholds based on video type (live stream, podcast, news)
"""

from typing import Dict


class DedupConfigManager:
    """Manages deduplication thresholds dynamically based on content type"""
    
    # Default thresholds for different content types
    # UPDATED: All profiles use conservative thresholds to preserve full content
    # User requirement: CLEAN must have full content for groundtruth verification
    PROFILES = {
        "live_stream": {
            "boundary": 0.98,  # Was 0.80 - too aggressive!
            "general": 0.95,   # Was 0.75 - too aggressive!
            "merge_overlap": 0.90
        },
        "podcast": {
            "boundary": 0.98,  # Was 0.85
            "general": 0.95,   # Was 0.78
            "merge_overlap": 0.90
        },
        "news": {
            "boundary": 0.98,  # Was 0.82
            "general": 0.95,   # Was 0.76
            "merge_overlap": 0.90
        },
        "default": {
            "boundary": 0.98,  # Was 0.80
            "general": 0.95,   # Was 0.75
            "merge_overlap": 0.90
        }
    }
    
    @classmethod
    def get_thresholds(cls, channel_name: str = "", video_title: str = "") -> Dict[str, float]:
        """
        Get optimal thresholds based on content type
        
        Args:
            channel_name: YouTube channel name
            video_title: Video title
        
        Returns:
            Dict with 'boundary', 'general', 'merge_overlap' thresholds
        """
        # Detect content type from channel/title
        content_lower = f"{channel_name} {video_title}".lower()
        
        if "coffee break" in content_lower or "live" in content_lower:
            profile = cls.PROFILES["live_stream"]
            print(f"  📺 Using 'live_stream' profile (boundary={profile['boundary']}, general={profile['general']})")
            return profile
        elif "podcast" in content_lower:
            profile = cls.PROFILES["podcast"]
            print(f"  🎙️ Using 'podcast' profile (boundary={profile['boundary']}, general={profile['general']})")
            return profile
        elif "ข่าว" in content_lower or "news" in content_lower:
            profile = cls.PROFILES["news"]
            print(f"  📰 Using 'news' profile (boundary={profile['boundary']}, general={profile['general']})")
            return profile
        else:
            # FALLBACK to config values
            try:
                from config import DEDUP_BOUNDARY_THRESHOLD, DEDUP_GENERAL_THRESHOLD, DEDUP_MERGE_OVERLAP_THRESHOLD
                thresholds = {
                    "boundary": DEDUP_BOUNDARY_THRESHOLD,
                    "general": DEDUP_GENERAL_THRESHOLD,
                    "merge_overlap": DEDUP_MERGE_OVERLAP_THRESHOLD
                }
                print(f"  ⚙️ Using config.py thresholds (boundary={thresholds['boundary']}, general={thresholds['general']})")
                return thresholds
            except ImportError:
                # Ultimate fallback if config not available
                profile = cls.PROFILES["default"]
                print(f"  ⚙️ Using default profile (boundary={profile['boundary']}, general={profile['general']})")
                return profile
