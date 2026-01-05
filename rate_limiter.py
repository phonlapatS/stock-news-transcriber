import time
import random
import math

class RateLimiter:
    """
    Centralized rate limiter to manage API request budgets and enforce delays.
    Designed to work within Google Gemini Free Tier limits.
    """
    def __init__(self, max_requests_per_video: int = 18):
        self.request_count = 0
        self.max_requests = max_requests_per_video
        self.start_time = time.time()
        self.last_request_time = 0
        
        # Free tier safety parameters
        self.min_interval = 3.0  # Minimum seconds between requests
        self.burst_threshold = 0.3  # Requests per second warning threshold
        
        print(f"🛡️ Rate Limiter initialized with budget: {max_requests_per_video} requests")

    def can_make_request(self) -> bool:
        """Check if we still have budget for more requests."""
        if self.request_count >= self.max_requests:
            print(f"⚠️ Request budget exhausted ({self.request_count}/{self.max_requests})!")
            return False
        return True
    
    def record_request(self):
        """Record that a request is being made."""
        self.request_count += 1
        self.last_request_time = time.time()
        # print(f"📊 Request recorded: {self.request_count}/{self.max_requests}")

    def get_backoff_delay(self, attempt: int, is_quota_error: bool = False) -> float:
        """
        Calculate delay for retries.
        Warning: Exponential backoff is aggressive for quota errors.
        """
        if is_quota_error:
            # 3s, 6s, 12s, 24s...
            delay = 3 * (2 ** attempt) + random.uniform(0.5, 1.5)
            print(f"⏳ Quota backoff (Attempt {attempt+1}): {delay:.2f}s")
            return delay
        
        # Standard error backoff: 1s, 2s, 4s...
        return (2 ** attempt) + random.uniform(0, 1)

    def get_smart_delay(self) -> float:
        """
        Calculate how long to sleep BEFORE the next request to stay safe.
        """
        current_time = time.time()
        elapsed_since_start = current_time - self.start_time
        
        # 1. Enforce minimum interval
        time_since_last = current_time - self.last_request_time
        base_wait = max(0, self.min_interval - time_since_last)
        
        # 2. Check overall rate (Requests Per Minute protection)
        # Prevent sustaining > 15-20 RPM
        if elapsed_since_start > 0:
            current_rate = self.request_count / elapsed_since_start
            if current_rate > self.burst_threshold: # > ~18 req/min
                print(f"🐌 Slowing down... Rate {current_rate:.2f} rps is high.")
                return base_wait + 2.0  # Add extra penalty
        
        return base_wait

    def get_status_report(self) -> str:
        return f"Requests used: {self.request_count}/{self.max_requests}"
