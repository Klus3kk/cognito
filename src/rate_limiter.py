"""
Rate limiting implementation for Cognito API.
"""

import time
import hashlib
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from threading import Lock

from config import get_config


@dataclass
class RateLimitInfo:
    """Information about rate limit status."""
    allowed: bool
    limit: int
    remaining: int
    reset_time: float
    retry_after: Optional[int] = None


class TokenBucket:
    """Token bucket algorithm for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        with self.lock:
            now = time.time()
            
            # Add tokens based on time elapsed
            time_passed = now - self.last_refill
            new_tokens = time_passed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait before tokens are available."""
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            needed_tokens = tokens - self.tokens
            return needed_tokens / self.refill_rate


class SlidingWindowCounter:
    """Sliding window counter for rate limiting."""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = Lock()
    
    def is_allowed(self) -> Tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, remaining)."""
        with self.lock:
            now = time.time()
            
            # Remove old requests outside window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                remaining = self.max_requests - len(self.requests)
                return True, remaining
            
            return False, 0
    
    def get_reset_time(self) -> float:
        """Get time when window resets."""
        if not self.requests:
            return time.time()
        return self.requests[0] + self.window_size


class RateLimiter:
    """Main rate limiter with multiple strategies."""
    
    def __init__(self):
        self.config = get_config().security
        self.per_ip_limiters: Dict[str, SlidingWindowCounter] = {}
        self.per_user_limiters: Dict[str, TokenBucket] = {}
        self.global_limiter = TokenBucket(
            capacity=self.config.rate_limit_per_hour,
            refill_rate=self.config.rate_limit_per_hour / 3600  # per second
        )
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        self.lock = Lock()
    
    def check_rate_limit(
        self, 
        client_ip: str, 
        user_id: Optional[str] = None,
        endpoint: str = "default"
    ) -> RateLimitInfo:
        """Check if request should be rate limited."""
        if not self.config.rate_limit_enabled:
            return RateLimitInfo(
                allowed=True,
                limit=999999,
                remaining=999999,
                reset_time=time.time() + 3600
            )
        
        # Cleanup old entries periodically
        self._cleanup_if_needed()
        
        # Check global rate limit first
        if not self.global_limiter.consume():
            wait_time = self.global_limiter.get_wait_time()
            return RateLimitInfo(
                allowed=False,
                limit=self.config.rate_limit_per_hour,
                remaining=0,
                reset_time=time.time() + wait_time,
                retry_after=int(wait_time) + 1
            )
        
        # Check per-IP rate limit
        ip_key = self._hash_ip(client_ip)
        ip_allowed, ip_remaining = self._check_ip_limit(ip_key)
        
        if not ip_allowed:
            reset_time = self.per_ip_limiters[ip_key].get_reset_time()
            return RateLimitInfo(
                allowed=False,
                limit=self.config.rate_limit_per_hour,
                remaining=0,
                reset_time=reset_time,
                retry_after=int(reset_time - time.time()) + 1
            )
        
        # Check per-user rate limit if user_id provided
        if user_id:
            user_allowed = self._check_user_limit(user_id)
            if not user_allowed:
                wait_time = self.per_user_limiters[user_id].get_wait_time()
                return RateLimitInfo(
                    allowed=False,
                    limit=self.config.rate_limit_per_hour * 2,  # Higher limit for authenticated users
                    remaining=0,
                    reset_time=time.time() + wait_time,
                    retry_after=int(wait_time) + 1
                )
        
        return RateLimitInfo(
            allowed=True,
            limit=self.config.rate_limit_per_hour,
            remaining=ip_remaining,
            reset_time=self.per_ip_limiters[ip_key].get_reset_time()
        )
    
    def _hash_ip(self, ip: str) -> str:
        """Hash IP address for privacy."""
        return hashlib.sha256(ip.encode()).hexdigest()[:16]
    
    def _check_ip_limit(self, ip_key: str) -> Tuple[bool, int]:
        """Check per-IP rate limit."""
        with self.lock:
            if ip_key not in self.per_ip_limiters:
                self.per_ip_limiters[ip_key] = SlidingWindowCounter(
                    window_size=3600,  # 1 hour
                    max_requests=self.config.rate_limit_per_hour
                )
            
            return self.per_ip_limiters[ip_key].is_allowed()
    
    def _check_user_limit(self, user_id: str) -> bool:
        """Check per-user rate limit."""
        with self.lock:
            if user_id not in self.per_user_limiters:
                self.per_user_limiters[user_id] = TokenBucket(
                    capacity=self.config.rate_limit_per_hour * 2,  # Higher limit for authenticated users
                    refill_rate=self.config.rate_limit_per_hour * 2 / 3600
                )
            
            return self.per_user_limiters[user_id].consume()
    
    def _cleanup_if_needed(self):
        """Clean up old rate limiter entries."""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        with self.lock:
            # Clean up IP limiters
            expired_ips = []
            for ip_key, limiter in self.per_ip_limiters.items():
                if now - limiter.requests[-1] if limiter.requests else 0 > 3600:
                    expired_ips.append(ip_key)
            
            for ip_key in expired_ips:
                del self.per_ip_limiters[ip_key]
            
            self.last_cleanup = now
    
    def get_rate_limit_headers(self, rate_limit_info: RateLimitInfo) -> Dict[str, str]:
        """Get HTTP headers for rate limiting."""
        headers = {
            "X-RateLimit-Limit": str(rate_limit_info.limit),
            "X-RateLimit-Remaining": str(rate_limit_info.remaining),
            "X-RateLimit-Reset": str(int(rate_limit_info.reset_time))
        }
        
        if not rate_limit_info.allowed and rate_limit_info.retry_after:
            headers["Retry-After"] = str(rate_limit_info.retry_after)
        
        return headers


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def check_rate_limit(client_ip: str, user_id: Optional[str] = None) -> RateLimitInfo:
    """Convenience function to check rate limit."""
    return get_rate_limiter().check_rate_limit(client_ip, user_id)