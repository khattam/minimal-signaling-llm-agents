"""Groq API client with rate limiting for free tier."""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from groq import Groq

from .protocol import RateLimitError


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, requests_per_minute: int = 30):
        self.rpm = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Wait until we can make another request."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_request
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            self.last_request = time.time()


class GroqClient:
    """Client for Groq API with rate limiting for free tier."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        requests_per_minute: int = 30
    ):
        """Initialize Groq client.
        
        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
            model: Model to use for inference.
            requests_per_minute: Rate limit (free tier is 30 rpm).
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY env var or pass api_key."
            )
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.rate_limiter = RateLimiter(requests_per_minute)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.0
    ) -> str:
        """Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            json_mode: If True, request JSON output format.
            temperature: Sampling temperature (0.0 for deterministic).
            
        Returns:
            The assistant's response content.
            
        Raises:
            RateLimitError: If rate limit is exceeded.
        """
        await self.rate_limiter.acquire()
        
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
    
    def chat_sync(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.0
    ) -> str:
        """Synchronous version of chat for non-async contexts."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
