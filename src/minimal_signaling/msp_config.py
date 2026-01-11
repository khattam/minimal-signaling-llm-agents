"""Configuration for MSP system."""

import os
from typing import Optional

from pydantic import BaseModel, Field


class MSPConfig(BaseModel):
    """Configuration for MSP system."""
    
    # Groq settings
    groq_api_key: Optional[str] = Field(default=None)
    groq_model: str = "llama-3.3-70b-versatile"
    rate_limit_rpm: int = 30
    
    # Judge settings
    judge_model: str = "all-MiniLM-L6-v2"
    judge_threshold: float = 0.80
    
    # Decoder settings
    default_style: str = "professional"
    
    @classmethod
    def from_env(cls) -> "MSPConfig":
        """Load configuration from environment variables."""
        return cls(
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            groq_model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            rate_limit_rpm=int(os.environ.get("GROQ_RATE_LIMIT", "30")),
            judge_model=os.environ.get("JUDGE_MODEL", "all-MiniLM-L6-v2"),
            judge_threshold=float(os.environ.get("JUDGE_THRESHOLD", "0.80")),
            default_style=os.environ.get("DEFAULT_STYLE", "professional"),
        )
