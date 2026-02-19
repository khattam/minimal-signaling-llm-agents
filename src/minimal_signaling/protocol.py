"""Minimal Signal Protocol (MSP) schema definitions.

MSP provides a structured, human-readable JSON format for agent-to-agent
communication. Instead of compressing natural language, MSP translates
verbose messages into compact structured signals.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ContentSection(BaseModel):
    """A section of detailed content for long messages."""
    
    title: str = Field(..., description="Section title/category")
    content: str = Field(..., description="Detailed content for this section")
    importance: Literal["critical", "high", "medium", "low"] = "medium"


class MinimalSignal(BaseModel):
    """Structured representation of agent communication.
    
    Two-tier architecture:
    - Tier 1 (summary): High-level structured metadata
    - Tier 2 (sections): Detailed content sections for complex messages
    
    Encoding strategy adapts based on message length:
    - compact (<500 tokens): Only summary, no sections
    - detailed (500-1500 tokens): Summary + sections
    - chunked (>1500 tokens): Summary + multiple detailed sections
    """
    
    model_config = {"extra": "forbid"}  # Strict validation
    
    # Protocol version for evolution
    version: str = "2.0"
    
    # TIER 1: Structural metadata (always present)
    intent: Literal[
        "ANALYZE",
        "GENERATE", 
        "EVALUATE",
        "TRANSFORM",
        "QUERY",
        "RESPOND",
        "DELEGATE",
        "REPORT"
    ]
    
    # What the action targets
    target: str
    
    # Priority level
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    
    # High-level summary (structured)
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="High-level structured summary of key information"
    )
    
    # TIER 2: Detailed content sections (for complex messages)
    sections: List[ContentSection] = Field(
        default_factory=list,
        description="Detailed content sections preserving full information"
    )
    
    # Execution constraints
    constraints: List[str] = Field(default_factory=list)
    
    # Current state information
    state: Dict[str, Any] = Field(default_factory=dict)
    
    # Encoding metadata
    encoding_strategy: Literal["compact", "detailed", "chunked"] = "compact"
    total_sections: int = Field(default=0, ge=0)
    
    # Tracing metadata
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    parent_id: Optional[str] = None


# Valid intent values for reference
VALID_INTENTS = [
    "ANALYZE",
    "GENERATE",
    "EVALUATE", 
    "TRANSFORM",
    "QUERY",
    "RESPOND",
    "DELEGATE",
    "REPORT"
]

VALID_PRIORITIES = ["low", "medium", "high", "critical"]



class JudgeResult(BaseModel):
    """Result from semantic judge evaluation."""
    
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    similarity_score: float = Field(ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)


class PipelineMetrics(BaseModel):
    """Metrics from pipeline execution."""
    
    original_tokens: int = Field(ge=0)
    signal_tokens: int = Field(ge=0)
    decoded_tokens: int = Field(ge=0)
    compression_ratio: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)


class PipelineResult(BaseModel):
    """Complete result from MSP pipeline processing."""
    
    original_text: str
    signal: MinimalSignal
    decoded_text: str
    judge: JudgeResult
    metrics: PipelineMetrics
    trace_id: str
    timestamp: datetime


# Error hierarchy
class MSPError(Exception):
    """Base exception for MSP errors."""
    pass


class EncoderError(MSPError):
    """Error during encoding NL → MSP."""
    pass


class DecoderError(MSPError):
    """Error during decoding MSP → NL."""
    pass


class JudgeError(MSPError):
    """Error during semantic evaluation."""
    pass


class RateLimitError(MSPError):
    """Rate limit exceeded for API calls."""
    pass
