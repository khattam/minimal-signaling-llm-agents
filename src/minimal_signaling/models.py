"""Core data models for the Mediated Minimal-Signaling Architecture.

This module defines the data structures used throughout the pipeline:
- SemanticKey: Structured symbolic units for agent communication
- CompressionResult: Output from Stage 1 compression
- ExtractionResult: Output from Stage 2 semantic key extraction
- JudgeResult: Output from optional verification layer
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class KeyType(str, Enum):
    """Types of semantic keys that can be extracted from messages."""
    
    INSTRUCTION = "INSTRUCTION"
    STATE = "STATE"
    GOAL = "GOAL"
    CONTEXT = "CONTEXT"
    CONSTRAINT = "CONSTRAINT"


class SemanticKey(BaseModel):
    """A semantic key representing a structured unit of meaning.
    
    Semantic keys are the output of Stage 2 extraction, providing
    a stable symbolic representation that can be reliably parsed
    across different LLM implementations.
    """
    
    type: KeyType
    value: str = Field(..., min_length=1)
    
    model_config = {"frozen": True}


class CompressionStep(BaseModel):
    """Record of a single compression pass."""
    
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    input_text: str
    output_text: str
    
    @property
    def ratio(self) -> float:
        """Compression ratio (output/input). Lower is better."""
        if self.input_tokens == 0:
            return 1.0
        return self.output_tokens / self.input_tokens


class CompressionResult(BaseModel):
    """Result from Stage 1 compression pipeline.
    
    Contains the final compressed text along with metadata about
    the compression process for analysis and debugging.
    """
    
    compressed_text: str
    original_tokens: int = Field(..., ge=0)
    final_tokens: int = Field(..., ge=0)
    passes: int = Field(..., ge=0)
    log: list[CompressionStep] = Field(default_factory=list)
    
    @property
    def total_ratio(self) -> float:
        """Overall compression ratio (final/original). Lower is better."""
        if self.original_tokens == 0:
            return 1.0
        return self.final_tokens / self.original_tokens
    
    @property
    def budget_met(self) -> bool:
        """Whether compression achieved the target (set externally)."""
        return True  # Determined by caller based on budget


class ExtractionResult(BaseModel):
    """Result from Stage 2 semantic key extraction.
    
    Contains the extracted semantic keys along with schema version
    and raw output for debugging purposes.
    """
    
    keys: list[SemanticKey] = Field(default_factory=list)
    schema_version: str = Field(default="1.0")
    raw_output: str = Field(default="")
    
    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if not v:
            raise ValueError("schema_version cannot be empty")
        return v


class JudgeResult(BaseModel):
    """Result from optional Judge verification layer.
    
    Evaluates whether semantic keys faithfully represent
    the original message content.
    """
    
    passed: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)


class PipelineError(BaseModel):
    """Error information from pipeline execution."""
    
    stage: str
    error_type: str
    message: str
    recoverable: bool = False


class MediatorResult(BaseModel):
    """Complete result from the Mediator pipeline.
    
    Contains all outputs from each stage along with metadata
    for the receiving agent.
    """
    
    success: bool
    compression: CompressionResult | None = None
    extraction: ExtractionResult | None = None
    judge: JudgeResult | None = None
    error: PipelineError | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: float = Field(default=0.0, ge=0.0)


class TraceRecord(BaseModel):
    """Complete trace record for a pipeline run.
    
    Used for logging and analysis of pipeline behavior.
    """
    
    message_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    original_text: str
    original_tokens: int = Field(..., ge=0)
    compression: CompressionResult | None = None
    extraction: ExtractionResult | None = None
    judge: JudgeResult | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
