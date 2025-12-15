"""Configuration management for the Mediated Minimal-Signaling Architecture.

This module handles loading and validating configuration from YAML files,
providing type-safe access to all pipeline parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class CompressionConfig(BaseModel):
    """Configuration for Stage 1: Compression."""
    
    enabled: bool = True
    token_budget: int = Field(50, gt=0)
    max_recursion: int = Field(5, gt=0)
    model: str = "sshleifer/distilbart-cnn-12-6"
    
    @field_validator("token_budget")
    @classmethod
    def validate_token_budget(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("token_budget must be positive")
        return v
    
    @field_validator("max_recursion")
    @classmethod
    def validate_max_recursion(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_recursion must be positive")
        return v


class SemanticKeysConfig(BaseModel):
    """Configuration for Stage 2: Semantic Key Extraction."""
    
    enabled: bool = True
    schema_version: str = "1.0"
    extractor: str = "placeholder"  # or "llm"


class JudgeConfig(BaseModel):
    """Configuration for optional Judge verification layer."""
    
    enabled: bool = False


class LoggingConfig(BaseModel):
    """Configuration for logging and tracing."""
    
    level: str = "INFO"
    trace_dir: str = "traces"


class DashboardConfig(BaseModel):
    """Configuration for the visualization dashboard."""
    
    enabled: bool = True
    host: str = "localhost"
    port: int = Field(8080, gt=0, lt=65536)
    ws_port: int = Field(8081, gt=0, lt=65536)


class MediatorConfig(BaseModel):
    """Complete configuration for the Mediator pipeline.
    
    This is the top-level configuration object that contains all
    settings for compression, extraction, verification, and visualization.
    """
    
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    semantic_keys: SemanticKeysConfig = Field(default_factory=SemanticKeysConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "MediatorConfig":
        """Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML configuration file.
            
        Returns:
            Validated MediatorConfig instance.
            
        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the config is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        if data is None:
            raise ValueError(f"Empty config file: {path}")
        
        try:
            return cls.model_validate(data)
        except Exception as e:
            raise ValueError(f"Invalid configuration in {path}: {e}") from e
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to a dictionary for serialization."""
        return self.model_dump()
