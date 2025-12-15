"""Trace logging for pipeline execution analysis."""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

from .models import TraceRecord, MediatorResult
from .config import MediatorConfig


class TraceLogger:
    """Logs complete pipeline traces to JSONL format.
    
    Each trace includes:
    - Original message metadata
    - Compression steps and token counts
    - Extracted semantic keys
    - Judge results (if enabled)
    - Timing information
    """
    
    def __init__(self, trace_dir: str = "traces"):
        """Initialize the trace logger.
        
        Args:
            trace_dir: Directory to write trace files
        """
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
    
    def log_trace(
        self,
        message_id: str,
        original_text: str,
        original_tokens: int,
        result: MediatorResult,
        config: MediatorConfig
    ) -> Path:
        """Log a complete pipeline trace.
        
        Args:
            message_id: Unique message identifier
            original_text: Original input message
            original_tokens: Token count of original message
            result: Pipeline execution result
            config: Configuration snapshot
            
        Returns:
            Path to the trace file
        """
        # Create trace record
        trace = TraceRecord(
            message_id=message_id,
            timestamp=result.timestamp,
            original_text=original_text,
            original_tokens=original_tokens,
            compression=result.compression,
            extraction=result.extraction,
            judge=result.judge,
            duration_ms=result.duration_ms,
            config_snapshot=self._config_to_dict(config)
        )
        
        # Write to JSONL file
        trace_file = self.trace_dir / f"trace_{message_id}.jsonl"
        
        with open(trace_file, "w", encoding="utf-8") as f:
            # Write as single-line JSON
            json.dump(trace.model_dump(mode="json"), f)
            f.write("\n")
        
        return trace_file
    
    def log_trace_from_result(
        self,
        original_text: str,
        original_tokens: int,
        result: MediatorResult,
        config: MediatorConfig,
        message_id: str | None = None
    ) -> Path:
        """Convenience method to log trace with auto-generated message ID.
        
        Args:
            original_text: Original input message
            original_tokens: Token count of original message
            result: Pipeline execution result
            config: Configuration snapshot
            message_id: Optional message ID (generated if not provided)
            
        Returns:
            Path to the trace file
        """
        if message_id is None:
            message_id = str(uuid.uuid4())
        
        return self.log_trace(
            message_id=message_id,
            original_text=original_text,
            original_tokens=original_tokens,
            result=result,
            config=config
        )
    
    def _config_to_dict(self, config: MediatorConfig) -> Dict[str, Any]:
        """Convert config to dictionary for trace snapshot.
        
        Args:
            config: Mediator configuration
            
        Returns:
            Dictionary representation of config
        """
        return {
            "compression": {
                "enabled": config.compression.enabled,
                "token_budget": config.compression.token_budget,
                "max_recursion": config.compression.max_recursion,
                "model": config.compression.model
            },
            "semantic_keys": {
                "enabled": config.semantic_keys.enabled,
                "schema_version": config.semantic_keys.schema_version,
                "extractor": config.semantic_keys.extractor
            },
            "judge": {
                "enabled": config.judge.enabled
            }
        }
    
    def read_trace(self, message_id: str) -> TraceRecord:
        """Read a trace record from file.
        
        Args:
            message_id: Message ID to read
            
        Returns:
            TraceRecord
            
        Raises:
            FileNotFoundError: If trace file doesn't exist
        """
        trace_file = self.trace_dir / f"trace_{message_id}.jsonl"
        
        with open(trace_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
            return TraceRecord(**data)
    
    def list_traces(self) -> list[str]:
        """List all trace message IDs.
        
        Returns:
            List of message IDs
        """
        traces = []
        for trace_file in self.trace_dir.glob("trace_*.jsonl"):
            # Extract message ID from filename
            message_id = trace_file.stem.replace("trace_", "")
            traces.append(message_id)
        return sorted(traces)
