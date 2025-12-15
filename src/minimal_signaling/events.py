"""Event system for real-time pipeline updates."""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class PipelineEvent(str, Enum):
    """Events emitted during pipeline execution."""
    
    MESSAGE_RECEIVED = "message_received"
    COMPRESSION_START = "compression_start"
    COMPRESSION_PASS = "compression_pass"
    COMPRESSION_COMPLETE = "compression_complete"
    EXTRACTION_START = "extraction_start"
    EXTRACTION_COMPLETE = "extraction_complete"
    JUDGE_START = "judge_start"
    JUDGE_COMPLETE = "judge_complete"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_ERROR = "pipeline_error"


class EventPayload(BaseModel):
    """Payload for pipeline events."""
    
    event: PipelineEvent
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"arbitrary_types_allowed": True}


# Type alias for event handlers
EventHandler = Callable[[EventPayload], None]
AsyncEventHandler = Callable[[EventPayload], Any]  # Can be coroutine


class AsyncEventEmitter:
    """Async event emitter for pipeline events.
    
    Supports both sync and async handlers. Events are emitted
    asynchronously to avoid blocking the pipeline.
    """
    
    def __init__(self):
        """Initialize the event emitter."""
        self._handlers: Dict[PipelineEvent, List[AsyncEventHandler]] = {}
        self._global_handlers: List[AsyncEventHandler] = []
    
    def on(self, event: PipelineEvent, handler: AsyncEventHandler) -> None:
        """Register a handler for a specific event.
        
        Args:
            event: Event type to listen for
            handler: Handler function (sync or async)
        """
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)
    
    def on_all(self, handler: AsyncEventHandler) -> None:
        """Register a handler for all events.
        
        Args:
            handler: Handler function (sync or async)
        """
        self._global_handlers.append(handler)
    
    def off(self, event: PipelineEvent, handler: AsyncEventHandler) -> None:
        """Remove a handler for a specific event.
        
        Args:
            event: Event type
            handler: Handler to remove
        """
        if event in self._handlers and handler in self._handlers[event]:
            self._handlers[event].remove(handler)
    
    def off_all(self, handler: AsyncEventHandler) -> None:
        """Remove a global handler.
        
        Args:
            handler: Handler to remove
        """
        if handler in self._global_handlers:
            self._global_handlers.remove(handler)
    
    async def emit(self, payload: EventPayload) -> None:
        """Emit an event to all registered handlers.
        
        Args:
            payload: Event payload
        """
        handlers = list(self._global_handlers)
        
        if payload.event in self._handlers:
            handlers.extend(self._handlers[payload.event])
        
        # Execute all handlers
        for handler in handlers:
            try:
                result = handler(payload)
                # If handler is async, await it
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                # Don't let handler errors break the pipeline
                pass
    
    def emit_sync(self, payload: EventPayload) -> None:
        """Emit an event synchronously (for non-async contexts).
        
        Creates a new event loop if needed.
        
        Args:
            payload: Event payload
        """
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, schedule the emit
            asyncio.create_task(self.emit(payload))
        except RuntimeError:
            # No running loop, run synchronously
            asyncio.run(self.emit(payload))


class SyncEventEmitter:
    """Synchronous event emitter for simpler use cases.
    
    All handlers are called synchronously in order.
    """
    
    def __init__(self):
        """Initialize the event emitter."""
        self._handlers: Dict[PipelineEvent, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
    
    def on(self, event: PipelineEvent, handler: EventHandler) -> None:
        """Register a handler for a specific event."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)
    
    def on_all(self, handler: EventHandler) -> None:
        """Register a handler for all events."""
        self._global_handlers.append(handler)
    
    def off(self, event: PipelineEvent, handler: EventHandler) -> None:
        """Remove a handler for a specific event."""
        if event in self._handlers and handler in self._handlers[event]:
            self._handlers[event].remove(handler)
    
    def emit(self, payload: EventPayload) -> None:
        """Emit an event to all registered handlers."""
        handlers = list(self._global_handlers)
        
        if payload.event in self._handlers:
            handlers.extend(self._handlers[payload.event])
        
        for handler in handlers:
            try:
                handler(payload)
            except Exception:
                # Don't let handler errors break the pipeline
                pass


# Helper functions for creating event payloads
def create_message_received_event(
    message: str,
    token_count: int
) -> EventPayload:
    """Create a MESSAGE_RECEIVED event."""
    return EventPayload(
        event=PipelineEvent.MESSAGE_RECEIVED,
        data={
            "message": message,
            "token_count": token_count
        }
    )


def create_compression_start_event(
    input_tokens: int,
    budget: int
) -> EventPayload:
    """Create a COMPRESSION_START event."""
    return EventPayload(
        event=PipelineEvent.COMPRESSION_START,
        data={
            "input_tokens": input_tokens,
            "budget": budget
        }
    )


def create_compression_pass_event(
    pass_number: int,
    input_tokens: int,
    output_tokens: int,
    ratio: float
) -> EventPayload:
    """Create a COMPRESSION_PASS event."""
    return EventPayload(
        event=PipelineEvent.COMPRESSION_PASS,
        data={
            "pass_number": pass_number,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ratio": ratio
        }
    )


def create_compression_complete_event(
    original_tokens: int,
    final_tokens: int,
    passes: int,
    total_ratio: float
) -> EventPayload:
    """Create a COMPRESSION_COMPLETE event."""
    return EventPayload(
        event=PipelineEvent.COMPRESSION_COMPLETE,
        data={
            "original_tokens": original_tokens,
            "final_tokens": final_tokens,
            "passes": passes,
            "total_ratio": total_ratio
        }
    )


def create_extraction_start_event() -> EventPayload:
    """Create an EXTRACTION_START event."""
    return EventPayload(
        event=PipelineEvent.EXTRACTION_START,
        data={}
    )


def create_extraction_complete_event(
    key_count: int,
    schema_version: str
) -> EventPayload:
    """Create an EXTRACTION_COMPLETE event."""
    return EventPayload(
        event=PipelineEvent.EXTRACTION_COMPLETE,
        data={
            "key_count": key_count,
            "schema_version": schema_version
        }
    )


def create_judge_start_event() -> EventPayload:
    """Create a JUDGE_START event."""
    return EventPayload(
        event=PipelineEvent.JUDGE_START,
        data={}
    )


def create_judge_complete_event(
    passed: bool,
    confidence: float,
    issue_count: int
) -> EventPayload:
    """Create a JUDGE_COMPLETE event."""
    return EventPayload(
        event=PipelineEvent.JUDGE_COMPLETE,
        data={
            "passed": passed,
            "confidence": confidence,
            "issue_count": issue_count
        }
    )


def create_pipeline_complete_event(
    success: bool,
    duration_ms: float
) -> EventPayload:
    """Create a PIPELINE_COMPLETE event."""
    return EventPayload(
        event=PipelineEvent.PIPELINE_COMPLETE,
        data={
            "success": success,
            "duration_ms": duration_ms
        }
    )


def create_pipeline_error_event(
    stage: str,
    error_type: str,
    message: str
) -> EventPayload:
    """Create a PIPELINE_ERROR event."""
    return EventPayload(
        event=PipelineEvent.PIPELINE_ERROR,
        data={
            "stage": stage,
            "error_type": error_type,
            "message": message
        }
    )
