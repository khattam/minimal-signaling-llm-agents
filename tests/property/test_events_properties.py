"""Property-based tests for event system."""

from hypothesis import given, strategies as st, settings
import pytest

from minimal_signaling.events import (
    PipelineEvent,
    EventPayload,
    SyncEventEmitter,
    AsyncEventEmitter,
    create_message_received_event,
    create_compression_start_event,
    create_compression_complete_event,
    create_extraction_complete_event,
    create_pipeline_complete_event,
)
from minimal_signaling.mediator import Mediator
from minimal_signaling.config import MediatorConfig, CompressionConfig, SemanticKeysConfig, JudgeConfig
from minimal_signaling.tokenization import TiktokenTokenizer
from minimal_signaling.extraction import PlaceholderExtractor
from minimal_signaling.interfaces import Compressor


class MockCompressor(Compressor):
    """Mock compressor for testing."""
    
    def compress(self, text: str) -> str:
        if not text.strip():
            return text
        words = text.split()
        return " ".join(words[: max(1, len(words) // 2)])


def test_sync_event_emitter_calls_handlers():
    """SyncEventEmitter should call registered handlers."""
    emitter = SyncEventEmitter()
    received_events = []
    
    def handler(payload: EventPayload):
        received_events.append(payload)
    
    emitter.on(PipelineEvent.MESSAGE_RECEIVED, handler)
    
    event = create_message_received_event("test", 5)
    emitter.emit(event)
    
    assert len(received_events) == 1
    assert received_events[0].event == PipelineEvent.MESSAGE_RECEIVED


def test_sync_event_emitter_global_handler():
    """SyncEventEmitter should call global handlers for all events."""
    emitter = SyncEventEmitter()
    received_events = []
    
    def handler(payload: EventPayload):
        received_events.append(payload)
    
    emitter.on_all(handler)
    
    # Emit different events
    emitter.emit(create_message_received_event("test", 5))
    emitter.emit(create_compression_start_event(100, 50))
    
    assert len(received_events) == 2


def test_sync_event_emitter_handler_removal():
    """SyncEventEmitter should support handler removal."""
    emitter = SyncEventEmitter()
    received_events = []
    
    def handler(payload: EventPayload):
        received_events.append(payload)
    
    emitter.on(PipelineEvent.MESSAGE_RECEIVED, handler)
    emitter.emit(create_message_received_event("test1", 5))
    
    emitter.off(PipelineEvent.MESSAGE_RECEIVED, handler)
    emitter.emit(create_message_received_event("test2", 5))
    
    # Should only have received first event
    assert len(received_events) == 1


def test_event_payload_has_timestamp():
    """EventPayload should have a timestamp."""
    event = create_message_received_event("test", 5)
    
    assert event.timestamp is not None
    assert event.event == PipelineEvent.MESSAGE_RECEIVED


@given(
    message=st.text(min_size=1, max_size=100),
    token_count=st.integers(min_value=0, max_value=1000)
)
@settings(max_examples=50, deadline=None)
def test_message_received_event_contains_data(message, token_count):
    """MESSAGE_RECEIVED event should contain message and token count."""
    event = create_message_received_event(message, token_count)
    
    assert event.data["message"] == message
    assert event.data["token_count"] == token_count


@given(
    original_tokens=st.integers(min_value=0, max_value=1000),
    final_tokens=st.integers(min_value=0, max_value=1000),
    passes=st.integers(min_value=0, max_value=10)
)
@settings(max_examples=50, deadline=None)
def test_compression_complete_event_contains_data(original_tokens, final_tokens, passes):
    """COMPRESSION_COMPLETE event should contain compression stats."""
    ratio = final_tokens / original_tokens if original_tokens > 0 else 1.0
    event = create_compression_complete_event(original_tokens, final_tokens, passes, ratio)
    
    assert event.data["original_tokens"] == original_tokens
    assert event.data["final_tokens"] == final_tokens
    assert event.data["passes"] == passes


def test_mediator_emits_events():
    """Mediator should emit events during pipeline execution."""
    emitter = SyncEventEmitter()
    received_events = []
    
    def handler(payload: EventPayload):
        received_events.append(payload.event)
    
    emitter.on_all(handler)
    
    config = MediatorConfig(
        compression=CompressionConfig(enabled=True, token_budget=50, max_recursion=3),
        semantic_keys=SemanticKeysConfig(enabled=True),
        judge=JudgeConfig(enabled=False)
    )
    
    mediator = Mediator(
        config=config,
        compressor=MockCompressor(),
        extractor=PlaceholderExtractor(),
        tokenizer=TiktokenTokenizer(),
        event_emitter=emitter
    )
    
    mediator.process("This is a test message for event emission")
    
    # Should have received events
    assert PipelineEvent.MESSAGE_RECEIVED in received_events
    assert PipelineEvent.COMPRESSION_START in received_events
    assert PipelineEvent.COMPRESSION_COMPLETE in received_events
    assert PipelineEvent.EXTRACTION_START in received_events
    assert PipelineEvent.EXTRACTION_COMPLETE in received_events
    assert PipelineEvent.PIPELINE_COMPLETE in received_events


def test_mediator_emits_events_in_order():
    """Mediator should emit events in correct order."""
    emitter = SyncEventEmitter()
    received_events = []
    
    def handler(payload: EventPayload):
        received_events.append(payload.event)
    
    emitter.on_all(handler)
    
    config = MediatorConfig(
        compression=CompressionConfig(enabled=True, token_budget=50, max_recursion=3),
        semantic_keys=SemanticKeysConfig(enabled=True),
        judge=JudgeConfig(enabled=False)
    )
    
    mediator = Mediator(
        config=config,
        compressor=MockCompressor(),
        extractor=PlaceholderExtractor(),
        tokenizer=TiktokenTokenizer(),
        event_emitter=emitter
    )
    
    mediator.process("Test message")
    
    # Check order
    msg_idx = received_events.index(PipelineEvent.MESSAGE_RECEIVED)
    comp_start_idx = received_events.index(PipelineEvent.COMPRESSION_START)
    comp_end_idx = received_events.index(PipelineEvent.COMPRESSION_COMPLETE)
    ext_start_idx = received_events.index(PipelineEvent.EXTRACTION_START)
    ext_end_idx = received_events.index(PipelineEvent.EXTRACTION_COMPLETE)
    pipe_end_idx = received_events.index(PipelineEvent.PIPELINE_COMPLETE)
    
    assert msg_idx < comp_start_idx < comp_end_idx < ext_start_idx < ext_end_idx < pipe_end_idx


def test_handler_errors_dont_break_pipeline():
    """Handler errors should not break the pipeline."""
    emitter = SyncEventEmitter()
    
    def bad_handler(payload: EventPayload):
        raise ValueError("Handler error!")
    
    emitter.on_all(bad_handler)
    
    config = MediatorConfig(
        compression=CompressionConfig(enabled=False, token_budget=50, max_recursion=3),
        semantic_keys=SemanticKeysConfig(enabled=False),
        judge=JudgeConfig(enabled=False)
    )
    
    mediator = Mediator(
        config=config,
        compressor=MockCompressor(),
        extractor=PlaceholderExtractor(),
        tokenizer=TiktokenTokenizer(),
        event_emitter=emitter
    )
    
    # Should not raise despite bad handler
    result = mediator.process("Test message")
    assert result.success is True
