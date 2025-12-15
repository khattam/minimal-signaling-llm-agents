"""Property-based tests for trace logging."""

from hypothesis import given, strategies as st, settings
import pytest
import tempfile
import shutil
from pathlib import Path

from minimal_signaling.trace import TraceLogger
from minimal_signaling.mediator import Mediator
from minimal_signaling.config import MediatorConfig, CompressionConfig, SemanticKeysConfig, JudgeConfig
from minimal_signaling.compression import CompressionEngine
from minimal_signaling.tokenization import TiktokenTokenizer
from minimal_signaling.extraction import PlaceholderExtractor
from minimal_signaling.judge import PlaceholderJudge
from minimal_signaling.interfaces import Compressor


class MockCompressor(Compressor):
    """Mock compressor for testing."""
    
    def compress(self, text: str) -> str:
        """Compress by taking first half of words."""
        if not text.strip():
            return text
        words = text.split()
        compressed_words = words[: max(1, len(words) // 2)]
        return " ".join(compressed_words)


# **Feature: mediated-minimal-signaling, Property 13: Trace record completeness**
@given(text=st.text(min_size=10, max_size=200))
@settings(max_examples=50, deadline=None)
def test_trace_contains_all_required_fields(text):
    """Trace record must contain all required fields."""
    if not text.strip():
        return
    
    # Create temporary trace directory
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_logger = TraceLogger(trace_dir=tmpdir)
        
        # Run pipeline
        config = MediatorConfig(
            compression=CompressionConfig(enabled=True, token_budget=50, max_recursion=3),
            semantic_keys=SemanticKeysConfig(enabled=True),
            judge=JudgeConfig(enabled=True)
        )
        
        mediator = Mediator(
            config=config,
            compressor=MockCompressor(),
            extractor=PlaceholderExtractor(),
            tokenizer=TiktokenTokenizer(),
            judge=PlaceholderJudge()
        )
        
        result = mediator.process(text)
        tokenizer = TiktokenTokenizer()
        original_tokens = tokenizer.count_tokens(text)
        
        # Log trace
        trace_file = trace_logger.log_trace_from_result(
            original_text=text,
            original_tokens=original_tokens,
            result=result,
            config=config
        )
        
        # Verify trace file exists
        assert trace_file.exists()
        
        # Read trace back
        message_id = trace_file.stem.replace("trace_", "")
        trace = trace_logger.read_trace(message_id)
        
        # Check all required fields
        assert trace.message_id == message_id
        assert trace.timestamp is not None
        assert trace.original_text == text
        assert trace.original_tokens == original_tokens
        assert trace.duration_ms >= 0.0
        assert isinstance(trace.config_snapshot, dict)
        
        # Check compression data if enabled
        if config.compression.enabled and result.compression:
            assert trace.compression is not None
            assert trace.compression.original_tokens >= 0
            assert trace.compression.final_tokens >= 0
        
        # Check extraction data if enabled
        if config.semantic_keys.enabled and result.extraction:
            assert trace.extraction is not None
            assert isinstance(trace.extraction.keys, list)


# **Feature: mediated-minimal-signaling, Property 13: Trace record completeness**
def test_trace_includes_compression_steps():
    """Trace should include detailed compression steps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_logger = TraceLogger(trace_dir=tmpdir)
        
        text = "This is a longer message that will require compression to meet the token budget."
        
        config = MediatorConfig(
            compression=CompressionConfig(enabled=True, token_budget=10, max_recursion=5),
            semantic_keys=SemanticKeysConfig(enabled=True),
            judge=JudgeConfig(enabled=False)
        )
        
        mediator = Mediator(
            config=config,
            compressor=MockCompressor(),
            extractor=PlaceholderExtractor(),
            tokenizer=TiktokenTokenizer()
        )
        
        result = mediator.process(text)
        tokenizer = TiktokenTokenizer()
        original_tokens = tokenizer.count_tokens(text)
        
        trace_file = trace_logger.log_trace_from_result(
            original_text=text,
            original_tokens=original_tokens,
            result=result,
            config=config
        )
        
        message_id = trace_file.stem.replace("trace_", "")
        trace = trace_logger.read_trace(message_id)
        
        # Should have compression data
        assert trace.compression is not None
        assert trace.compression.passes >= 0
        assert isinstance(trace.compression.log, list)


# **Feature: mediated-minimal-signaling, Property 13: Trace record completeness**
def test_trace_includes_extracted_keys():
    """Trace should include extracted semantic keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_logger = TraceLogger(trace_dir=tmpdir)
        
        text = "INSTRUCTION: test instruction\nSTATE: current state"
        
        config = MediatorConfig(
            compression=CompressionConfig(enabled=False, token_budget=50, max_recursion=3),
            semantic_keys=SemanticKeysConfig(enabled=True),
            judge=JudgeConfig(enabled=False)
        )
        
        mediator = Mediator(
            config=config,
            compressor=MockCompressor(),
            extractor=PlaceholderExtractor(),
            tokenizer=TiktokenTokenizer()
        )
        
        result = mediator.process(text)
        tokenizer = TiktokenTokenizer()
        original_tokens = tokenizer.count_tokens(text)
        
        trace_file = trace_logger.log_trace_from_result(
            original_text=text,
            original_tokens=original_tokens,
            result=result,
            config=config
        )
        
        message_id = trace_file.stem.replace("trace_", "")
        trace = trace_logger.read_trace(message_id)
        
        # Should have extraction data
        assert trace.extraction is not None
        assert len(trace.extraction.keys) > 0
        assert trace.extraction.schema_version != ""


# **Feature: mediated-minimal-signaling, Property 13: Trace record completeness**
def test_trace_includes_judge_results():
    """Trace should include judge results when enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_logger = TraceLogger(trace_dir=tmpdir)
        
        text = "INSTRUCTION: test instruction"
        
        config = MediatorConfig(
            compression=CompressionConfig(enabled=False, token_budget=50, max_recursion=3),
            semantic_keys=SemanticKeysConfig(enabled=True),
            judge=JudgeConfig(enabled=True)
        )
        
        mediator = Mediator(
            config=config,
            compressor=MockCompressor(),
            extractor=PlaceholderExtractor(),
            tokenizer=TiktokenTokenizer(),
            judge=PlaceholderJudge()
        )
        
        result = mediator.process(text)
        tokenizer = TiktokenTokenizer()
        original_tokens = tokenizer.count_tokens(text)
        
        trace_file = trace_logger.log_trace_from_result(
            original_text=text,
            original_tokens=original_tokens,
            result=result,
            config=config
        )
        
        message_id = trace_file.stem.replace("trace_", "")
        trace = trace_logger.read_trace(message_id)
        
        # Should have judge data
        assert trace.judge is not None
        assert isinstance(trace.judge.passed, bool)
        assert 0.0 <= trace.judge.confidence <= 1.0


# **Feature: mediated-minimal-signaling, Property 13: Trace record completeness**
def test_trace_includes_config_snapshot():
    """Trace should include configuration snapshot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_logger = TraceLogger(trace_dir=tmpdir)
        
        text = "Test message"
        
        config = MediatorConfig(
            compression=CompressionConfig(
                enabled=True,
                token_budget=25,
                max_recursion=4,
                model="test-model"
            ),
            semantic_keys=SemanticKeysConfig(
                enabled=True,
                schema_version="2.0",
                extractor="placeholder"
            ),
            judge=JudgeConfig(enabled=True)
        )
        
        mediator = Mediator(
            config=config,
            compressor=MockCompressor(),
            extractor=PlaceholderExtractor(schema_version="2.0"),
            tokenizer=TiktokenTokenizer(),
            judge=PlaceholderJudge()
        )
        
        result = mediator.process(text)
        tokenizer = TiktokenTokenizer()
        original_tokens = tokenizer.count_tokens(text)
        
        trace_file = trace_logger.log_trace_from_result(
            original_text=text,
            original_tokens=original_tokens,
            result=result,
            config=config
        )
        
        message_id = trace_file.stem.replace("trace_", "")
        trace = trace_logger.read_trace(message_id)
        
        # Check config snapshot
        assert "compression" in trace.config_snapshot
        assert trace.config_snapshot["compression"]["token_budget"] == 25
        assert trace.config_snapshot["compression"]["max_recursion"] == 4
        assert "semantic_keys" in trace.config_snapshot
        assert trace.config_snapshot["semantic_keys"]["schema_version"] == "2.0"
        assert "judge" in trace.config_snapshot
        assert trace.config_snapshot["judge"]["enabled"] is True


# **Feature: mediated-minimal-signaling, Property 13: Trace record completeness**
def test_trace_logger_lists_traces():
    """TraceLogger should be able to list all traces."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_logger = TraceLogger(trace_dir=tmpdir)
        
        config = MediatorConfig(
            compression=CompressionConfig(enabled=False, token_budget=50, max_recursion=3),
            semantic_keys=SemanticKeysConfig(enabled=False),
            judge=JudgeConfig(enabled=False)
        )
        
        mediator = Mediator(
            config=config,
            compressor=MockCompressor(),
            extractor=PlaceholderExtractor(),
            tokenizer=TiktokenTokenizer()
        )
        
        # Create multiple traces
        messages = ["message 1", "message 2", "message 3"]
        message_ids = []
        
        for msg in messages:
            result = mediator.process(msg)
            tokenizer = TiktokenTokenizer()
            original_tokens = tokenizer.count_tokens(msg)
            
            trace_file = trace_logger.log_trace_from_result(
                original_text=msg,
                original_tokens=original_tokens,
                result=result,
                config=config
            )
            message_id = trace_file.stem.replace("trace_", "")
            message_ids.append(message_id)
        
        # List traces
        traces = trace_logger.list_traces()
        
        # Should have all traces
        assert len(traces) == 3
        for msg_id in message_ids:
            assert msg_id in traces


# **Feature: mediated-minimal-signaling, Property 13: Trace record completeness**
@given(text=st.text(min_size=10, max_size=100))
@settings(max_examples=50, deadline=None)
def test_trace_round_trip(text):
    """Trace should be readable after writing."""
    if not text.strip():
        return
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_logger = TraceLogger(trace_dir=tmpdir)
        
        config = MediatorConfig(
            compression=CompressionConfig(enabled=True, token_budget=50, max_recursion=3),
            semantic_keys=SemanticKeysConfig(enabled=True),
            judge=JudgeConfig(enabled=False)
        )
        
        mediator = Mediator(
            config=config,
            compressor=MockCompressor(),
            extractor=PlaceholderExtractor(),
            tokenizer=TiktokenTokenizer()
        )
        
        result = mediator.process(text)
        tokenizer = TiktokenTokenizer()
        original_tokens = tokenizer.count_tokens(text)
        
        # Write trace
        trace_file = trace_logger.log_trace_from_result(
            original_text=text,
            original_tokens=original_tokens,
            result=result,
            config=config
        )
        
        # Read trace back
        message_id = trace_file.stem.replace("trace_", "")
        trace = trace_logger.read_trace(message_id)
        
        # Verify data matches
        assert trace.original_text == text
        assert trace.original_tokens == original_tokens
