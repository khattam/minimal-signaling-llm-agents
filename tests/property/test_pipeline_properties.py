"""Property-based tests for end-to-end pipeline and mediator."""

from hypothesis import given, strategies as st, settings
import pytest

from minimal_signaling.mediator import Mediator
from minimal_signaling.config import MediatorConfig, CompressionConfig, SemanticKeysConfig, JudgeConfig
from minimal_signaling.interfaces import Compressor, SemanticKeyExtractor, Judge
from minimal_signaling.tokenization import TiktokenTokenizer
from minimal_signaling.extraction import PlaceholderExtractor
from minimal_signaling.judge import PlaceholderJudge
from minimal_signaling.models import SemanticKey, KeyType, ExtractionResult, JudgeResult


class MockCompressor(Compressor):
    """Mock compressor for testing."""
    
    def compress(self, text: str) -> str:
        """Compress by taking first half of words."""
        if not text.strip():
            return text
        words = text.split()
        compressed_words = words[: max(1, len(words) // 2)]
        return " ".join(compressed_words)


class AlternativeCompressor(Compressor):
    """Alternative compressor implementation for substitutability testing."""
    
    def compress(self, text: str) -> str:
        """Compress by taking every other word."""
        if not text.strip():
            return text
        words = text.split()
        compressed_words = [w for i, w in enumerate(words) if i % 2 == 0]
        return " ".join(compressed_words) if compressed_words else text


class NoOpCompressor(Compressor):
    """No-op compressor that returns text unchanged."""
    
    def compress(self, text: str) -> str:
        """Return text unchanged."""
        return text


# **Feature: mediated-minimal-signaling, Property 5: Compressor interface substitutability**
@given(text=st.text(min_size=10, max_size=200))
@settings(max_examples=100, deadline=None)
def test_mediator_works_with_different_compressors(text):
    """Mediator should work with any Compressor implementation."""
    if not text.strip():
        return
    
    tokenizer = TiktokenTokenizer()
    extractor = PlaceholderExtractor()
    
    # Test with different compressor implementations
    compressors = [
        MockCompressor(),
        AlternativeCompressor(),
        NoOpCompressor()
    ]
    
    for compressor in compressors:
        config = MediatorConfig(
            compression=CompressionConfig(
                enabled=True,
                token_budget=50,
                max_recursion=3
            ),
            semantic_keys=SemanticKeysConfig(enabled=True),
            judge=JudgeConfig(enabled=False)
        )
        
        mediator = Mediator(
            config=config,
            compressor=compressor,
            extractor=extractor,
            tokenizer=tokenizer
        )
        
        # Should process without errors
        result = mediator.process(text)
        assert result is not None
        assert isinstance(result.success, bool)


# **Feature: mediated-minimal-signaling, Property 14: End-to-end pipeline integrity**
@given(text=st.text(min_size=10, max_size=200))
@settings(max_examples=100, deadline=None)
def test_pipeline_never_silently_drops_messages(text):
    """Pipeline must either succeed or fail with logged error, never silently drop."""
    if not text.strip():
        return
    
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
    
    # Must return a result
    assert result is not None
    
    # Must have success status
    assert isinstance(result.success, bool)
    
    # If failed, must have error information
    if not result.success:
        assert result.error is not None
        assert result.error.stage != ""
        assert result.error.message != ""


# **Feature: mediated-minimal-signaling, Property 14: End-to-end pipeline integrity**
@given(
    text=st.text(min_size=20, max_size=200),
    compression_enabled=st.booleans(),
    extraction_enabled=st.booleans(),
    judge_enabled=st.booleans()
)
@settings(max_examples=100, deadline=None)
def test_pipeline_respects_stage_toggles(text, compression_enabled, extraction_enabled, judge_enabled):
    """Pipeline should respect stage enable/disable configuration."""
    if not text.strip():
        return
    
    config = MediatorConfig(
        compression=CompressionConfig(
            enabled=compression_enabled,
            token_budget=50,
            max_recursion=3
        ),
        semantic_keys=SemanticKeysConfig(enabled=extraction_enabled),
        judge=JudgeConfig(enabled=judge_enabled)
    )
    
    mediator = Mediator(
        config=config,
        compressor=MockCompressor(),
        extractor=PlaceholderExtractor(),
        tokenizer=TiktokenTokenizer(),
        judge=PlaceholderJudge()
    )
    
    result = mediator.process(text)
    
    # Check that stages were executed according to config
    if compression_enabled:
        assert result.compression is not None or not result.success
    else:
        assert result.compression is None
    
    if extraction_enabled:
        assert result.extraction is not None or not result.success
    else:
        assert result.extraction is None
    
    if judge_enabled and extraction_enabled:
        # Judge only runs if extraction is also enabled
        assert result.judge is not None or not result.success
    elif not judge_enabled:
        assert result.judge is None


# **Feature: mediated-minimal-signaling, Property 10: Judge invocation follows configuration**
@given(text=st.text(min_size=20, max_size=200))
@settings(max_examples=100, deadline=None)
def test_judge_invoked_only_when_enabled(text):
    """Judge should only be invoked when judge_enabled is True."""
    if not text.strip():
        return
    
    tokenizer = TiktokenTokenizer()
    compressor = MockCompressor()
    extractor = PlaceholderExtractor()
    judge = PlaceholderJudge()
    
    # Test with judge enabled
    config_enabled = MediatorConfig(
        compression=CompressionConfig(enabled=True, token_budget=50, max_recursion=3),
        semantic_keys=SemanticKeysConfig(enabled=True),
        judge=JudgeConfig(enabled=True)
    )
    
    mediator_enabled = Mediator(
        config=config_enabled,
        compressor=compressor,
        extractor=extractor,
        tokenizer=tokenizer,
        judge=judge
    )
    
    result_enabled = mediator_enabled.process(text)
    
    # Judge should be invoked (result present)
    if result_enabled.success and result_enabled.extraction is not None:
        assert result_enabled.judge is not None
    
    # Test with judge disabled
    config_disabled = MediatorConfig(
        compression=CompressionConfig(enabled=True, token_budget=50, max_recursion=3),
        semantic_keys=SemanticKeysConfig(enabled=True),
        judge=JudgeConfig(enabled=False)
    )
    
    mediator_disabled = Mediator(
        config=config_disabled,
        compressor=compressor,
        extractor=extractor,
        tokenizer=tokenizer,
        judge=judge
    )
    
    result_disabled = mediator_disabled.process(text)
    
    # Judge should NOT be invoked
    assert result_disabled.judge is None


# **Feature: mediated-minimal-signaling, Property 14: End-to-end pipeline integrity**
def test_pipeline_handles_empty_message():
    """Pipeline should handle empty messages gracefully."""
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
    
    result = mediator.process("")
    
    # Should complete successfully
    assert result is not None
    assert isinstance(result.success, bool)


# **Feature: mediated-minimal-signaling, Property 14: End-to-end pipeline integrity**
def test_pipeline_produces_valid_result_structure():
    """Pipeline result must have valid structure."""
    text = "INSTRUCTION: test instruction\nSTATE: current state"
    
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
    
    # Check result structure
    assert result.success is True
    assert result.compression is not None
    assert result.extraction is not None
    assert result.judge is not None
    assert result.error is None
    assert result.duration_ms >= 0.0
    
    # Check compression result
    assert result.compression.compressed_text is not None
    assert result.compression.original_tokens >= 0
    assert result.compression.final_tokens >= 0
    
    # Check extraction result
    assert isinstance(result.extraction.keys, list)
    assert result.extraction.schema_version != ""
    
    # Check judge result
    assert isinstance(result.judge.passed, bool)
    assert 0.0 <= result.judge.confidence <= 1.0


# **Feature: mediated-minimal-signaling, Property 14: End-to-end pipeline integrity**
@given(text=st.text(min_size=10, max_size=200))
@settings(max_examples=100, deadline=None)
def test_pipeline_duration_is_recorded(text):
    """Pipeline must record execution duration."""
    if not text.strip():
        return
    
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
    
    # Duration must be recorded and non-negative
    assert result.duration_ms >= 0.0
    # Duration should be reasonable (less than 10 seconds for these simple operations)
    assert result.duration_ms < 10000.0
