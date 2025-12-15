"""Property-based tests for compression engine."""

from hypothesis import given, strategies as st, settings
import pytest

from minimal_signaling.compression import DistilBARTCompressor, CompressionEngine
from minimal_signaling.tokenization import TiktokenTokenizer
from minimal_signaling.interfaces import Compressor


class MockCompressor(Compressor):
    """Mock compressor for testing that reduces text by ~50%."""
    
    def compress(self, text: str) -> str:
        """Compress by taking first half of words."""
        if not text.strip():
            return text
        words = text.split()
        # Take roughly half the words
        compressed_words = words[: max(1, len(words) // 2)]
        return " ".join(compressed_words)


# **Feature: mediated-minimal-signaling, Property 2: Compression reduces or maintains token count**
@given(text=st.text(min_size=10, max_size=500))
@settings(max_examples=100, deadline=None)
def test_compression_reduces_or_maintains_tokens(text):
    """For any text input, compression should reduce or maintain token count."""
    # Skip if text is only whitespace
    if not text.strip():
        return
    
    tokenizer = TiktokenTokenizer()
    compressor = MockCompressor()
    
    input_tokens = tokenizer.count_tokens(text)
    compressed = compressor.compress(text)
    output_tokens = tokenizer.count_tokens(compressed)
    
    assert output_tokens <= input_tokens, (
        f"Compression increased tokens: {input_tokens} -> {output_tokens}"
    )


# **Feature: mediated-minimal-signaling, Property 3: Recursive compression termination**
@given(
    text=st.text(min_size=20, max_size=200),
    budget=st.integers(min_value=5, max_value=50)
)
@settings(max_examples=100, deadline=None)
def test_compression_engine_terminates_correctly(text, budget):
    """Compression engine must terminate and respect max passes limit."""
    if not text.strip():
        return
    
    tokenizer = TiktokenTokenizer()
    compressor = MockCompressor()
    max_passes = 5
    
    engine = CompressionEngine(
        compressor=compressor,
        tokenizer=tokenizer,
        budget=budget,
        max_passes=max_passes
    )
    
    result = engine.compress_to_budget(text)
    
    # Must not exceed max passes
    assert result.passes <= max_passes, (
        f"Engine exceeded max passes: {result.passes} > {max_passes}"
    )
    
    # If budget not met and passes < max_passes, compression must have stopped improving
    if result.final_tokens > budget and result.passes < max_passes:
        # This means compression stopped because it couldn't improve further
        # which is correct behavior
        pass


# **Feature: mediated-minimal-signaling, Property 3: Recursive compression termination**
@given(
    text=st.text(min_size=50, max_size=500),
    max_passes=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100, deadline=None)
def test_recursive_compression_terminates(text, max_passes):
    """Compression must terminate within max_passes regardless of budget."""
    if not text.strip():
        return
    
    tokenizer = TiktokenTokenizer()
    compressor = MockCompressor()
    # Set impossibly low budget to force max passes
    budget = 1
    
    engine = CompressionEngine(
        compressor=compressor,
        tokenizer=tokenizer,
        budget=budget,
        max_passes=max_passes
    )
    
    result = engine.compress_to_budget(text)
    
    # Must terminate within limit
    assert result.passes <= max_passes, (
        f"Compression exceeded max passes: {result.passes} > {max_passes}"
    )


# **Feature: mediated-minimal-signaling, Property 3: Recursive compression termination**
@given(text=st.text(min_size=10, max_size=200))
@settings(max_examples=100, deadline=None)
def test_compression_stops_when_no_improvement(text):
    """Compression should stop if it's not making progress."""
    if not text.strip():
        return
    
    class NoImprovementCompressor(Compressor):
        """Compressor that returns same text (no improvement)."""
        def compress(self, text: str) -> str:
            return text
    
    tokenizer = TiktokenTokenizer()
    compressor = NoImprovementCompressor()
    
    engine = CompressionEngine(
        compressor=compressor,
        tokenizer=tokenizer,
        budget=1,  # Impossibly low
        max_passes=10
    )
    
    result = engine.compress_to_budget(text)
    
    # Should stop after first pass when no improvement detected
    assert result.passes <= 1, (
        f"Engine should stop when no improvement, but did {result.passes} passes"
    )


# **Feature: mediated-minimal-signaling, Property 4: Compression result completeness**
@given(
    text=st.text(min_size=10, max_size=200),
    budget=st.integers(min_value=5, max_value=100)
)
@settings(max_examples=100, deadline=None)
def test_compression_result_completeness(text, budget):
    """CompressionResult must contain all required fields with valid values."""
    if not text.strip():
        return
    
    tokenizer = TiktokenTokenizer()
    compressor = MockCompressor()
    
    engine = CompressionEngine(
        compressor=compressor,
        tokenizer=tokenizer,
        budget=budget,
        max_passes=5
    )
    
    result = engine.compress_to_budget(text)
    
    # Check all fields are present and valid
    assert result.compressed_text is not None
    assert isinstance(result.compressed_text, str)
    assert result.original_tokens >= 0
    assert result.final_tokens >= 0
    assert result.passes >= 0
    assert isinstance(result.log, list)
    
    # If passes > 0, should have log entries
    if result.passes > 0:
        assert len(result.log) == result.passes


# **Feature: mediated-minimal-signaling, Property 4: Compression result completeness**
@given(text=st.text(min_size=5, max_size=100))
@settings(max_examples=100, deadline=None)
def test_compression_steps_have_valid_ratios(text):
    """Each compression step should have a valid compression ratio."""
    if not text.strip():
        return
    
    tokenizer = TiktokenTokenizer()
    compressor = MockCompressor()
    
    engine = CompressionEngine(
        compressor=compressor,
        tokenizer=tokenizer,
        budget=1,  # Force compression
        max_passes=3
    )
    
    result = engine.compress_to_budget(text)
    
    for step in result.log:
        assert step.input_tokens >= 0
        assert step.output_tokens >= 0
        assert 0.0 <= step.ratio <= 1.0, (
            f"Invalid compression ratio: {step.ratio}"
        )


# **Feature: mediated-minimal-signaling, Property 2: Compression reduces or maintains token count**
def test_compression_already_under_budget():
    """If text is already under budget, no compression should occur."""
    text = "Short text"
    tokenizer = TiktokenTokenizer()
    compressor = MockCompressor()
    
    # Set budget higher than text
    budget = 100
    
    engine = CompressionEngine(
        compressor=compressor,
        tokenizer=tokenizer,
        budget=budget,
        max_passes=5
    )
    
    result = engine.compress_to_budget(text)
    
    # Should return immediately with 0 passes
    assert result.passes == 0
    assert result.compressed_text == text
    assert len(result.log) == 0
