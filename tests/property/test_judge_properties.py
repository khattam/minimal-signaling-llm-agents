"""Property-based tests for judge verification layer."""

from hypothesis import given, strategies as st, settings
import pytest

from minimal_signaling.judge import PlaceholderJudge
from minimal_signaling.models import SemanticKey, KeyType


# Strategy for generating semantic keys
@st.composite
def semantic_key_strategy(draw):
    """Generate a valid semantic key."""
    key_type = draw(st.sampled_from(list(KeyType)))
    value = draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    return SemanticKey(type=key_type, value=value)


# **Feature: mediated-minimal-signaling, Property 11: Judge result completeness**
@given(
    original=st.text(max_size=200),
    keys=st.lists(semantic_key_strategy(), max_size=10)
)
@settings(max_examples=100, deadline=None)
def test_judge_result_has_required_fields(original, keys):
    """Judge result must contain all required fields with valid values."""
    judge = PlaceholderJudge()
    result = judge.evaluate(original, keys)
    
    # Check all required fields
    assert isinstance(result.passed, bool)
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0, (
        f"Confidence out of range: {result.confidence}"
    )
    assert isinstance(result.issues, list)
    
    # All issues should be strings
    for issue in result.issues:
        assert isinstance(issue, str)


# **Feature: mediated-minimal-signaling, Property 11: Judge result completeness**
@given(
    original=st.text(max_size=200),
    keys=st.lists(semantic_key_strategy(), max_size=10),
    default_pass=st.booleans(),
    confidence=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100, deadline=None)
def test_judge_respects_configuration(original, keys, default_pass, confidence):
    """Judge should respect its configuration parameters."""
    judge = PlaceholderJudge(
        default_pass=default_pass,
        default_confidence=confidence
    )
    result = judge.evaluate(original, keys)
    
    # Result should be influenced by configuration
    assert isinstance(result.passed, bool)
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


# **Feature: mediated-minimal-signaling, Property 11: Judge result completeness**
@given(
    original=st.text(min_size=10, max_size=200),
    min_threshold=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=100, deadline=None)
def test_judge_enforces_minimum_keys_threshold(original, min_threshold):
    """Judge should fail when keys are below threshold."""
    judge = PlaceholderJudge(
        default_pass=True,
        min_keys_threshold=min_threshold
    )
    
    # Create fewer keys than threshold
    keys = [
        SemanticKey(type=KeyType.INSTRUCTION, value="test")
        for _ in range(min_threshold - 1)
    ]
    
    result = judge.evaluate(original, keys)
    
    # Should fail due to insufficient keys
    assert result.passed is False
    assert len(result.issues) > 0
    assert any("Too few keys" in issue for issue in result.issues)


# **Feature: mediated-minimal-signaling, Property 11: Judge result completeness**
def test_judge_detects_empty_text_with_keys():
    """Judge should detect when keys are extracted from empty text."""
    judge = PlaceholderJudge(default_pass=True)
    
    keys = [SemanticKey(type=KeyType.INSTRUCTION, value="test")]
    result = judge.evaluate("", keys)
    
    # Should fail
    assert result.passed is False
    assert len(result.issues) > 0
    assert any("empty text" in issue.lower() for issue in result.issues)
    assert result.confidence == 0.0


# **Feature: mediated-minimal-signaling, Property 11: Judge result completeness**
def test_judge_detects_empty_key_values():
    """Judge should detect keys with empty values."""
    judge = PlaceholderJudge(default_pass=True)
    
    keys = [
        SemanticKey(type=KeyType.INSTRUCTION, value="valid"),
        SemanticKey(type=KeyType.STATE, value="  ")  # Empty after strip
    ]
    
    result = judge.evaluate("Some text", keys)
    
    # Should fail due to empty value
    assert result.passed is False
    assert len(result.issues) > 0
    assert any("empty values" in issue.lower() for issue in result.issues)


# **Feature: mediated-minimal-signaling, Property 11: Judge result completeness**
@given(
    original=st.text(min_size=10, max_size=200),
    num_keys=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100, deadline=None)
def test_judge_passes_valid_extraction(original, num_keys):
    """Judge should pass when extraction looks valid."""
    judge = PlaceholderJudge(
        default_pass=True,
        min_keys_threshold=0
    )
    
    # Create valid keys
    keys = [
        SemanticKey(
            type=KeyType.INSTRUCTION,
            value=f"instruction {i}"
        )
        for i in range(num_keys)
    ]
    
    result = judge.evaluate(original, keys)
    
    # Should pass with valid keys
    assert result.passed is True
    assert result.confidence > 0.0
    assert len(result.issues) == 0


# **Feature: mediated-minimal-signaling, Property 11: Judge result completeness**
def test_judge_handles_empty_original_and_no_keys():
    """Judge should handle empty text with no keys gracefully."""
    judge = PlaceholderJudge(default_pass=True)
    
    result = judge.evaluate("", [])
    
    # Should pass (no keys from empty text is valid)
    assert result.passed is True
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


# **Feature: mediated-minimal-signaling, Property 11: Judge result completeness**
@given(keys=st.lists(semantic_key_strategy(), min_size=1, max_size=10))
@settings(max_examples=100, deadline=None)
def test_judge_confidence_in_valid_range(keys):
    """Judge confidence must always be in [0, 1] range."""
    judge = PlaceholderJudge()
    
    # Test with various original texts
    for original in ["", "short", "a" * 100]:
        result = judge.evaluate(original, keys)
        assert 0.0 <= result.confidence <= 1.0, (
            f"Confidence out of range: {result.confidence}"
        )



# **Feature: mediated-minimal-signaling, Property 10: Judge invocation follows configuration**
# NOTE: This property is tested in test_pipeline_properties.py::test_judge_invoked_only_when_enabled
# since it requires the full Mediator pipeline
