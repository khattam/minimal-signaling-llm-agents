"""Property-based tests for semantic key extraction."""

from hypothesis import given, strategies as st, settings
import pytest

from minimal_signaling.extraction import PlaceholderExtractor
from minimal_signaling.models import KeyType, SemanticKey
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


# Strategy for generating valid key type strings
key_type_strategy = st.sampled_from([
    "INSTRUCTION", "STATE", "GOAL", "CONTEXT", "CONSTRAINT"
])


# Strategy for generating text with embedded keys
@st.composite
def text_with_keys(draw):
    """Generate text that contains semantic key patterns."""
    num_keys = draw(st.integers(min_value=0, max_value=5))
    lines = []
    
    for _ in range(num_keys):
        key_type = draw(key_type_strategy)
        value = draw(st.text(min_size=1, max_size=50).filter(lambda x: '\n' not in x and x.strip()))
        lines.append(f"{key_type}: {value}")
    
    # Add some random text between keys
    if lines:
        result = []
        for line in lines:
            if draw(st.booleans()):
                filler = draw(st.text(max_size=20).filter(lambda x: '\n' not in x))
                result.append(filler)
            result.append(line)
        return "\n".join(result)
    
    return draw(st.text(max_size=100))


# **Feature: mediated-minimal-signaling, Property 7: Extraction result schema conformance**
@given(text=st.text(max_size=500))
@settings(max_examples=100, deadline=None)
def test_extraction_result_has_valid_schema(text):
    """For any text, extraction result must conform to schema."""
    extractor = PlaceholderExtractor()
    result = extractor.extract(text)
    
    # Check schema conformance
    assert isinstance(result.keys, list)
    assert isinstance(result.schema_version, str)
    assert len(result.schema_version) > 0
    assert isinstance(result.raw_output, str)
    
    # All keys must be valid SemanticKey objects
    for key in result.keys:
        assert isinstance(key, SemanticKey)
        assert isinstance(key.type, KeyType)
        assert isinstance(key.value, str)
        assert len(key.value) > 0


# **Feature: mediated-minimal-signaling, Property 7: Extraction result schema conformance**
@given(text=text_with_keys())
@settings(max_examples=100, deadline=None)
def test_extracted_keys_have_valid_types(text):
    """All extracted keys must have valid KeyType values."""
    extractor = PlaceholderExtractor()
    result = extractor.extract(text)
    
    valid_types = {KeyType.INSTRUCTION, KeyType.STATE, KeyType.GOAL, 
                   KeyType.CONTEXT, KeyType.CONSTRAINT}
    
    for key in result.keys:
        assert key.type in valid_types, f"Invalid key type: {key.type}"


# **Feature: mediated-minimal-signaling, Property 7: Extraction result schema conformance**
@given(text=st.text(max_size=200))
@settings(max_examples=100, deadline=None)
def test_extraction_preserves_raw_output(text):
    """Extraction should preserve the raw input text."""
    extractor = PlaceholderExtractor()
    result = extractor.extract(text)
    
    assert result.raw_output == text


# **Feature: mediated-minimal-signaling, Property 7: Extraction result schema conformance**
def test_extraction_empty_text():
    """Extracting from empty text should return empty keys list."""
    extractor = PlaceholderExtractor()
    result = extractor.extract("")
    
    assert result.keys == []
    assert result.schema_version == "1.0"
    assert result.raw_output == ""


# **Feature: mediated-minimal-signaling, Property 7: Extraction result schema conformance**
def test_extraction_recognizes_all_key_types():
    """Extractor should recognize all defined key types."""
    text = """
    INSTRUCTION: do something
    STATE: current state
    GOAL: achieve goal
    CONTEXT: background info
    CONSTRAINT: must satisfy
    """
    
    extractor = PlaceholderExtractor()
    result = extractor.extract(text)
    
    # Should extract all 5 keys
    assert len(result.keys) == 5
    
    # Check all types are present
    extracted_types = {key.type for key in result.keys}
    expected_types = {KeyType.INSTRUCTION, KeyType.STATE, KeyType.GOAL,
                     KeyType.CONTEXT, KeyType.CONSTRAINT}
    assert extracted_types == expected_types


# **Feature: mediated-minimal-signaling, Property 6: Extraction follows compression**
@given(
    text=st.text(min_size=10, max_size=200),
    schema_version=st.text(min_size=1, max_size=10).filter(lambda x: x.strip())
)
@settings(max_examples=100, deadline=None)
def test_extraction_uses_configured_schema_version(text, schema_version):
    """Extractor should use the configured schema version."""
    extractor = PlaceholderExtractor(schema_version=schema_version)
    result = extractor.extract(text)
    
    assert result.schema_version == schema_version


# **Feature: mediated-minimal-signaling, Property 7: Extraction result schema conformance**
@given(text=text_with_keys())
@settings(max_examples=100, deadline=None)
def test_extraction_handles_case_insensitive_keys(text):
    """Extractor should handle case-insensitive key patterns."""
    # Add some lowercase versions
    text_with_lower = text + "\ninstruction: lowercase test"
    
    extractor = PlaceholderExtractor()
    result = extractor.extract(text_with_lower)
    
    # Should extract the lowercase key too
    lowercase_keys = [k for k in result.keys if k.value == "lowercase test"]
    assert len(lowercase_keys) >= 1


# **Feature: mediated-minimal-signaling, Property 7: Extraction result schema conformance**
def test_extraction_handles_multiline_text():
    """Extractor should handle text with multiple lines."""
    text = """Some preamble text
    INSTRUCTION: first instruction
    More text in between
    STATE: current state
    GOAL: final goal
    """
    
    extractor = PlaceholderExtractor()
    result = extractor.extract(text)
    
    # Should extract 3 keys
    assert len(result.keys) == 3
    assert result.keys[0].type == KeyType.INSTRUCTION
    assert result.keys[1].type == KeyType.STATE
    assert result.keys[2].type == KeyType.GOAL



# **Feature: mediated-minimal-signaling, Property 6: Extraction follows compression**
@given(text=st.text(min_size=20, max_size=200))
@settings(max_examples=100, deadline=None)
def test_extraction_follows_compression(text):
    """Extraction should work on compressed text output."""
    from minimal_signaling.compression import CompressionEngine
    from minimal_signaling.tokenization import TiktokenTokenizer
    
    if not text.strip():
        return
    
    # Compress text first
    tokenizer = TiktokenTokenizer()
    compressor = MockCompressor()
    engine = CompressionEngine(
        compressor=compressor,
        tokenizer=tokenizer,
        budget=10,
        max_passes=3
    )
    
    compression_result = engine.compress_to_budget(text)
    
    # Extract from compressed text
    extractor = PlaceholderExtractor()
    extraction_result = extractor.extract(compression_result.compressed_text)
    
    # Extraction should succeed (return valid result)
    assert extraction_result is not None
    assert isinstance(extraction_result.keys, list)
    assert extraction_result.schema_version == "1.0"
    assert extraction_result.raw_output == compression_result.compressed_text


# **Feature: mediated-minimal-signaling, Property 6: Extraction follows compression**
def test_extraction_pipeline_integration():
    """Test that compression -> extraction pipeline works end-to-end."""
    from minimal_signaling.compression import CompressionEngine
    from minimal_signaling.tokenization import TiktokenTokenizer
    
    # Create a message with semantic keys
    text = """
    INSTRUCTION: Process the user request
    STATE: System is ready
    GOAL: Complete the task successfully
    CONTEXT: User has admin privileges
    CONSTRAINT: Must complete within 5 seconds
    """
    
    # Compress
    tokenizer = TiktokenTokenizer()
    compressor = MockCompressor()
    engine = CompressionEngine(
        compressor=compressor,
        tokenizer=tokenizer,
        budget=20,
        max_passes=3
    )
    
    compression_result = engine.compress_to_budget(text)
    
    # Extract
    extractor = PlaceholderExtractor()
    extraction_result = extractor.extract(compression_result.compressed_text)
    
    # Should extract at least some keys (compression may have removed some)
    assert isinstance(extraction_result.keys, list)
    # All extracted keys should be valid
    for key in extraction_result.keys:
        assert isinstance(key, SemanticKey)
        assert key.type in {KeyType.INSTRUCTION, KeyType.STATE, KeyType.GOAL,
                           KeyType.CONTEXT, KeyType.CONSTRAINT}
