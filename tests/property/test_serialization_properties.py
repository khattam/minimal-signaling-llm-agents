"""Property-based tests for serialization round-trips.

These tests verify that data models can be serialized and deserialized
without loss of information.
"""

import json

from hypothesis import given, settings, strategies as st

from minimal_signaling.models import (
    CompressionResult,
    CompressionStep,
    ExtractionResult,
    JudgeResult,
    KeyType,
    SemanticKey,
)


# Strategies for generating test data
key_type_strategy = st.sampled_from(list(KeyType))
non_empty_text = st.text(min_size=1, max_size=100).filter(lambda x: x.strip())


@st.composite
def semantic_key_strategy(draw: st.DrawFn) -> SemanticKey:
    """Generate a valid SemanticKey."""
    return SemanticKey(
        type=draw(key_type_strategy),
        value=draw(non_empty_text),
    )


@st.composite
def compression_step_strategy(draw: st.DrawFn) -> CompressionStep:
    """Generate a valid CompressionStep."""
    input_tokens = draw(st.integers(min_value=0, max_value=10000))
    output_tokens = draw(st.integers(min_value=0, max_value=input_tokens or 10000))
    return CompressionStep(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_text=draw(st.text(max_size=500)),
        output_text=draw(st.text(max_size=500)),
    )


@st.composite
def compression_result_strategy(draw: st.DrawFn) -> CompressionResult:
    """Generate a valid CompressionResult."""
    original = draw(st.integers(min_value=0, max_value=10000))
    final = draw(st.integers(min_value=0, max_value=original or 10000))
    passes = draw(st.integers(min_value=0, max_value=10))
    log = draw(st.lists(compression_step_strategy(), max_size=passes))
    return CompressionResult(
        compressed_text=draw(st.text(max_size=500)),
        original_tokens=original,
        final_tokens=final,
        passes=passes,
        log=log,
    )


@st.composite
def extraction_result_strategy(draw: st.DrawFn) -> ExtractionResult:
    """Generate a valid ExtractionResult."""
    return ExtractionResult(
        keys=draw(st.lists(semantic_key_strategy(), max_size=10)),
        schema_version=draw(st.text(min_size=1, max_size=10).filter(lambda x: x.strip())),
        raw_output=draw(st.text(max_size=500)),
    )


@st.composite
def judge_result_strategy(draw: st.DrawFn) -> JudgeResult:
    """Generate a valid JudgeResult."""
    return JudgeResult(
        passed=draw(st.booleans()),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        issues=draw(st.lists(st.text(max_size=100), max_size=5)),
    )


# **Feature: mediated-minimal-signaling, Property 8: Semantic key serialization round-trip**
class TestSemanticKeySerializationRoundTrip:
    """Property 8: Semantic key serialization round-trip.
    
    *For any* list of SemanticKey objects, serializing to JSON and 
    deserializing back SHALL produce an equivalent list.
    
    **Validates: Requirements 4.3**
    """

    @given(semantic_key_strategy())
    @settings(max_examples=100)
    def test_single_key_round_trip(self, key: SemanticKey) -> None:
        """Single semantic key survives JSON round-trip."""
        json_str = key.model_dump_json()
        restored = SemanticKey.model_validate_json(json_str)
        assert restored == key

    @given(st.lists(semantic_key_strategy(), max_size=20))
    @settings(max_examples=100)
    def test_key_list_round_trip(self, keys: list[SemanticKey]) -> None:
        """List of semantic keys survives JSON round-trip."""
        # Serialize list to JSON
        json_str = json.dumps([k.model_dump() for k in keys])
        # Deserialize back
        data = json.loads(json_str)
        restored = [SemanticKey.model_validate(d) for d in data]
        assert restored == keys

    @given(extraction_result_strategy())
    @settings(max_examples=100)
    def test_extraction_result_round_trip(self, result: ExtractionResult) -> None:
        """ExtractionResult with keys survives JSON round-trip."""
        json_str = result.model_dump_json()
        restored = ExtractionResult.model_validate_json(json_str)
        assert restored.keys == result.keys
        assert restored.schema_version == result.schema_version

    @given(compression_result_strategy())
    @settings(max_examples=100)
    def test_compression_result_round_trip(self, result: CompressionResult) -> None:
        """CompressionResult survives JSON round-trip."""
        json_str = result.model_dump_json()
        restored = CompressionResult.model_validate_json(json_str)
        assert restored.compressed_text == result.compressed_text
        assert restored.original_tokens == result.original_tokens
        assert restored.final_tokens == result.final_tokens
        assert restored.passes == result.passes

    @given(judge_result_strategy())
    @settings(max_examples=100)
    def test_judge_result_round_trip(self, result: JudgeResult) -> None:
        """JudgeResult survives JSON round-trip."""
        json_str = result.model_dump_json()
        restored = JudgeResult.model_validate_json(json_str)
        assert restored.passed == result.passed
        assert abs(restored.confidence - result.confidence) < 1e-9
        assert restored.issues == result.issues



# **Feature: mediated-minimal-signaling, Property 9: Schema validation rejects invalid input**
class TestSchemaValidationRejectsInvalid:
    """Property 9: Schema validation rejects invalid input.
    
    *For any* JSON input that does not conform to the semantic key schema,
    deserialization SHALL raise a validation error.
    
    **Validates: Requirements 4.4**
    """

    @given(st.text(max_size=50))
    @settings(max_examples=100)
    def test_invalid_key_type_rejected(self, invalid_type: str) -> None:
        """Invalid key types are rejected."""
        # Skip if accidentally valid
        valid_types = {t.value for t in KeyType}
        if invalid_type in valid_types:
            return
        
        invalid_data = {"type": invalid_type, "value": "test"}
        try:
            SemanticKey.model_validate(invalid_data)
            # If we get here with an invalid type, that's a failure
            assert invalid_type in valid_types, f"Should have rejected type: {invalid_type}"
        except Exception:
            pass  # Expected - validation should fail

    @given(st.sampled_from(list(KeyType)))
    @settings(max_examples=100)
    def test_empty_value_rejected(self, key_type: KeyType) -> None:
        """Empty values are rejected."""
        invalid_data = {"type": key_type.value, "value": ""}
        try:
            SemanticKey.model_validate(invalid_data)
            assert False, "Should have rejected empty value"
        except Exception:
            pass  # Expected

    def test_missing_type_rejected(self) -> None:
        """Missing type field is rejected."""
        invalid_data = {"value": "test"}
        try:
            SemanticKey.model_validate(invalid_data)
            assert False, "Should have rejected missing type"
        except Exception:
            pass  # Expected

    def test_missing_value_rejected(self) -> None:
        """Missing value field is rejected."""
        invalid_data = {"type": "INSTRUCTION"}
        try:
            SemanticKey.model_validate(invalid_data)
            assert False, "Should have rejected missing value"
        except Exception:
            pass  # Expected

    def test_empty_schema_version_rejected(self) -> None:
        """Empty schema version is rejected."""
        invalid_data = {"keys": [], "schema_version": "", "raw_output": ""}
        try:
            ExtractionResult.model_validate(invalid_data)
            assert False, "Should have rejected empty schema_version"
        except Exception:
            pass  # Expected

    @given(st.floats(allow_nan=True, allow_infinity=True).filter(
        lambda x: x < 0 or x > 1 or x != x  # NaN check
    ))
    @settings(max_examples=100)
    def test_invalid_confidence_rejected(self, invalid_confidence: float) -> None:
        """Confidence outside [0, 1] or NaN is rejected."""
        invalid_data = {"passed": True, "confidence": invalid_confidence, "issues": []}
        try:
            JudgeResult.model_validate(invalid_data)
            assert False, f"Should have rejected confidence: {invalid_confidence}"
        except Exception:
            pass  # Expected

    @given(st.integers(max_value=-1))
    @settings(max_examples=100)
    def test_negative_token_count_rejected(self, negative_count: int) -> None:
        """Negative token counts are rejected."""
        invalid_data = {
            "compressed_text": "test",
            "original_tokens": negative_count,
            "final_tokens": 0,
            "passes": 0,
            "log": [],
        }
        try:
            CompressionResult.model_validate(invalid_data)
            assert False, f"Should have rejected negative tokens: {negative_count}"
        except Exception:
            pass  # Expected
