"""Property-based tests for configuration validation.

These tests verify that configuration validation correctly accepts
valid configs and rejects invalid ones.
"""

from hypothesis import given, settings, strategies as st
import pytest
from pydantic import ValidationError

from minimal_signaling.config import (
    CompressionConfig,
    MediatorConfig,
)


# **Feature: mediated-minimal-signaling, Property 12: Configuration validation**
class TestConfigurationValidation:
    """Property 12: Configuration validation.
    
    *For any* configuration with invalid values (negative budget, negative 
    recursion limit), loading SHALL raise a validation error with a 
    descriptive message.
    
    **Validates: Requirements 6.7**
    """

    @given(st.integers(max_value=0))
    @settings(max_examples=100)
    def test_negative_token_budget_rejected(self, invalid_budget: int) -> None:
        """Negative or zero token budgets are rejected."""
        with pytest.raises(ValidationError, match="greater than 0"):
            CompressionConfig(token_budget=invalid_budget)

    @given(st.integers(max_value=0))
    @settings(max_examples=100)
    def test_negative_recursion_limit_rejected(self, invalid_limit: int) -> None:
        """Negative or zero recursion limits are rejected."""
        with pytest.raises(ValidationError, match="greater than 0"):
            CompressionConfig(max_recursion=invalid_limit)

    @given(
        st.integers(min_value=1, max_value=10000),
        st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100)
    def test_valid_compression_config_accepted(
        self, budget: int, recursion: int
    ) -> None:
        """Valid positive values are accepted."""
        config = CompressionConfig(token_budget=budget, max_recursion=recursion)
        assert config.token_budget == budget
        assert config.max_recursion == recursion

    def test_missing_config_file_raises_error(self, tmp_path) -> None:
        """Missing config file raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            MediatorConfig.from_yaml(nonexistent)

    def test_empty_config_file_raises_error(self, tmp_path) -> None:
        """Empty config file raises ValueError."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        with pytest.raises(ValueError, match="Empty config file"):
            MediatorConfig.from_yaml(empty_file)

    def test_invalid_yaml_raises_error(self, tmp_path) -> None:
        """Invalid YAML raises ValueError."""
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("compression:\n  token_budget: -10")
        with pytest.raises(ValueError, match="Invalid configuration"):
            MediatorConfig.from_yaml(invalid_file)

    def test_valid_config_loads_successfully(self, tmp_path) -> None:
        """Valid config file loads successfully."""
        valid_file = tmp_path / "valid.yaml"
        valid_file.write_text("""
compression:
  enabled: true
  token_budget: 100
  max_recursion: 3
  model: "test-model"
semantic_keys:
  enabled: true
  schema_version: "1.0"
  extractor: "placeholder"
judge:
  enabled: false
logging:
  level: "DEBUG"
  trace_dir: "test_traces"
dashboard:
  enabled: false
  host: "0.0.0.0"
  port: 9000
  ws_port: 9001
""")
        config = MediatorConfig.from_yaml(valid_file)
        assert config.compression.token_budget == 100
        assert config.compression.max_recursion == 3
        assert config.logging.level == "DEBUG"
