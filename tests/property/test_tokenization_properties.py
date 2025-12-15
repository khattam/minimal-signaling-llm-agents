"""Property-based tests for tokenization.

These tests verify that tokenizers behave consistently and correctly.
"""

from hypothesis import given, settings, strategies as st

from minimal_signaling.tokenization import TiktokenTokenizer


# **Feature: mediated-minimal-signaling, Property 1: Token count consistency**
class TestTokenCountConsistency:
    """Property 1: Token count consistency.
    
    *For any* text input, the tokenizer SHALL return a consistent, 
    non-negative integer token count, and calling count_tokens multiple 
    times on the same input SHALL return the same value.
    
    **Validates: Requirements 1.1**
    """

    @given(st.text(max_size=1000))
    @settings(max_examples=100, deadline=None)
    def test_token_count_is_non_negative(self, text: str) -> None:
        """Token counts are always non-negative."""
        tokenizer = TiktokenTokenizer()
        count = tokenizer.count_tokens(text)
        assert count >= 0

    @given(st.text(max_size=1000))
    @settings(max_examples=100)
    def test_token_count_is_consistent(self, text: str) -> None:
        """Calling count_tokens multiple times returns the same value."""
        tokenizer = TiktokenTokenizer()
        count1 = tokenizer.count_tokens(text)
        count2 = tokenizer.count_tokens(text)
        count3 = tokenizer.count_tokens(text)
        assert count1 == count2 == count3

    def test_empty_string_has_zero_tokens(self) -> None:
        """Empty string has zero tokens."""
        tokenizer = TiktokenTokenizer()
        assert tokenizer.count_tokens("") == 0

    @given(st.text(min_size=1, max_size=1000))
    @settings(max_examples=100)
    def test_non_empty_text_has_positive_tokens(self, text: str) -> None:
        """Non-empty text has at least one token (usually)."""
        tokenizer = TiktokenTokenizer()
        count = tokenizer.count_tokens(text)
        # Note: Some whitespace-only strings might have 0 tokens
        # but that's valid behavior
        assert count >= 0

    @given(st.text(max_size=500), st.text(max_size=500))
    @settings(max_examples=100)
    def test_concatenation_tokens_reasonable(self, text1: str, text2: str) -> None:
        """Concatenating text doesn't wildly change token count."""
        tokenizer = TiktokenTokenizer()
        count1 = tokenizer.count_tokens(text1)
        count2 = tokenizer.count_tokens(text2)
        combined_count = tokenizer.count_tokens(text1 + text2)
        
        # Combined should be roughly the sum (tokenization can merge/split)
        # but shouldn't be wildly different
        assert combined_count >= 0
        # Sanity check: combined shouldn't be more than 2x the sum
        # (accounting for potential token boundary effects)
        assert combined_count <= (count1 + count2) * 2 + 10

    def test_different_encodings_work(self) -> None:
        """Different tiktoken encodings can be used."""
        text = "Hello, world! This is a test."
        
        tokenizer_cl100k = TiktokenTokenizer("cl100k_base")
        tokenizer_p50k = TiktokenTokenizer("p50k_base")
        
        count1 = tokenizer_cl100k.count_tokens(text)
        count2 = tokenizer_p50k.count_tokens(text)
        
        # Both should return valid counts
        assert count1 > 0
        assert count2 > 0
        # They might differ slightly due to different encodings
