"""Tokenization implementations for accurate token counting.

This module provides tokenizers that count tokens consistently,
which is critical for enforcing token budgets in the compression pipeline.
"""

from __future__ import annotations

import tiktoken

from minimal_signaling.interfaces import Tokenizer


class TiktokenTokenizer(Tokenizer):
    """Tokenizer using tiktoken for accurate token counting.
    
    This implementation uses OpenAI's tiktoken library, which provides
    the same tokenization as GPT models. This ensures accurate token
    counts that match what LLMs actually see.
    """
    
    def __init__(self, encoding: str = "cl100k_base") -> None:
        """Initialize the tokenizer with a specific encoding.
        
        Args:
            encoding: The tiktoken encoding to use. Common options:
                - "cl100k_base" (GPT-4, GPT-3.5-turbo)
                - "p50k_base" (Codex, text-davinci-002)
                - "r50k_base" (GPT-3 models like davinci)
        """
        self.encoding_name = encoding
        self._encoder = tiktoken.get_encoding(encoding)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.
        
        Args:
            text: The text to tokenize and count.
            
        Returns:
            Non-negative integer count of tokens.
        """
        if not text:
            return 0
        return len(self._encoder.encode(text))
    
    def __repr__(self) -> str:
        return f"TiktokenTokenizer(encoding={self.encoding_name!r})"
