"""Abstract interfaces for the Mediated Minimal-Signaling Architecture.

This module defines the core abstractions that allow swapping implementations
of compression, extraction, verification, and tokenization components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minimal_signaling.models import (
        ExtractionResult,
        JudgeResult,
        SemanticKey,
    )


class Tokenizer(ABC):
    """Abstract interface for token counting.
    
    Implementations should provide consistent token counts
    for the same input text.
    """
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.
        
        Args:
            text: The text to tokenize and count.
            
        Returns:
            Non-negative integer count of tokens.
        """
        pass


class Compressor(ABC):
    """Abstract interface for text compression.
    
    Implementations should reduce text length while preserving
    semantic meaning. The compression ratio may vary by implementation.
    """
    
    @abstractmethod
    def compress(self, text: str) -> str:
        """Compress the given text to a shorter form.
        
        Args:
            text: The text to compress.
            
        Returns:
            Compressed version of the text.
        """
        pass


class SemanticKeyExtractor(ABC):
    """Abstract interface for semantic key extraction.
    
    Implementations convert compressed text into structured
    semantic keys following a defined schema.
    """
    
    @abstractmethod
    def extract(self, text: str) -> "ExtractionResult":
        """Extract semantic keys from the given text.
        
        Args:
            text: The text to extract keys from (typically compressed).
            
        Returns:
            ExtractionResult containing the extracted keys and metadata.
        """
        pass


class Judge(ABC):
    """Abstract interface for semantic key verification.
    
    Implementations evaluate whether extracted semantic keys
    faithfully represent the original message content.
    """
    
    @abstractmethod
    def evaluate(
        self,
        original_text: str,
        compressed_text: str,
        keys: list["SemanticKey"],
    ) -> "JudgeResult":
        """Evaluate the fidelity of extracted semantic keys.
        
        Args:
            original_text: The original message before compression.
            compressed_text: The compressed version of the message.
            keys: The semantic keys extracted from the compressed text.
            
        Returns:
            JudgeResult with pass/fail status, confidence, and any issues.
        """
        pass


class EventEmitter(ABC):
    """Abstract interface for pipeline event emission.
    
    Implementations broadcast events to subscribers for
    real-time monitoring and visualization.
    """
    
    @abstractmethod
    async def emit(self, event_type: str, data: dict) -> None:
        """Emit an event to all subscribers.
        
        Args:
            event_type: The type of event being emitted.
            data: Event payload data.
        """
        pass
