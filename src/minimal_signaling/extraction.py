"""Stage 2: Semantic Key Extraction from compressed text."""

import re
from typing import List

from .interfaces import SemanticKeyExtractor
from .models import SemanticKey, ExtractionResult, KeyType


class PlaceholderExtractor(SemanticKeyExtractor):
    """Deterministic placeholder extractor for initial development.
    
    Parses text for key patterns like:
    - INSTRUCTION: <value>
    - STATE: <value>
    - GOAL: <value>
    - CONTEXT: <value>
    - CONSTRAINT: <value>
    
    This is a simple rule-based extractor that will be replaced
    with an LLM-based extractor in future iterations.
    """
    
    def __init__(self, schema_version: str = "1.0"):
        """Initialize the placeholder extractor.
        
        Args:
            schema_version: Schema version for extracted keys
        """
        self.schema_version = schema_version
        
        # Pattern to match key declarations
        # Matches: "INSTRUCTION: value" or "STATE: value" etc.
        self.key_pattern = re.compile(
            r'(INSTRUCTION|STATE|GOAL|CONTEXT|CONSTRAINT)\s*:\s*(.+?)(?=\n|$)',
            re.IGNORECASE | re.MULTILINE
        )
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract semantic keys from compressed text.
        
        Args:
            text: Compressed text to extract keys from
            
        Returns:
            ExtractionResult with extracted keys
        """
        if not text.strip():
            return ExtractionResult(
                keys=[],
                schema_version=self.schema_version,
                raw_output=text
            )
        
        keys: List[SemanticKey] = []
        
        # Find all key patterns in text
        matches = self.key_pattern.findall(text)
        
        for key_type_str, value in matches:
            # Convert string to KeyType enum
            try:
                key_type = KeyType[key_type_str.upper()]
                key = SemanticKey(
                    type=key_type,
                    value=value.strip()
                )
                keys.append(key)
            except (KeyError, ValueError):
                # Skip invalid key types
                continue
        
        return ExtractionResult(
            keys=keys,
            schema_version=self.schema_version,
            raw_output=text
        )


class LLMExtractor(SemanticKeyExtractor):
    """LLM-based semantic key extraction (future implementation).
    
    This will use a compact LLM to extract semantic keys from
    compressed text in a more intelligent way than the placeholder.
    """
    
    def __init__(self, model: str, schema_version: str = "1.0"):
        """Initialize the LLM extractor.
        
        Args:
            model: Model identifier for the LLM
            schema_version: Schema version for extracted keys
        """
        self.model = model
        self.schema_version = schema_version
        # TODO: Load model
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract semantic keys using LLM.
        
        Args:
            text: Compressed text to extract keys from
            
        Returns:
            ExtractionResult with extracted keys
        """
        # TODO: Implement LLM-based extraction
        raise NotImplementedError("LLM extraction not yet implemented")
