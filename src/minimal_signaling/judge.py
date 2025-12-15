"""Judge verification layer for semantic key fidelity checking."""

from typing import List

from .interfaces import Judge
from .models import SemanticKey, JudgeResult


class PlaceholderJudge(Judge):
    """Placeholder judge implementation for initial development.
    
    Returns configurable pass/fail results with confidence scores.
    This is a simple stub that will be replaced with an LLM-based
    judge in future iterations.
    """
    
    def __init__(
        self,
        default_pass: bool = True,
        default_confidence: float = 0.85,
        min_keys_threshold: int = 0
    ):
        """Initialize the placeholder judge.
        
        Args:
            default_pass: Whether to pass by default
            default_confidence: Default confidence score
            min_keys_threshold: Minimum number of keys required to pass
        """
        self.default_pass = default_pass
        self.default_confidence = default_confidence
        self.min_keys_threshold = min_keys_threshold
    
    def evaluate(self, original: str, keys: List[SemanticKey]) -> JudgeResult:
        """Evaluate if semantic keys faithfully represent the original.
        
        Args:
            original: Original message text
            keys: Extracted semantic keys
            
        Returns:
            JudgeResult with pass/fail and confidence
        """
        issues: List[str] = []
        passed = self.default_pass
        confidence = self.default_confidence
        
        # Simple heuristic checks
        if len(keys) < self.min_keys_threshold:
            issues.append(
                f"Too few keys extracted: {len(keys)} < {self.min_keys_threshold}"
            )
            passed = False
            confidence = max(0.0, confidence - 0.2)
        
        # Check if original is empty but keys exist
        if not original.strip() and len(keys) > 0:
            issues.append("Keys extracted from empty text")
            passed = False
            confidence = 0.0
        
        # Check if keys have empty values
        empty_value_keys = [k for k in keys if not k.value.strip()]
        if empty_value_keys:
            issues.append(f"Found {len(empty_value_keys)} keys with empty values")
            passed = False
            confidence = max(0.0, confidence - 0.3)
        
        return JudgeResult(
            passed=passed,
            confidence=confidence,
            issues=issues
        )


class LLMJudge(Judge):
    """LLM-based judge for semantic key verification (future implementation).
    
    This will use an LLM to evaluate whether the extracted semantic keys
    faithfully represent the original message content.
    """
    
    def __init__(self, model: str):
        """Initialize the LLM judge.
        
        Args:
            model: Model identifier for the LLM
        """
        self.model = model
        # TODO: Load model
    
    def evaluate(self, original: str, keys: List[SemanticKey]) -> JudgeResult:
        """Evaluate using LLM.
        
        Args:
            original: Original message text
            keys: Extracted semantic keys
            
        Returns:
            JudgeResult with pass/fail and confidence
        """
        # TODO: Implement LLM-based evaluation
        raise NotImplementedError("LLM judge not yet implemented")
