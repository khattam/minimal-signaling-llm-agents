"""Semantic Judge - verifies semantic preservation using embeddings."""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .protocol import JudgeResult, JudgeError


class SemanticJudge:
    """Judges semantic fidelity using embedding similarity.
    
    Uses sentence-transformers to compute embeddings and cosine
    similarity to measure semantic preservation.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.80
    ):
        """Initialize semantic judge.
        
        Args:
            model_name: Sentence transformer model to use.
            threshold: Minimum similarity score to pass.
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise JudgeError(f"Failed to load model {model_name}: {e}")
        
        self.threshold = threshold
    
    def evaluate(self, original: str, decoded: str) -> JudgeResult:
        """Evaluate semantic fidelity between original and decoded text.
        
        Args:
            original: Original natural language text.
            decoded: Decoded text from MSP signal.
            
        Returns:
            JudgeResult with similarity score and pass/fail.
            
        Raises:
            JudgeError: If evaluation fails.
        """
        try:
            # Compute embeddings
            emb_original = self.model.encode(original)
            emb_decoded = self.model.encode(decoded)
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                [emb_original],
                [emb_decoded]
            )[0][0]
            
            # Ensure score is in valid range
            similarity = float(np.clip(similarity, 0.0, 1.0))
            
            # Determine pass/fail
            passed = similarity >= self.threshold
            issues = [] if passed else ["Semantic drift detected"]
            
            return JudgeResult(
                passed=passed,
                confidence=similarity,
                similarity_score=similarity,
                issues=issues
            )
            
        except Exception as e:
            raise JudgeError(f"Failed to compute embeddings: {e}")
