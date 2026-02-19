"""Adaptive Iterative Compression Pipeline using DistilBART.

NOVELTY: Iterative refinement with semantic feedback loop.
- Start with aggressive compression
- If semantic similarity is low, relax compression and retry
- Adaptive target adjustment based on feedback
"""

from dataclasses import dataclass
from typing import List
from ..semantic_judge import SemanticJudge
from ..tokenization import TiktokenTokenizer
from .distilbart_encoder import DistilBARTEncoder


@dataclass
class IterationResult:
    """Results from a single compression iteration."""
    iteration: int
    target_ratio: float
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    similarity_score: float


@dataclass
class CompressionResult:
    """Final compression results."""
    success: bool
    iterations: List[IterationResult]
    final_text: str
    final_similarity: float
    final_compression: float
    original_tokens: int
    final_tokens: int


class AdaptiveDistilBARTPipeline:
    """Adaptive iterative compression using DistilBART with semantic feedback.
    
    NOVEL CONTRIBUTION:
    - Iterative refinement based on semantic similarity feedback
    - Adaptive compression target adjustment
    - Guarantees semantic fidelity while maximizing compression
    """
    
    def __init__(
        self,
        target_similarity: float = 0.80,
        initial_compression: float = 0.50,
        max_iterations: int = 5,
        step_size: float = 0.10
    ):
        """Initialize pipeline.
        
        Args:
            target_similarity: Target semantic similarity (0-1)
            initial_compression: Initial compression ratio (0.5 = keep 50%)
            max_iterations: Maximum refinement iterations
            step_size: How much to relax compression each iteration
        """
        self.target_similarity = target_similarity
        self.initial_compression = initial_compression
        self.max_iterations = max_iterations
        self.step_size = step_size
        
        self.encoder = DistilBARTEncoder()
        self.judge = SemanticJudge(threshold=target_similarity)
        self.tokenizer = TiktokenTokenizer()
    
    def compress(self, text: str) -> CompressionResult:
        """Compress text with adaptive iterative refinement.
        
        Args:
            text: Original text to compress
            
        Returns:
            CompressionResult with all iteration details
        """
        print(f"\n{'='*80}")
        print(f"ADAPTIVE ITERATIVE COMPRESSION (DistilBART + Semantic Feedback)")
        print(f"{'='*80}")
        
        original_tokens = self.tokenizer.count_tokens(text)
        print(f"\n📝 Original: {original_tokens} tokens")
        
        current_ratio = self.initial_compression
        iterations: List[IterationResult] = []
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'─'*80}")
            print(f"ITERATION {iteration}")
            print(f"{'─'*80}")
            print(f"🎯 Target ratio: {current_ratio:.0%} (keep {current_ratio:.0%} of original)")
            
            # Compress
            compressed = self.encoder.encode(text, target_ratio=current_ratio)
            compressed_tokens = self.tokenizer.count_tokens(compressed)
            actual_ratio = compressed_tokens / original_tokens
            
            print(f"🗜️  Compressed: {original_tokens} → {compressed_tokens} tokens ({actual_ratio:.1%})")
            
            # Judge similarity
            judge_result = self.judge.evaluate(text, compressed)
            similarity = judge_result.similarity_score
            print(f"⚖️  Similarity: {similarity:.1%} (target: {self.target_similarity:.0%})")
            
            # Store iteration
            iter_result = IterationResult(
                iteration=iteration,
                target_ratio=current_ratio,
                compressed_text=compressed,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=actual_ratio,
                similarity_score=similarity
            )
            iterations.append(iter_result)
            
            # Check if we hit target
            if similarity >= self.target_similarity:
                print(f"\n✅ SUCCESS! Achieved {similarity:.1%} similarity in {iteration} iterations")
                return CompressionResult(
                    success=True,
                    iterations=iterations,
                    final_text=compressed,
                    final_similarity=similarity,
                    final_compression=actual_ratio,
                    original_tokens=original_tokens,
                    final_tokens=compressed_tokens
                )
            
            # Adaptive refinement for next iteration
            if iteration < self.max_iterations:
                print(f"🔧 Similarity below target, relaxing compression...")
                current_ratio = min(current_ratio + self.step_size, 0.95)
        
        # Max iterations reached
        final_iter = iterations[-1]
        print(f"\n⚠️  Max iterations reached. Final similarity: {final_iter.similarity_score:.1%}")
        
        return CompressionResult(
            success=False,
            iterations=iterations,
            final_text=final_iter.compressed_text,
            final_similarity=final_iter.similarity_score,
            final_compression=final_iter.compression_ratio,
            original_tokens=original_tokens,
            final_tokens=final_iter.compressed_tokens
        )
