"""Iterative Graph-Based Compression Pipeline with Adaptive Refinement.

This pipeline implements adaptive compression with feedback:
1. Start with aggressive compression (40% entropy)
2. If fidelity is low, boost node importance AND relax compression
3. Iterate until target fidelity (80%) or max iterations (5)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ...groq_client import GroqClient
from ...semantic_judge import SemanticJudge
from ...tokenization import TiktokenTokenizer
from .graph_encoder import GraphEncoder
from .graph_compressor import GraphCompressor
from .graph_decoder import GraphDecoder
from .semantic_graph import SemanticGraph, SemanticNode


@dataclass
class IterationResult:
    """Results from a single iteration."""
    iteration: int
    entropy_target: float
    nodes_kept: int
    total_nodes: int
    decoded_tokens: int
    original_tokens: int
    compression_ratio: float
    similarity_score: float
    decoded_message: str
    missing_concepts: List[str]
    graph: SemanticGraph


@dataclass
class PipelineResult:
    """Final results from the pipeline."""
    success: bool
    iterations: List[IterationResult]
    final_message: str
    final_similarity: float
    final_compression: float
    original_tokens: int
    final_tokens: int


class IterativeGraphPipeline:
    """Adaptive iterative graph compression pipeline."""
    
    def __init__(
        self,
        groq_client: GroqClient,
        target_similarity: float = 0.80,
        initial_entropy_target: float = 0.40,
        max_iterations: int = 5,
        entropy_step: float = 0.10
    ):
        """Initialize pipeline.
        
        Args:
            groq_client: Groq client for LLM calls
            target_similarity: Target semantic similarity (0-1)
            initial_entropy_target: Starting compression ratio (0-1)
            max_iterations: Maximum refinement iterations
            entropy_step: How much to relax compression each iteration
        """
        self.client = groq_client
        self.target_similarity = target_similarity
        self.initial_entropy_target = initial_entropy_target
        self.max_iterations = max_iterations
        self.entropy_step = entropy_step
        
        self.encoder = GraphEncoder(groq_client, use_spacy=True)
        self.compressor = GraphCompressor()
        self.decoder = GraphDecoder(groq_client)
        self.judge = SemanticJudge(threshold=target_similarity)
        self.tokenizer = TiktokenTokenizer()
    
    async def compress(self, message: str) -> PipelineResult:
        """Compress message with adaptive iterative refinement.
        
        Args:
            message: Original message to compress
            
        Returns:
            PipelineResult with all iteration details
        """
        print(f"\n{'='*80}")
        print(f"ADAPTIVE ITERATIVE GRAPH COMPRESSION")
        print(f"{'='*80}")
        
        original_tokens = self.tokenizer.count_tokens(message)
        print(f"\n📝 Original message: {original_tokens} tokens")
        
        # Encode to graph (only once)
        print(f"\n🔄 Encoding to semantic graph...")
        graph = await self.encoder.encode(message)
        print(f"   Extracted {graph.node_count()} nodes, {graph.edge_count()} edges")
        
        iterations: List[IterationResult] = []
        current_entropy_target = self.initial_entropy_target
        
        # Iterative refinement
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'─'*80}")
            print(f"ITERATION {iteration}")
            print(f"{'─'*80}")
            print(f"🎯 Entropy target: {current_entropy_target:.0%}")
            
            # Compress graph
            compressed = self.compressor.compress(graph, target_ratio=current_entropy_target)
            print(f"🗜️  Compressed: {graph.node_count()} → {compressed.node_count()} nodes")
            
            # Decode
            decoded = await self.decoder.decode(compressed)
            decoded_tokens = self.tokenizer.count_tokens(decoded)
            compression_ratio = decoded_tokens / original_tokens
            print(f"📤 Decoded: {decoded_tokens} tokens ({compression_ratio:.1%} of original)")
            
            # Judge similarity
            judge_result = self.judge.evaluate(message, decoded)
            similarity = judge_result.similarity_score
            print(f"⚖️  Similarity: {similarity:.1%} (target: {self.target_similarity:.0%})")
            
            # Analyze what's missing
            missing_concepts = await self._analyze_loss(message, decoded)
            if missing_concepts:
                print(f"❌ Missing concepts: {len(missing_concepts)}")
                for concept in missing_concepts[:3]:
                    print(f"   - {concept}")
            
            # Store iteration result
            iter_result = IterationResult(
                iteration=iteration,
                entropy_target=current_entropy_target,
                nodes_kept=compressed.node_count(),
                total_nodes=graph.node_count(),
                decoded_tokens=decoded_tokens,
                original_tokens=original_tokens,
                compression_ratio=compression_ratio,
                similarity_score=similarity,
                decoded_message=decoded,
                missing_concepts=missing_concepts,
                graph=compressed
            )
            iterations.append(iter_result)
            
            # Check if we hit target
            if similarity >= self.target_similarity:
                print(f"\n✅ SUCCESS! Achieved {similarity:.1%} similarity in {iteration} iterations")
                return PipelineResult(
                    success=True,
                    iterations=iterations,
                    final_message=decoded,
                    final_similarity=similarity,
                    final_compression=compression_ratio,
                    original_tokens=original_tokens,
                    final_tokens=decoded_tokens
                )
            
            # Adaptive refinement for next iteration
            if iteration < self.max_iterations:
                print(f"\n🔧 Refining for next iteration...")
                
                # Boost importance of nodes related to missing concepts
                boosted_count = self._boost_importance(graph, missing_concepts)
                print(f"   Boosted importance of {boosted_count} nodes")
                
                # Relax compression if similarity is low
                if similarity < 0.70:
                    current_entropy_target = min(current_entropy_target + self.entropy_step, 0.80)
                    print(f"   Relaxed entropy target to {current_entropy_target:.0%}")
                else:
                    print(f"   Keeping entropy target at {current_entropy_target:.0%}")
        
        # Max iterations reached
        final_iter = iterations[-1]
        print(f"\n⚠️  Max iterations reached. Final similarity: {final_iter.similarity_score:.1%}")
        
        return PipelineResult(
            success=False,
            iterations=iterations,
            final_message=final_iter.decoded_message,
            final_similarity=final_iter.similarity_score,
            final_compression=final_iter.compression_ratio,
            original_tokens=original_tokens,
            final_tokens=final_iter.decoded_tokens
        )
    
    async def _analyze_loss(self, original: str, decoded: str) -> List[str]:
        """Analyze what information was lost in compression.
        
        Args:
            original: Original message
            decoded: Decoded message
            
        Returns:
            List of missing concepts
        """
        prompt = f"""Compare these two messages and identify key information that is present in the ORIGINAL but missing or significantly altered in the DECODED version.

ORIGINAL:
{original}

DECODED:
{decoded}

List the missing or altered concepts as a JSON array of strings. Focus on:
- Specific numbers, dates, or metrics that are missing
- Important entities or names that are missing
- Key actions or requirements that are missing
- Critical constraints or deadlines that are missing

Output format: {{"missing_concepts": ["concept 1", "concept 2", ...]}}

Output ONLY valid JSON."""
        
        try:
            response = await self.client.chat(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Analyze the loss."}
                ],
                json_mode=True,
                temperature=0.0
            )
            result = json.loads(response)
            return result.get("missing_concepts", [])
        except Exception as e:
            print(f"Warning: Loss analysis failed: {e}")
            return []
    
    def _boost_importance(self, graph: SemanticGraph, missing_concepts: List[str]) -> int:
        """Boost importance of nodes related to missing concepts.
        
        Args:
            graph: Semantic graph to update
            missing_concepts: List of missing concepts
            
        Returns:
            Number of nodes boosted
        """
        if not missing_concepts:
            return 0
        
        boosted_count = 0
        boost_factor = 1.3  # Increase importance by 30%
        
        for node in graph.nodes.values():
            # Check if node content relates to any missing concept
            node_content_lower = node.content.lower()
            for concept in missing_concepts:
                concept_lower = concept.lower()
                
                # Simple substring matching (could be improved with embeddings)
                if concept_lower in node_content_lower or node_content_lower in concept_lower:
                    old_importance = node.importance
                    node.importance = min(node.importance * boost_factor, 1.0)
                    if node.importance > old_importance:
                        boosted_count += 1
                    break
        
        return boosted_count
    
    def save_results(self, result: PipelineResult, output_path: str):
        """Save pipeline results to JSON file.
        
        Args:
            result: Pipeline result to save
            output_path: Path to save JSON file
        """
        output_data = {
            "success": result.success,
            "final_similarity": result.final_similarity,
            "final_compression": result.final_compression,
            "original_tokens": result.original_tokens,
            "final_tokens": result.final_tokens,
            "iterations": [
                {
                    "iteration": iter.iteration,
                    "entropy_target": iter.entropy_target,
                    "nodes_kept": iter.nodes_kept,
                    "total_nodes": iter.total_nodes,
                    "compression_ratio": iter.compression_ratio,
                    "similarity_score": iter.similarity_score,
                    "decoded_tokens": iter.decoded_tokens,
                    "missing_concepts": iter.missing_concepts,
                    "decoded_message": iter.decoded_message
                }
                for iter in result.iterations
            ]
        }
        
        Path(output_path).write_text(json.dumps(output_data, indent=2), encoding='utf-8')
        print(f"\n💾 Results saved to: {output_path}")
