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
    # Detailed metrics
    total_entropy: float
    retained_entropy: float
    total_importance: float
    retained_importance: float
    nodes_by_type: Dict[str, int]
    avg_node_importance: float
    compression_stats: Dict[str, Any]


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
        
        # Adaptive initial entropy based on message length
        # For long messages, we need to preserve more information to hit 30% compression with 80% similarity
        if original_tokens > 1000:
            current_entropy_target = 0.90  # Start at 90% for long messages (preserve almost everything)
            print(f"   📊 Long message detected - starting at 90% entropy to preserve fidelity")
        else:
            current_entropy_target = self.initial_entropy_target
        
        iterations: List[IterationResult] = []
        
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
            
            # Calculate detailed metrics
            compression_stats = self.compressor.get_compression_stats(graph, compressed)
            nodes_by_type = {}
            for node in compressed.nodes.values():
                node_type = node.node_type.value
                nodes_by_type[node_type] = nodes_by_type.get(node_type, 0) + 1
            
            avg_importance = (compressed.total_importance() / compressed.node_count() 
                            if compressed.node_count() > 0 else 0)
            
            # Store iteration result with ALL data
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
                graph=compressed,
                total_entropy=graph.total_entropy(),
                retained_entropy=compressed.total_entropy(),
                total_importance=graph.total_importance(),
                retained_importance=compressed.total_importance(),
                nodes_by_type=nodes_by_type,
                avg_node_importance=avg_importance,
                compression_stats=compression_stats
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
                    current_entropy_target = min(current_entropy_target + self.entropy_step, 0.95)
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
        boost_factor = 1.5  # Increase importance by 50% (was 30%, now more aggressive)
        
        # Extract key terms from missing concepts (remove common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        missing_terms = set()
        for concept in missing_concepts:
            terms = concept.lower().split()
            missing_terms.update(term for term in terms if term not in stop_words and len(term) > 2)
        
        for node in graph.nodes.values():
            node_content_lower = node.content.lower()
            node_terms = set(term for term in node_content_lower.split() if term not in stop_words and len(term) > 2)
            
            # Check for term overlap
            overlap = missing_terms & node_terms
            if overlap:
                old_importance = node.importance
                # Boost more if there's significant overlap
                overlap_ratio = len(overlap) / max(len(node_terms), 1)
                boost = boost_factor if overlap_ratio > 0.3 else 1.2
                node.importance = min(node.importance * boost, 1.0)
                if node.importance > old_importance:
                    boosted_count += 1
        
        return boosted_count
    
    def save_results(self, result: PipelineResult, output_dir: str = "results"):
        """Save pipeline results with graphs and visualizations.
        
        Args:
            result: Pipeline result to save
            output_dir: Directory to save all results
        """
        from pathlib import Path
        from .visualizer import GraphVisualizer
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save JSON results with ALL data
        output_data = {
            "success": result.success,
            "final_similarity": result.final_similarity,
            "final_compression": result.final_compression,
            "original_tokens": result.original_tokens,
            "final_tokens": result.final_tokens,
            "total_iterations": len(result.iterations),
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
                    "decoded_message": iter.decoded_message,
                    # Detailed metrics
                    "total_entropy": iter.total_entropy,
                    "retained_entropy": iter.retained_entropy,
                    "entropy_retention": iter.retained_entropy / iter.total_entropy if iter.total_entropy > 0 else 0,
                    "total_importance": iter.total_importance,
                    "retained_importance": iter.retained_importance,
                    "importance_retention": iter.retained_importance / iter.total_importance if iter.total_importance > 0 else 0,
                    "nodes_by_type": iter.nodes_by_type,
                    "avg_node_importance": iter.avg_node_importance,
                    "compression_stats": iter.compression_stats,
                    # Full graph data
                    "graph_data": {
                        "nodes": [
                            {
                                "id": node.id,
                                "content": node.content,
                                "type": node.node_type.value,
                                "importance": node.importance,
                                "entropy": node.entropy,
                                "metadata": node.metadata
                            }
                            for node in iter.graph.nodes.values()
                        ],
                        "edges": [
                            {
                                "source": edge.source,
                                "target": edge.target,
                                "relation": edge.relation,
                                "weight": edge.weight
                            }
                            for edge in iter.graph.edges
                        ]
                    }
                }
                for iter in result.iterations
            ]
        }
        
        json_path = output_path / "results.json"
        json_path.write_text(json.dumps(output_data, indent=2), encoding='utf-8')
        print(f"\n💾 Results saved to: {json_path}")
        
        # Create visualizations for each iteration
        visualizer = GraphVisualizer()
        viz_dir = output_path / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        print(f"\n🎨 Creating visualizations...")
        for iter in result.iterations:
            viz_file = viz_dir / f"iteration_{iter.iteration}.html"
            visualizer.visualize(
                iter.graph,
                output_path=str(viz_file),
                title=f"Iteration {iter.iteration} - Similarity: {iter.similarity_score:.1%}, Compression: {iter.compression_ratio:.1%}"
            )
            print(f"   Iteration {iter.iteration}: {viz_file.name}")
        
        # Create comparison HTML
        self._create_comparison_html(result, output_path)
        
        print(f"\n✅ All results saved to: {output_path}/")
        print(f"   📊 Open {output_path}/comparison.html to review all iterations")
    
    def _create_comparison_html(self, result: PipelineResult, output_dir: Path):
        """Create a comprehensive comparison HTML showing all iterations."""
        
        # Build iteration cards HTML
        iteration_cards = ""
        for iter in result.iterations:
            status_emoji = "✅" if iter.similarity_score >= self.target_similarity else "🔄"
            iteration_cards += f"""
            <div class="iteration-card">
                <h3>{status_emoji} Iteration {iter.iteration}</h3>
                <div class="metrics">
                    <div class="metric">
                        <span class="label">Entropy Target:</span>
                        <span class="value">{iter.entropy_target:.0%}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Nodes:</span>
                        <span class="value">{iter.nodes_kept}/{iter.total_nodes}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Similarity:</span>
                        <span class="value" style="color: {'#28a745' if iter.similarity_score >= self.target_similarity else '#dc3545'}">
                            {iter.similarity_score:.1%}
                        </span>
                    </div>
                    <div class="metric">
                        <span class="label">Compression:</span>
                        <span class="value">{iter.compression_ratio:.1%}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Tokens:</span>
                        <span class="value">{iter.decoded_tokens}</span>
                    </div>
                </div>
                <div class="missing-concepts">
                    <strong>Missing Concepts ({len(iter.missing_concepts)}):</strong>
                    <ul>
                        {''.join(f'<li>{concept[:80]}...</li>' if len(concept) > 80 else f'<li>{concept}</li>' for concept in iter.missing_concepts[:5])}
                        {f'<li><em>...and {len(iter.missing_concepts) - 5} more</em></li>' if len(iter.missing_concepts) > 5 else ''}
                    </ul>
                </div>
                <div class="decoded-preview">
                    <strong>Decoded Message:</strong>
                    <p>{iter.decoded_message[:200]}{'...' if len(iter.decoded_message) > 200 else ''}</p>
                </div>
                <a href="visualizations/iteration_{iter.iteration}.html" target="_blank" class="view-graph-btn">
                    View Graph →
                </a>
            </div>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Iterative Compression Results</title>
            <meta charset="utf-8">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', system-ui, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 40px 20px;
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                .header {{
                    background: white;
                    padding: 40px;
                    border-radius: 16px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    margin-bottom: 30px;
                }}
                h1 {{
                    color: #2d3748;
                    font-size: 36px;
                    margin-bottom: 20px;
                }}
                .summary {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-top: 30px;
                }}
                .summary-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 12px;
                    text-align: center;
                }}
                .summary-card .label {{
                    font-size: 14px;
                    opacity: 0.9;
                    margin-bottom: 8px;
                }}
                .summary-card .value {{
                    font-size: 32px;
                    font-weight: bold;
                }}
                .iterations-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                    gap: 20px;
                }}
                .iteration-card {{
                    background: white;
                    padding: 30px;
                    border-radius: 16px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    transition: transform 0.2s, box-shadow 0.2s;
                }}
                .iteration-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 15px 40px rgba(0,0,0,0.3);
                }}
                .iteration-card h3 {{
                    color: #2d3748;
                    margin-bottom: 20px;
                    font-size: 24px;
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                    margin-bottom: 20px;
                }}
                .metric {{
                    display: flex;
                    flex-direction: column;
                    gap: 5px;
                }}
                .metric .label {{
                    font-size: 12px;
                    color: #718096;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                .metric .value {{
                    font-size: 20px;
                    font-weight: bold;
                    color: #2d3748;
                }}
                .missing-concepts {{
                    background: #fff5f5;
                    border-left: 4px solid #fc8181;
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 4px;
                }}
                .missing-concepts ul {{
                    margin-top: 10px;
                    margin-left: 20px;
                    font-size: 13px;
                    color: #742a2a;
                }}
                .missing-concepts li {{
                    margin: 5px 0;
                }}
                .decoded-preview {{
                    background: #f7fafc;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                    font-size: 14px;
                    color: #4a5568;
                    line-height: 1.6;
                }}
                .view-graph-btn {{
                    display: inline-block;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 12px 24px;
                    border-radius: 8px;
                    text-decoration: none;
                    font-weight: 600;
                    transition: transform 0.2s;
                }}
                .view-graph-btn:hover {{
                    transform: scale(1.05);
                }}
                .status-badge {{
                    display: inline-block;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-weight: 600;
                    font-size: 14px;
                }}
                .status-success {{
                    background: #c6f6d5;
                    color: #22543d;
                }}
                .status-failed {{
                    background: #fed7d7;
                    color: #742a2a;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔬 Adaptive Iterative Graph Compression Results</h1>
                    <span class="status-badge {'status-success' if result.success else 'status-failed'}">
                        {'✅ SUCCESS' if result.success else '⚠️ MAX ITERATIONS REACHED'}
                    </span>
                    
                    <div class="summary">
                        <div class="summary-card">
                            <div class="label">Total Iterations</div>
                            <div class="value">{len(result.iterations)}</div>
                        </div>
                        <div class="summary-card">
                            <div class="label">Final Similarity</div>
                            <div class="value">{result.final_similarity:.1%}</div>
                        </div>
                        <div class="summary-card">
                            <div class="label">Final Compression</div>
                            <div class="value">{result.final_compression:.1%}</div>
                        </div>
                        <div class="summary-card">
                            <div class="label">Token Reduction</div>
                            <div class="value">{result.original_tokens} → {result.final_tokens}</div>
                        </div>
                    </div>
                </div>
                
                <div class="iterations-grid">
                    {iteration_cards}
                </div>
            </div>
        </body>
        </html>
        """
        
        comparison_path = output_dir / "comparison.html"
        comparison_path.write_text(html_content, encoding='utf-8')
        print(f"   📊 Comparison page: comparison.html")
