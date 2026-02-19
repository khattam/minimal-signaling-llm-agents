"""Test the graph-based encoding system."""
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from minimal_signaling.groq_client import GroqClient
from minimal_signaling.encoding.graph_based import (
    GraphEncoder,
    GraphCompressor,
    GraphDecoder,
)
from minimal_signaling.encoding.graph_based.visualizer import GraphVisualizer
from minimal_signaling.tokenization import TiktokenTokenizer

async def test_graph():
    # Test message - realistic agent communication scenario (~350 tokens)
    message = """I need you to conduct a comprehensive analysis of our Q3 2024 sales performance data, 
    with particular focus on the enterprise segment which has shown concerning trends. The preliminary 
    data indicates a 23% decline in enterprise sales compared to Q2, representing approximately $2.3M 
    in lost revenue. This decline is particularly concentrated in the financial services vertical, 
    where we've seen three major clients reduce their contract values.
    
    The board of directors has scheduled an emergency meeting for next Friday, November 15th at 2 PM EST, 
    where we need to present a detailed analysis with actionable recommendations. The CEO specifically 
    requested that we identify the root causes of this decline, assess whether it's a temporary market 
    fluctuation or a systemic issue with our product-market fit, and provide at least 3 concrete action 
    items we can implement immediately.
    
    For context, our main competitors TechCorp and DataSolutions have both reported growth in the same 
    period, so this appears to be a company-specific issue rather than a market-wide trend. Our customer 
    success team has noted increased churn risk flags from 12 enterprise accounts, and our NPS score 
    dropped from 42 to 31 in the last quarter.
    
    Budget constraints are important here - we have $500K allocated for remediation efforts this quarter, 
    but any initiatives requiring more than that will need board approval. The CFO wants to see ROI 
    projections for any proposed solutions. This is being treated as critical priority, so please 
    prioritize this analysis over other ongoing projects. I need the initial findings by Wednesday EOD 
    so we can review before the board presentation."""
    
    print("=" * 80)
    print("GRAPH-BASED SEMANTIC COMPRESSION TEST")
    print("=" * 80)
    
    tokenizer = TiktokenTokenizer()
    groq = GroqClient()
    
    original_tokens = tokenizer.count_tokens(message)
    print(f"\n📝 Original message: {original_tokens} tokens")
    print(f"   {message[:100]}...")
    
    # Encode to graph
    print("\n🔄 Encoding to semantic graph...")
    encoder = GraphEncoder(groq, use_spacy=True)  # Enable spaCy
    graph = await encoder.encode(message)
    
    print(f"\n📊 Graph statistics:")
    print(f"   Nodes: {graph.node_count()}")
    print(f"   Edges: {graph.edge_count()}")
    print(f"   Total entropy: {graph.total_entropy():.2f} bits")
    print(f"   Total importance: {graph.total_importance():.3f}")
    
    # Show nodes by type
    from minimal_signaling.encoding.graph_based.semantic_graph import NodeType
    for node_type in NodeType:
        nodes = graph.get_nodes_by_type(node_type)
        if nodes:
            print(f"\n   {node_type.value.upper()} nodes ({len(nodes)}):")
            for node in nodes[:3]:  # Show first 3
                print(f"      - {node.content[:60]} (importance: {node.importance:.2f})")
    
    # Compress graph
    print("\n🗜️  Compressing graph to 60% of original...")
    compressor = GraphCompressor()
    compressed = compressor.compress(graph, target_ratio=0.6)
    
    stats = compressor.get_compression_stats(graph, compressed)
    print(f"\n   Compression stats:")
    print(f"      Nodes: {stats['original_nodes']} → {stats['compressed_nodes']}")
    print(f"      Entropy retention: {stats['entropy_retention']:.1%}")
    print(f"      Importance retention: {stats['importance_retention']:.1%}")
    
    # Decode compressed graph
    print("\n🔄 Decoding compressed graph...")
    decoder = GraphDecoder(groq)
    decoded = await decoder.decode(compressed)
    
    decoded_tokens = tokenizer.count_tokens(decoded)
    print(f"\n📤 Decoded message: {decoded_tokens} tokens")
    print(f"   Compression: {decoded_tokens/original_tokens:.1%}")
    print(f"\n   {decoded}")
    
    # Visualize
    print("\n🎨 Creating visualizations...")
    visualizer = GraphVisualizer()
    visualizer.visualize_comparison(graph, compressed, output_dir="graph_viz")
    
    print("\n✅ Test complete! Open graph_viz/comparison.html to see visualizations")

if __name__ == "__main__":
    asyncio.run(test_graph())
