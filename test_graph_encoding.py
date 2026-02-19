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
    # Test message - completely different domain (technical debugging scenario)
    message = """Hey team, we're experiencing critical performance issues in production. 
    The API response times have spiked to 8-12 seconds for the /users/search endpoint, 
    which is completely unacceptable. This started happening around 3 AM UTC this morning.
    
    Looking at the logs, I can see we're hitting the database connection pool limit of 50 
    connections, and there are 200+ queries queued up. The Redis cache hit rate has also 
    dropped from 95% to 23%, which suggests something is bypassing the cache layer.
    
    I need someone to investigate this ASAP. Priority tasks:
    1. Check if there was a recent deployment that might have introduced this regression
    2. Analyze the slow query logs to identify which queries are causing the bottleneck  
    3. Verify that the Redis cluster is healthy and not experiencing any network issues
    4. Review the connection pool configuration - we might need to increase it temporarily
    
    Customer support is getting flooded with complaints, and we have 3 enterprise clients 
    threatening to churn if this isn't fixed by EOD. The SLA breach is going to cost us 
    $50K in credits. Please update me every 30 minutes with progress. This is P0."""
    
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
