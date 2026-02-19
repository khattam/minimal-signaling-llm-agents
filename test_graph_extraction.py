"""Test ONLY graph extraction to analyze quality without wasting LLM calls on full pipeline."""
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from minimal_signaling.groq_client import GroqClient
from minimal_signaling.encoding.graph_based import GraphEncoder
from minimal_signaling.tokenization import TiktokenTokenizer

# Short test message first
SHORT_MESSAGE = """Hey team, we're experiencing critical performance issues in production. 
The API response times have spiked to 8-12 seconds for the /users/search endpoint. 
Looking at the logs, I can see we're hitting the database connection pool limit of 50 
connections, and there are 200+ queries queued up. The Redis cache hit rate has also 
dropped from 95% to 23%. Please investigate this ASAP and update me every 30 minutes."""

async def test_extraction(message, message_name):
    print(f"\n{'='*80}")
    print(f"TESTING GRAPH EXTRACTION: {message_name}")
    print(f"{'='*80}")
    
    tokenizer = TiktokenTokenizer()
    token_count = tokenizer.count_tokens(message)
    print(f"\nOriginal message: {token_count} tokens")
    print(f"\n{message[:200]}...\n")
    
    groq = GroqClient()
    encoder = GraphEncoder(groq, use_spacy=True)
    
    print("🔄 Extracting semantic graph...")
    graph = await encoder.encode(message)
    
    print(f"\n📊 Graph Statistics:")
    print(f"   Total nodes: {graph.node_count()}")
    print(f"   Total edges: {graph.edge_count()}")
    print(f"   Total entropy: {graph.total_entropy():.2f} bits")
    print(f"   Total importance: {graph.total_importance():.2f}")
    
    # Group nodes by type
    from minimal_signaling.encoding.graph_based.semantic_graph import NodeType
    
    print(f"\n{'─'*80}")
    print("NODES BY TYPE (sorted by importance)")
    print(f"{'─'*80}")
    
    for node_type in NodeType:
        nodes = graph.get_nodes_by_type(node_type)
        if nodes:
            # Sort by importance
            nodes_sorted = sorted(nodes, key=lambda n: n.importance, reverse=True)
            print(f"\n{node_type.value.upper()} ({len(nodes)} nodes):")
            for node in nodes_sorted:
                print(f"  [{node.importance:.2f}] {node.content}")
                if node.metadata:
                    print(f"       metadata: {node.metadata}")
    
    print(f"\n{'─'*80}")
    print("EDGES (relationships)")
    print(f"{'─'*80}")
    
    # Show first 20 edges
    for i, edge in enumerate(graph.edges[:20]):
        source_node = graph.get_node(edge.source)
        target_node = graph.get_node(edge.target)
        if source_node and target_node:
            print(f"  {source_node.content[:30]:30} --[{edge.relation}]--> {target_node.content[:30]}")
    
    if len(graph.edges) > 20:
        print(f"  ... and {len(graph.edges) - 20} more edges")
    
    print(f"\n{'─'*80}")
    print("ANALYSIS QUESTIONS:")
    print(f"{'─'*80}")
    print("\n1. Are numbers kept WITH their context?")
    print("   Example: '8-12 seconds' should be ONE node, not separate '8', '12', 'seconds'")
    print("\n2. Are the INTENT nodes meaningful?")
    print("   They should capture what actions are being requested")
    print("\n3. Are ENTITY nodes specific enough?")
    print("   'Redis cache hit rate' is good, just 'Redis' is too vague")
    print("\n4. Are there too many meaningless nodes?")
    print("   Random percentages or numbers without context are useless")
    print("\n5. Do the edges make sense?")
    print("   Relationships should be logical and meaningful")
    
    return graph

async def main():
    # Test with long message
    print("\n" + "="*80)
    print("TEST: LONG MESSAGE (1763 TOKENS)")
    print("="*80)
    from test_long_message import LONG_MESSAGE
    await test_extraction(LONG_MESSAGE, "Long Message")

if __name__ == "__main__":
    asyncio.run(main())
