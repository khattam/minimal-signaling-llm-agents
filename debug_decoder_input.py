"""Debug script to see exactly what data is sent to the decoder."""

import asyncio
from dotenv import load_dotenv
from src.minimal_signaling.groq_client import GroqClient
from src.minimal_signaling.encoding.graph_based.graph_encoder import GraphEncoder
from src.minimal_signaling.encoding.graph_based.graph_compressor import GraphCompressor
from test_long_message import LONG_MESSAGE

load_dotenv()

async def main():
    print("="*80)
    print("DEBUGGING DECODER INPUT")
    print("="*80)
    
    # Step 1: Extract graph
    client = GroqClient()
    encoder = GraphEncoder(client, use_spacy=False)
    
    print("\n1. ENCODING ORIGINAL MESSAGE...")
    print(f"Original message length: {len(LONG_MESSAGE)} chars, ~1763 tokens")
    print(f"\nFirst 200 chars of original:")
    print(LONG_MESSAGE[:200])
    print("...")
    
    graph = await encoder.encode(LONG_MESSAGE)
    print(f"\n2. GRAPH EXTRACTED:")
    print(f"   Total nodes: {graph.node_count()}")
    print(f"   Total edges: {graph.edge_count()}")
    
    # Step 2: Compress
    compressor = GraphCompressor()
    compressed = compressor.compress(graph, target_ratio=0.90)
    
    print(f"\n3. GRAPH COMPRESSED (90% entropy):")
    print(f"   Nodes: {graph.node_count()} → {compressed.node_count()}")
    
    # Step 3: Show what decoder receives
    print(f"\n4. DATA SENT TO DECODER:")
    print(f"   Number of nodes: {compressed.node_count()}")
    print(f"\n   NODE CONTENTS:")
    for i, node in enumerate(compressed.nodes.values(), 1):
        print(f"   {i}. [{node.node_type.value}] {node.content}")
    
    print(f"\n5. WHAT THE PROMPT LOOKS LIKE:")
    all_nodes_text = []
    for node in compressed.nodes.values():
        all_nodes_text.append(f"- {node.content} ({node.node_type.value})")
    
    prompt_preview = f"""You are reconstructing a message from a semantic graph with {compressed.node_count()} nodes.

Here are ALL the nodes you MUST include in your reconstruction:

{chr(10).join(all_nodes_text[:10])}
... and {compressed.node_count() - 10} more nodes

CRITICAL RULES:
1. You MUST address EVERY node listed above in your reconstruction
2. Keep numbers EXACT
3. NO fluff phrases
4. Write in complete paragraphs

Reconstruct the complete message ensuring you include information from ALL {compressed.node_count()} nodes."""
    
    print(prompt_preview)
    
    print(f"\n{'='*80}")
    print("ANALYSIS:")
    print(f"- Original message: ~1763 tokens")
    print(f"- Nodes extracted: {graph.node_count()}")
    print(f"- Nodes after compression: {compressed.node_count()}")
    print(f"- Expected output: ~{compressed.node_count() * 30} tokens (if 30 tokens per node)")
    print(f"- Actual output from previous test: 441 tokens")
    print(f"\nPROBLEM: Even with {compressed.node_count()} nodes of data, decoder only produces 441 tokens!")
    print(f"This suggests the decoder is SUMMARIZING instead of RECONSTRUCTING")

if __name__ == "__main__":
    asyncio.run(main())
