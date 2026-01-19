"""Simple test to understand the hierarchical tree."""
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from src.minimal_signaling.groq_client import GroqClient
from src.minimal_signaling.hierarchical_encoder import HierarchicalEncoder

# SIMPLE MESSAGE - easy to understand
SIMPLE_MESSAGE = """
Fix bug #123 for customer Acme Corp. 
Priority: urgent. 
Deadline: 2 hours.
"""

async def main():
    groq = GroqClient(api_key=os.environ["GROQ_API_KEY"])
    encoder = HierarchicalEncoder(groq)
    
    print("=" * 60)
    print("INPUT MESSAGE:")
    print(SIMPLE_MESSAGE)
    print("=" * 60)
    
    result = await encoder.encode(SIMPLE_MESSAGE)
    tree = result.signal.root
    
    print("\nTREE STRUCTURE:")
    print("-" * 60)
    
    def print_node(node, indent=0):
        prefix = "  " * indent
        imp = f"{node.importance*100:.1f}%"
        bits = f"{node.entropy:.1f} bits"
        print(f"{prefix}[{node.level.name}] {node.node_type}: {node.content}")
        print(f"{prefix}  → importance: {imp}, entropy: {bits}")
        for child in node.children:
            print_node(child, indent + 1)
    
    print_node(tree)
    
    print("\n" + "=" * 60)
    print(f"TOTAL NODES: {result.signal.node_count()}")
    print(f"TOTAL BITS: {result.signal.total_entropy():.1f}")
    print(f"TOTAL IMPORTANCE: {result.signal.total_importance()*100:.1f}%")
    print("=" * 60)
    
    # Now show compression
    print("\n\nCOMPRESSION (keep top 3 nodes):")
    print("-" * 60)
    
    from src.minimal_signaling.hierarchical_encoder import HierarchicalCompressor
    compressor = HierarchicalCompressor()
    compressed = compressor.compress(result.signal, preserve_top_k=3)
    
    print_node(compressed.root)
    print(f"\nCompressed to {compressed.node_count()} nodes")
    print(f"Importance preserved: {compressed.total_importance() / result.signal.total_importance() * 100:.0f}%")

if __name__ == "__main__":
    asyncio.run(main())
