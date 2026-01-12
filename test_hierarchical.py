"""Test hierarchical semantic encoding with information-theoretic bounds."""
import os
import asyncio
import json

# Load from .env file
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from minimal_signaling.groq_client import GroqClient
from minimal_signaling.hierarchical_encoder import HierarchicalEncoder, HierarchicalCompressor
from minimal_signaling.hierarchical_signal import SemanticLevel


async def test_hierarchical():
    print("=" * 70)
    print("HIERARCHICAL SEMANTIC ENCODING")
    print("With Information-Theoretic Importance Scores")
    print("=" * 70)
    
    # Complex test message
    message = """I've completed my analysis of the customer support ticket backlog and need to coordinate with the escalation team. Here's what I found:

URGENT ISSUES (require immediate attention):
1. Ticket #4521 - Enterprise client Acme Corp experiencing complete service outage for 6+ hours. Their SLA guarantees 99.9% uptime and we're now in breach. They're threatening contract termination worth $2.3M annually. The root cause appears to be a misconfigured load balancer after last night's deployment.

2. Ticket #4518 - Payment processing failures affecting approximately 340 transactions since 2:00 AM EST. Customers are being charged but orders aren't completing. Finance team estimates $47,000 in pending refunds needed.

HIGH PRIORITY (within 24 hours):
3. Tickets #4502, #4507, #4511 - All related to the new authentication flow. Users report being logged out randomly mid-session. Affects roughly 12% of active users based on error logs.

4. Ticket #4499 - Data export feature returning corrupted CSV files for reports over 10,000 rows. Three enterprise clients have reported this.

I need you to:
- Immediately escalate tickets #4521 and #4518 to the on-call engineering team
- Create an incident report for the Acme Corp situation for executive review
- Group the authentication tickets and assign to the identity team
- Verify if the CSV issue is related to the recent database migration

Please confirm receipt and provide ETAs for each action item. I'll continue monitoring incoming tickets and flag anything else critical."""

    print(f"\n📝 Original message: {len(message)} chars")
    print("-" * 70)
    
    # Initialize encoder
    groq = GroqClient()
    encoder = HierarchicalEncoder(groq)
    
    print("\n🔄 Encoding to hierarchical signal...")
    result = await encoder.encode(message)
    
    # Display hierarchy
    print("\n" + "=" * 70)
    print("HIERARCHICAL SIGNAL STRUCTURE")
    print("=" * 70)
    
    def print_node(node, indent=0):
        prefix = "  " * indent
        level_emoji = {
            SemanticLevel.INTENT: "🎯",
            SemanticLevel.ENTITIES: "👤",
            SemanticLevel.ATTRIBUTES: "📊",
            SemanticLevel.DETAILS: "📝"
        }
        emoji = level_emoji.get(node.level, "•")
        
        print(f"{prefix}{emoji} [{node.level.name}] {node.node_type}: {node.content[:50]}...")
        print(f"{prefix}   importance: {node.importance:.4f}, entropy: {node.entropy:.2f} bits")
        
        for child in node.children:
            print_node(child, indent + 1)
    
    print_node(result.signal.root)
    
    # Metrics
    print("\n" + "=" * 70)
    print("INFORMATION-THEORETIC METRICS")
    print("=" * 70)
    
    print(f"\n📊 Signal Statistics:")
    print(f"   Total nodes: {result.signal.node_count()}")
    print(f"   Total entropy: {result.signal.total_entropy():.2f} bits")
    print(f"   Total importance: {result.signal.total_importance():.4f}")
    print(f"   Original tokens: {result.signal.original_tokens}")
    print(f"   Encoded tokens: {result.encoding_tokens}")
    
    print(f"\n📐 Theoretical Bounds:")
    print(f"   Min bits for 80% similarity: {result.theoretical_bound:.2f}")
    print(f"   Compression efficiency: {result.efficiency:.2%}")
    
    # Show Pareto frontier
    print("\n📈 Pareto Frontier (Similarity vs Bits):")
    frontier = encoder.bound_calc.pareto_frontier(result.signal)
    for point in frontier:
        bar = "█" * int(point['compression_ratio'] * 40)
        print(f"   {point['target_similarity']:.0%} sim → {point['minimum_bits']:.1f} bits ({point['compression_ratio']:.1%}) {bar}")
    
    # Test compression
    print("\n" + "=" * 70)
    print("COMPRESSION TEST")
    print("=" * 70)
    
    compressor = HierarchicalCompressor()
    
    # Show compression by keeping top K nodes
    for k in [20, 10, 5]:
        compressed = compressor.compress(result.signal, preserve_top_k=k)
        
        print(f"\n🗜️ Keep top {k} nodes by importance:")
        print(f"   Nodes: {result.signal.node_count()} → {compressed.node_count()}")
        print(f"   Entropy: {result.signal.total_entropy():.1f} → {compressed.total_entropy():.1f} bits")
        print(f"   Importance preserved: {compressed.total_importance() / result.signal.total_importance():.1%}")
        
        # Show which nodes were kept
        kept_nodes = compressed.root.flatten()
        kept_types = [f"{n.node_type}:{n.content[:15]}" for n in kept_nodes]
        print(f"   Kept: {kept_types[:5]}...")
    
    # Show importance ranking
    print("\n" + "=" * 70)
    print("IMPORTANCE RANKING (Top 10)")
    print("=" * 70)
    
    all_nodes = result.signal.root.flatten()
    ranked = sorted(all_nodes, key=lambda n: n.importance, reverse=True)
    
    for i, node in enumerate(ranked[:10], 1):
        print(f"   {i}. [{node.level.name}] {node.node_type}: {node.content[:40]}...")
        print(f"      importance={node.importance:.4f}, entropy={node.entropy:.1f} bits")
    
    # Show JSON output
    print("\n" + "=" * 70)
    print("JSON OUTPUT (truncated)")
    print("=" * 70)
    
    json_output = result.signal.to_json()
    print(json_output[:1500] + "..." if len(json_output) > 1500 else json_output)


if __name__ == "__main__":
    asyncio.run(test_hierarchical())
