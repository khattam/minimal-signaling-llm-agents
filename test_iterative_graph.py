"""Test the iterative graph-based compression pipeline."""
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from minimal_signaling.groq_client import GroqClient
from minimal_signaling.encoding.graph_based import IterativeGraphPipeline
from minimal_signaling.encoding.graph_based.visualizer import GraphVisualizer

async def test_iterative():
    # Test message - technical debugging scenario
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
    
    groq = GroqClient()
    
    # Create pipeline with adaptive settings
    pipeline = IterativeGraphPipeline(
        groq_client=groq,
        target_similarity=0.80,      # 80% similarity target
        initial_entropy_target=0.40,  # Start aggressive (40% compression)
        max_iterations=5,
        entropy_step=0.10             # Relax by 10% each iteration if needed
    )
    
    # Run compression
    result = await pipeline.compress(message)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Success: {result.success}")
    print(f"Iterations: {len(result.iterations)}")
    print(f"Final Similarity: {result.final_similarity:.1%}")
    print(f"Final Compression: {result.final_compression:.1%}")
    print(f"Tokens: {result.original_tokens} → {result.final_tokens}")
    
    print(f"\n📊 Iteration Summary:")
    for iter in result.iterations:
        print(f"  Iter {iter.iteration}: "
              f"entropy={iter.entropy_target:.0%}, "
              f"nodes={iter.nodes_kept}/{iter.total_nodes}, "
              f"similarity={iter.similarity_score:.1%}, "
              f"compression={iter.compression_ratio:.1%}")
    
    print(f"\n📝 Final Decoded Message:")
    print(f"{result.final_message}")
    
    # Save results
    pipeline.save_results(result, "iterative_results.json")
    
    # Visualize final iteration
    if result.iterations:
        final_iter = result.iterations[-1]
        visualizer = GraphVisualizer()
        
        # Get original graph from first iteration
        original_graph = result.iterations[0].graph
        # Note: We'd need to store the original uncompressed graph to visualize properly
        # For now, just visualize the final compressed graph
        visualizer.visualize(
            final_iter.graph,
            output_path="graph_viz/final_iteration.html",
            title=f"Final Iteration (Similarity: {final_iter.similarity_score:.1%})"
        )
        print(f"\n🎨 Visualization saved to: graph_viz/final_iteration.html")

if __name__ == "__main__":
    asyncio.run(test_iterative())
