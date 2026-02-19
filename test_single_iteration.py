"""Test single iteration to verify decoder output length."""

import asyncio
import os
from dotenv import load_dotenv
from src.minimal_signaling.groq_client import GroqClient
from src.minimal_signaling.encoding.graph_based.iterative_graph_pipeline import IterativeGraphPipeline
from test_long_message import LONG_MESSAGE

load_dotenv()

async def main():
    print(f"Testing decoder with long message (1763 tokens)")
    print(f"Goal: Achieve ~1200 tokens output (70% of original) with 80% similarity\n")
    
    client = GroqClient()
    pipeline = IterativeGraphPipeline(
        groq_client=client,
        target_similarity=0.80,
        initial_entropy_target=0.90,  # Start very high
        max_iterations=5  # Full iterations
    )
    
    result = await pipeline.compress(LONG_MESSAGE)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    for i, iter_result in enumerate(result.iterations, 1):
        print(f"Iteration {i}: {iter_result.nodes_kept} nodes → {iter_result.decoded_tokens} tokens ({iter_result.compression_ratio:.1%}) - Similarity: {iter_result.similarity_score:.1%}")
    
    print(f"\nFinal: {result.final_tokens} tokens ({result.final_compression:.1%}) - Similarity: {result.final_similarity:.1%}")
    
    # Save results
    pipeline.save_results(result, "final_test_results")

if __name__ == "__main__":
    asyncio.run(main())
