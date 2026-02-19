"""Test decoder improvements with a medium-sized message."""

import asyncio
import os
from dotenv import load_dotenv
from src.minimal_signaling.groq_client import GroqClient
from src.minimal_signaling.encoding.graph_based.iterative_graph_pipeline import IterativeGraphPipeline

# Load environment variables
load_dotenv()


async def main():
    # Medium-sized test message (~350 tokens)
    test_message = """
    Our Q3 product launch exceeded expectations with 15,000 new signups in the first week, 
    representing a 230% increase over our initial target of 6,500 users. The mobile app 
    achieved a 4.8/5.0 rating on both iOS and Android platforms, with particularly strong 
    feedback on the new onboarding flow and dashboard redesign.
    
    However, we're experiencing significant infrastructure challenges. Our database query 
    response times have increased from an average of 120ms to 450ms during peak hours 
    (2-4 PM EST), affecting approximately 35% of our user base. The engineering team 
    identified the root cause as inefficient indexing on the user_activity table, which 
    has grown to 2.3 million rows. We need to implement database sharding and optimize 
    our queries by Friday to prevent further degradation.
    
    The customer success team reports a 12% increase in support tickets, primarily related 
    to payment processing issues with Stripe integration. We've identified a race condition 
    in the webhook handler that causes duplicate charge attempts for about 3% of transactions. 
    This has resulted in $23,000 in disputed charges this month. The fix requires updating 
    our idempotency key generation logic and will take 2 days to implement and test.
    
    On the positive side, our enterprise sales pipeline has grown to $1.2M in potential ARR, 
    with 8 qualified leads in the final negotiation stage. The new analytics dashboard feature 
    is driving significant interest, with 67% of enterprise prospects specifically requesting 
    demos of this functionality. We should prioritize the custom reporting module to close 
    these deals before Q4.
    """
    
    print(f"Testing with message of {len(test_message.split())} words")
    
    # Initialize pipeline
    client = GroqClient()
    pipeline = IterativeGraphPipeline(
        groq_client=client,
        target_similarity=0.80,
        initial_entropy_target=0.50,
        max_iterations=3
    )
    
    # Run compression
    result = await pipeline.compress(test_message)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Success: {result.success}")
    print(f"Iterations: {len(result.iterations)}")
    print(f"Final Similarity: {result.final_similarity:.1%}")
    print(f"Final Compression: {result.final_compression:.1%}")
    print(f"Tokens: {result.original_tokens} → {result.final_tokens}")
    print(f"\nDecoded Message:")
    print(f"{result.final_message}")
    
    # Save results
    pipeline.save_results(result, "decoder_test_results")


if __name__ == "__main__":
    asyncio.run(main())
