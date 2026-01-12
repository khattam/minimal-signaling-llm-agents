"""Test iterative encoding with semantic feedback loop."""
import os
import asyncio

# Load from .env file
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from minimal_signaling.groq_client import GroqClient
from minimal_signaling.iterative_encoder import IterativeEncoder
from minimal_signaling.semantic_judge import SemanticJudge
from minimal_signaling.msp_decoder import MSPDecoder

async def test_iterative():
    print("=" * 70)
    print("ITERATIVE ENCODING WITH SEMANTIC FEEDBACK LOOP")
    print("Novel: Closed-loop optimization for semantic preservation")
    print("=" * 70)
    
    # Complex message that might lose info in one-shot encoding
    message = """I have completed my initial assessment of the codebase and need to 
communicate findings to the code review agent. 

After analyzing the repository, I found several critical issues:

1. SECURITY: The authentication module has a session hijacking vulnerability - 
   tokens aren't invalidated on refresh. Also, SQL injection risk in user 
   management due to string concatenation instead of parameterized queries.

2. PERFORMANCE: Product catalog makes 47 redundant API calls per request cycle.
   Same data fetched multiple times when it could be cached at request level.

3. ERROR HANDLING: Generic exceptions caught and silently logged throughout.
   Error info not propagated to callers, making debugging nearly impossible.

I need you to:
- Review authentication and user management modules specifically
- Verify my security vulnerability assessment  
- Recommend remediation approaches
- Provide caching strategy for the performance issues

CRITICAL: Security audit is next week. SQL injection and session handling 
fixes must be in place before then. This is highest priority.

I'll continue analyzing remaining modules and send additional findings."""

    print(f"\n📝 Original message length: {len(message)} chars")
    print("-" * 70)
    
    # Initialize components
    groq = GroqClient()
    judge = SemanticJudge(threshold=0.85)  # Higher threshold to trigger refinement
    decoder = MSPDecoder(groq)
    
    encoder = IterativeEncoder(
        groq_client=groq,
        judge=judge,
        decoder=decoder,
        max_iterations=3,
        target_similarity=0.85
    )
    
    print("\n🔄 Starting iterative encoding...")
    print("   (Will refine until similarity >= 85% or max 3 iterations)\n")
    
    result = await encoder.encode_with_refinement(message)
    
    # Show each iteration
    for step in result.refinement_history:
        print(f"{'='*70}")
        print(f"ITERATION {step.iteration}")
        print(f"{'='*70}")
        print(f"Signal tokens: {step.signal_tokens}")
        print(f"Similarity: {step.similarity_score:.1%}")
        
        if step.feedback:
            print(f"\n📋 Feedback for next iteration:")
            print(f"   {step.feedback[:200]}...")
        
        print(f"\n📦 Signal intent: {step.signal.intent}")
        print(f"   Target: {step.signal.target}")
        print(f"   Params keys: {list(step.signal.params.keys())}")
        print(f"   Constraints: {len(step.signal.constraints)} items")
        print()
    
    # Final summary
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Converged: {'✅ YES' if result.converged else '❌ NO'}")
    print(f"Iterations: {result.iterations}")
    print(f"Original tokens: {result.original_tokens}")
    print(f"Final signal tokens: {result.signal_tokens}")
    print(f"Compression: {result.signal_tokens/result.original_tokens:.1%}")
    print(f"Final similarity: {result.final_similarity:.1%}")
    print()
    
    # Show improvement across iterations
    if len(result.refinement_history) > 1:
        print("📈 SIMILARITY IMPROVEMENT:")
        for step in result.refinement_history:
            bar = "█" * int(step.similarity_score * 50)
            print(f"   Iter {step.iteration}: {bar} {step.similarity_score:.1%}")

if __name__ == "__main__":
    asyncio.run(test_iterative())
