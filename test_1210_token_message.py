"""Test the 1210 token message with current system."""
import os
import asyncio

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from minimal_signaling.groq_client import GroqClient
from minimal_signaling.iterative_encoder import IterativeEncoder
from minimal_signaling.semantic_judge import SemanticJudge
from minimal_signaling.msp_decoder import MSPDecoder
from minimal_signaling.tokenization import TiktokenTokenizer

async def test_long():
    # Load the long message
    with open("long_test_message.txt", "r") as f:
        message = f.read()
    
    tokenizer = TiktokenTokenizer()
    groq = GroqClient()
    judge = SemanticJudge(threshold=0.80)  # 80% target
    decoder = MSPDecoder(groq)
    
    encoder = IterativeEncoder(
        groq_client=groq,
        judge=judge,
        decoder=decoder,
        max_iterations=3,
        target_similarity=0.80
    )
    
    original_tokens = tokenizer.count_tokens(message)
    
    print("=" * 80)
    print("TESTING 1210 TOKEN MESSAGE")
    print("=" * 80)
    print(f"\n📝 Original: {original_tokens} tokens")
    print(f"   Target similarity: 80%")
    print(f"   Max iterations: 3")
    
    print("\n🔄 Running iterative encoding...\n")
    
    result = await encoder.encode_with_refinement(message)
    
    # Show results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for i, step in enumerate(result.refinement_history, 1):
        decoded_tokens = tokenizer.count_tokens(step.decoded_text)
        compression = decoded_tokens / original_tokens
        
        print(f"\nIteration {i}:")
        print(f"  Similarity: {step.similarity_score:.1%}")
        print(f"  Decoded tokens: {decoded_tokens}")
        print(f"  Compression: {compression:.1%} ({100-compression*100:.1f}% reduction)")
        
        if step.feedback:
            print(f"  Feedback: {step.feedback[:100]}...")
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    final_decoded_tokens = tokenizer.count_tokens(result.final_decoded)
    final_compression = final_decoded_tokens / original_tokens
    
    print(f"Converged: {'✅ YES' if result.converged else '❌ NO'}")
    print(f"Final similarity: {result.final_similarity:.1%}")
    print(f"Original tokens: {original_tokens}")
    print(f"Final decoded tokens: {final_decoded_tokens}")
    print(f"Compression: {final_compression:.1%} ({100-final_compression*100:.1f}% reduction)")
    print(f"Iterations used: {result.iterations}")

if __name__ == "__main__":
    asyncio.run(test_long())
