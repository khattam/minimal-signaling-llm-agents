"""Test with detailed JSON logging of all iterations."""
import os
import asyncio
import json
from datetime import datetime

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from minimal_signaling.groq_client import GroqClient
from minimal_signaling.iterative_encoder import IterativeEncoder
from minimal_signaling.semantic_judge import SemanticJudge
from minimal_signaling.msp_decoder import MSPDecoder
from minimal_signaling.tokenization import TiktokenTokenizer

async def test_with_logging():
    # Load the long message
    with open("long_test_message.txt", "r") as f:
        message = f.read()
    
    tokenizer = TiktokenTokenizer()
    groq = GroqClient()
    judge = SemanticJudge(threshold=0.80)
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
    print("TESTING WITH DETAILED LOGGING")
    print("=" * 80)
    print(f"\n📝 Original: {original_tokens} tokens")
    
    print("\n🔄 Running iterative encoding...\n")
    
    result = await encoder.encode_with_refinement(message)
    
    # Build detailed log
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "original_message": message,
        "original_tokens": original_tokens,
        "target_similarity": 0.80,
        "max_iterations": 3,
        "converged": result.converged,
        "final_similarity": result.final_similarity,
        "iterations": []
    }
    
    for i, step in enumerate(result.refinement_history, 1):
        decoded_tokens = tokenizer.count_tokens(step.decoded_text)
        signal_json = json.loads(step.signal.model_dump_json())
        
        iteration_data = {
            "iteration": i,
            "signal": signal_json,
            "signal_tokens": step.signal_tokens,
            "decoded_text": step.decoded_text,
            "decoded_tokens": decoded_tokens,
            "similarity_score": step.similarity_score,
            "compression_ratio": decoded_tokens / original_tokens,
            "feedback": step.feedback
        }
        
        log_data["iterations"].append(iteration_data)
        
        print(f"Iteration {i}:")
        print(f"  Similarity: {step.similarity_score:.1%}")
        print(f"  Decoded tokens: {decoded_tokens}")
        print(f"  Compression: {decoded_tokens/original_tokens:.1%}")
    
    # Save to JSON file
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"test_run_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Converged: {'✅ YES' if result.converged else '❌ NO'}")
    print(f"Final similarity: {result.final_similarity:.1%}")
    print(f"Iterations: {result.iterations}")
    print(f"\n📁 Detailed log saved to: {filename}")

if __name__ == "__main__":
    asyncio.run(test_with_logging())
