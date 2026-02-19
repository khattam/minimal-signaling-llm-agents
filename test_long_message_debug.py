"""Debug test for long messages - see what's actually in the signal."""
import os
import asyncio
import json

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from minimal_signaling.groq_client import GroqClient
from minimal_signaling.msp_encoder import MSPEncoder
from minimal_signaling.msp_decoder import MSPDecoder
from minimal_signaling.semantic_judge import SemanticJudge
from minimal_signaling.tokenization import TiktokenTokenizer

async def debug_long_message():
    # Long complex message
    long_message = """I have completed my initial assessment of the codebase and need to 
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

    groq = GroqClient()
    encoder = MSPEncoder(groq)
    decoder = MSPDecoder(groq)
    judge = SemanticJudge(threshold=0.85)
    tokenizer = TiktokenTokenizer()
    
    print("=" * 80)
    print("LONG MESSAGE DEBUG")
    print("=" * 80)
    
    original_tokens = tokenizer.count_tokens(long_message)
    print(f"\n📝 Original message: {original_tokens} tokens")
    print(f"   Length: {len(long_message)} chars")
    
    # Encode
    print("\n🔄 Encoding...")
    signal = await encoder.encode(long_message)
    
    # Show the actual signal
    signal_json = signal.model_dump_json(indent=2)
    signal_tokens = tokenizer.count_tokens(signal_json)
    
    print(f"\n📦 MSP SIGNAL ({signal_tokens} tokens):")
    print("=" * 80)
    print(signal_json)
    print("=" * 80)
    
    # Decode
    print("\n🔄 Decoding...")
    decoded = await decoder.decode(signal, "professional")
    decoded_tokens = tokenizer.count_tokens(decoded)
    
    print(f"\n📤 DECODED ({decoded_tokens} tokens):")
    print("=" * 80)
    print(decoded)
    print("=" * 80)
    
    # Judge
    print("\n⚖️ Judging...")
    result = judge.evaluate(long_message, decoded)
    
    print(f"\n📊 RESULTS:")
    print(f"   Similarity: {result.similarity_score:.1%}")
    print(f"   Passed: {result.passed}")
    print(f"   Compression: {signal_tokens}/{original_tokens} = {signal_tokens/original_tokens:.1%}")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    print(f"   Signal has {len(signal.params)} param keys")
    print(f"   Signal has {len(signal.constraints)} constraints")
    print(f"   Signal has {len(signal.details)} details")
    
    # Check what's in params
    print(f"\n   Params structure:")
    for key, value in signal.params.items():
        if isinstance(value, list):
            print(f"      {key}: list with {len(value)} items")
        elif isinstance(value, dict):
            print(f"      {key}: dict with {len(value)} keys")
        else:
            print(f"      {key}: {type(value).__name__}")

if __name__ == "__main__":
    asyncio.run(debug_long_message())
