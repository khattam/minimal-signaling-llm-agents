"""Organic multi-agent test - NO hardcoding of MSP understanding."""
import os
import asyncio

# Load from .env file
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from minimal_signaling.groq_client import GroqClient
from minimal_signaling.msp_encoder import MSPEncoder
from minimal_signaling.tokenization import TiktokenTokenizer

async def test_organic():
    print("=" * 70)
    print("ORGANIC MULTI-AGENT TEST")
    print("No hardcoding - agents figure it out themselves")
    print("=" * 70)
    
    groq = GroqClient()
    encoder = MSPEncoder(groq)
    tokenizer = TiktokenTokenizer()
    
    # =========================================================================
    # STEP 1: Agent A generates a natural message (no MSP knowledge)
    # =========================================================================
    print("\n🤖 AGENT A (Project Manager) - generating task request...")
    print("-" * 70)
    
    agent_a_response = await groq.chat(
        messages=[
            {"role": "system", "content": "You are a project manager. Write a detailed task request to your data analyst colleague asking them to analyze Q3 sales data and prepare a report for the board meeting next week."},
            {"role": "user", "content": "Write the task request."}
        ],
        temperature=0.7
    )
    
    original_tokens = tokenizer.count_tokens(agent_a_response)
    print(f"Agent A says ({original_tokens} tokens):")
    print(agent_a_response)
    
    # =========================================================================
    # STEP 2: Mediator encodes to MSP (the compression step)
    # =========================================================================
    print("\n\n📡 MEDIATOR - encoding to MSP signal...")
    print("-" * 70)
    
    signal = await encoder.encode(agent_a_response)
    signal_json = signal.model_dump_json(indent=2)
    signal_tokens = tokenizer.count_tokens(signal_json)
    
    print(f"MSP Signal ({signal_tokens} tokens):")
    print(signal_json)
    
    # =========================================================================
    # STEP 3: Agent B receives ONLY the JSON - no explanation!
    # =========================================================================
    print("\n\n🤖 AGENT B (Data Analyst) - receives raw JSON...")
    print("-" * 70)
    
    # Agent B has NO idea what MSP is - just receives JSON
    agent_b_response = await groq.chat(
        messages=[
            {"role": "system", "content": "You are a data analyst. Respond to incoming messages."},
            {"role": "user", "content": signal_json}  # Just the raw JSON, nothing else!
        ],
        temperature=0.3
    )
    
    response_tokens = tokenizer.count_tokens(agent_b_response)
    print(f"Agent B responds ({response_tokens} tokens):")
    print(agent_b_response)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"  Agent A original message:  {original_tokens} tokens")
    print(f"  MSP Signal transmitted:    {signal_tokens} tokens")
    print(f"  Compression:               {signal_tokens/original_tokens:.1%}")
    print(f"  Tokens saved:              {original_tokens - signal_tokens}")
    print()
    print("❓ KEY QUESTION: Did Agent B understand the task from just the JSON?")

if __name__ == "__main__":
    asyncio.run(test_organic())
