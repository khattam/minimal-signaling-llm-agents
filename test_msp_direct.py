"""Test MSP signal being used directly by another LLM."""
import os
import asyncio

# Load from .env file
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from minimal_signaling.msp_pipeline import MSPPipeline
from minimal_signaling.msp_config import MSPConfig
from minimal_signaling.groq_client import GroqClient
from minimal_signaling.tokenization import TiktokenTokenizer

async def test_direct_signal():
    print("=" * 70)
    print("MSP Direct Signal Test - Signal as LLM Input")
    print("=" * 70)
    
    tokenizer = TiktokenTokenizer()
    
    # Verbose original message (what Agent A would normally send)
    original = """Hello there! I hope this message finds you well. I wanted to reach out 
to you today because I have an important request that I believe requires your immediate 
attention and expertise. 

As you may already be aware, our company has been collecting quarterly sales data from 
all of our regional offices across the country, including the Northeast, Southeast, 
Midwest, Southwest, and West Coast divisions. 

What I need from you is a comprehensive analysis of this data. Specifically, I am 
looking for you to identify any significant trends that have emerged over the past year, 
as well as any anomalies or outliers that might warrant further investigation. This 
analysis is critically important because we will be presenting the findings at our 
upcoming Q4 board meeting, where senior leadership will be making key strategic 
decisions based on your insights.

Please note that there is a firm deadline for this work - the report absolutely must be 
completed and ready for review by this Friday at the latest. Given the importance of 
this deliverable and the tight timeline, I am marking this as a high priority task.

Thank you so much for your help with this matter."""

    original_tokens = tokenizer.count_tokens(original)
    
    print(f"\n📝 ORIGINAL MESSAGE (what Agent A would send):")
    print(f"   Tokens: {original_tokens}")
    print("-" * 70)
    print(original[:200] + "...")
    print()
    
    # Encode to MSP
    config = MSPConfig.from_env()
    pipeline = MSPPipeline(config=config)
    
    print("🔄 Encoding to MSP signal...")
    signal = await pipeline.encoder.encode(original)
    
    # The MSP signal as JSON - THIS is what gets sent between agents
    signal_json = signal.model_dump_json(indent=2)
    signal_tokens = tokenizer.count_tokens(signal_json)
    
    print(f"\n📦 MSP SIGNAL (what actually gets transmitted):")
    print(f"   Tokens: {signal_tokens}")
    print("-" * 70)
    print(signal_json)
    print()
    
    # Now simulate Agent B receiving the signal and acting on it
    print("🤖 AGENT B receives MSP signal and executes task...")
    print("-" * 70)
    
    groq = GroqClient()
    
    # Agent B's system prompt - it understands MSP format
    agent_b_prompt = """You are Agent B, a data analyst. You receive task instructions 
in MSP (Minimal Signal Protocol) JSON format. Execute the task based on the signal.

The signal contains:
- intent: The action to perform
- target: What to act on  
- params: Key parameters
- constraints: Requirements/deadlines
- priority: Urgency level

Respond with a brief acknowledgment and your plan to execute."""

    # Agent B processes the MSP signal directly (not decoded NL!)
    response = await groq.chat(
        messages=[
            {"role": "system", "content": agent_b_prompt},
            {"role": "user", "content": f"Execute this task:\n{signal_json}"}
        ],
        temperature=0.3
    )
    
    response_tokens = tokenizer.count_tokens(response)
    
    print(f"\n📤 AGENT B RESPONSE:")
    print(response)
    print()
    
    # Summary
    print("=" * 70)
    print("📊 COMPRESSION SUMMARY")
    print("=" * 70)
    print(f"  Original NL message:  {original_tokens} tokens")
    print(f"  MSP Signal:           {signal_tokens} tokens")
    print(f"  Compression ratio:    {signal_tokens/original_tokens:.1%}")
    print(f"  Tokens SAVED:         {original_tokens - signal_tokens} tokens")
    print()
    print("💡 KEY INSIGHT:")
    print("   The MSP signal IS the message between agents.")
    print("   No decode needed - Agent B understands structured JSON directly!")
    print("   Human traceability: You can read the JSON to see exactly what was sent.")

if __name__ == "__main__":
    asyncio.run(test_direct_signal())
