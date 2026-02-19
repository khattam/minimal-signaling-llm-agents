"""Inspect what the signal actually contains for the long message."""
import os
import asyncio
import json

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from minimal_signaling.groq_client import GroqClient
from minimal_signaling.msp_encoder import MSPEncoder
from minimal_signaling.tokenization import TiktokenTokenizer

async def inspect():
    with open("long_test_message.txt", "r") as f:
        message = f.read()
    
    groq = GroqClient()
    encoder = MSPEncoder(groq)
    tokenizer = TiktokenTokenizer()
    
    print("Encoding the 1210 token message...")
    signal = await encoder.encode(message)
    
    signal_json = json.loads(signal.model_dump_json())
    
    print("\n" + "=" * 80)
    print("SIGNAL CONTENTS")
    print("=" * 80)
    
    print(f"\nIntent: {signal_json['intent']}")
    print(f"Target: {signal_json['target']}")
    print(f"Priority: {signal_json['priority']}")
    
    print(f"\nParams ({len(signal_json['params'])} keys):")
    for key, value in signal_json['params'].items():
        if isinstance(value, list):
            print(f"  {key}: [{len(value)} items]")
            for item in value[:3]:  # Show first 3
                print(f"    - {str(item)[:80]}")
            if len(value) > 3:
                print(f"    ... and {len(value)-3} more")
        elif isinstance(value, dict):
            print(f"  {key}: {{{len(value)} keys}}")
        else:
            print(f"  {key}: {str(value)[:80]}")
    
    print(f"\nConstraints ({len(signal_json['constraints'])} items):")
    for c in signal_json['constraints'][:5]:
        print(f"  - {c}")
    if len(signal_json['constraints']) > 5:
        print(f"  ... and {len(signal_json['constraints'])-5} more")
    
    print(f"\nState: {signal_json['state']}")
    
    signal_tokens = tokenizer.count_tokens(signal.model_dump_json())
    print(f"\nSignal size: {signal_tokens} tokens")

if __name__ == "__main__":
    asyncio.run(inspect())
