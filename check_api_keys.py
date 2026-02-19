"""Check which API keys are not rate limited."""

import asyncio
import os
from dotenv import load_dotenv
from src.minimal_signaling.groq_client import GroqClient

load_dotenv()

async def test_key(key_name: str, api_key: str):
    """Test a single API key."""
    print(f"\nTesting {key_name}...")
    try:
        client = GroqClient(api_key=api_key)
        response = await client.chat(
            messages=[
                {"role": "user", "content": "Hi"}
            ],
            temperature=0.0
        )
        print(f"✅ {key_name} is working!")
        print(f"   Response: {response[:50]}...")
        return True
    except Exception as e:
        print(f"❌ {key_name} failed: {str(e)[:100]}")
        return False

async def main():
    keys = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "GROQ_BACKUP_KEY": os.getenv("GROQ_BACKUP_KEY"),
        "GROQ_BACKUP_KEY_2": os.getenv("GROQ_BACKUP_KEY_2")
    }
    
    print("Checking all API keys...")
    
    working_keys = []
    for key_name, api_key in keys.items():
        if api_key:
            if await test_key(key_name, api_key):
                working_keys.append(key_name)
        else:
            print(f"⚠️  {key_name} not found in .env")
    
    print(f"\n{'='*60}")
    print(f"Working keys: {len(working_keys)}/{len(keys)}")
    if working_keys:
        print(f"Available: {', '.join(working_keys)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
