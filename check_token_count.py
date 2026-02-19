"""Just check token count of the long message."""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from minimal_signaling.tokenization import TiktokenTokenizer

# Import the message from test file
import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_long_message import LONG_MESSAGE

tokenizer = TiktokenTokenizer()
token_count = tokenizer.count_tokens(LONG_MESSAGE)

print(f"Message token count: {token_count} tokens")
print(f"Target: ~1500 tokens")
print(f"Difference: {token_count - 1500:+d} tokens")

if 1400 <= token_count <= 1600:
    print(f"\n✅ Message length is perfect for testing!")
else:
    print(f"\n⚠️  Message is {'too short' if token_count < 1400 else 'too long'}")
