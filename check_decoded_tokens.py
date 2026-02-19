from minimal_signaling.tokenization import TiktokenTokenizer
import json

t = TiktokenTokenizer()
data = json.load(open('data/run_20260219_092223.json'))

original_tokens = data["original_tokens"]
decoded_tokens = t.count_tokens(data["texts"]["final_decoded"])
compression_pct = (1 - decoded_tokens / original_tokens) * 100

print(f'Original: {original_tokens} tokens')
print(f'Decoded: {decoded_tokens} tokens')
print(f'Compression: {compression_pct:.1f}%')
print(f'Signal: {data["final_tokens"]} tokens')
