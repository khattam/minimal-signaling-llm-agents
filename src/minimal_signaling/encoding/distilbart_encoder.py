"""DistilBART-based encoder for reliable text compression."""

from typing import Optional
from transformers import pipeline
from ..tokenization import TiktokenTokenizer


class DistilBARTEncoder:
    """Encoder using DistilBART for text summarization/compression."""
    
    def __init__(self):
        """Initialize DistilBART summarization pipeline."""
        # Use DistilBART for fast, reliable summarization
        self.summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1  # CPU
        )
        self.tokenizer = TiktokenTokenizer()
    
    def encode(self, text: str, target_ratio: float = 0.5) -> str:
        """Compress text using DistilBART.
        
        Args:
            text: Input text to compress
            target_ratio: Target compression ratio (0.5 = 50% of original length)
            
        Returns:
            Compressed text
        """
        # Calculate target length
        input_tokens = self.tokenizer.count_tokens(text)
        target_tokens = int(input_tokens * target_ratio)
        
        # DistilBART works best with max_length between 50-1024
        max_length = min(max(target_tokens, 50), 1024)
        min_length = max(int(max_length * 0.5), 30)
        
        # Compress using DistilBART
        result = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        
        return result[0]['summary_text']
