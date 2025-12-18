"""Stage 1: Compression Engine with recursive compression logic."""

from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from .interfaces import Compressor, Tokenizer
from .models import CompressionResult, CompressionStep
from .events import SyncEventEmitter, create_compression_pass_event


class DistilBARTCompressor(Compressor):
    """DistilBART-based summarization compressor.
    
    Uses the sshleifer/distilbart-cnn-12-6 model for text compression
    via abstractive summarization.
    """
    
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        """Initialize the DistilBART compressor.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def compress(self, text: str) -> str:
        """Compress text using DistilBART summarization.
        
        Args:
            text: Input text to compress
            
        Returns:
            Compressed text
        """
        if not text.strip():
            return text
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode summary
        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )
        
        return summary


class CompressionEngine:
    """Orchestrates recursive compression to meet token budget.
    
    Implements the core compression loop with termination conditions:
    - Budget is met
    - Recursion limit is reached
    - Compression is no longer improving
    """
    
    def __init__(
        self,
        compressor: Compressor,
        tokenizer: Tokenizer,
        budget: int,
        max_passes: int,
        event_emitter: Optional[SyncEventEmitter] = None
    ):
        """Initialize the compression engine.
        
        Args:
            compressor: Compressor implementation to use
            tokenizer: Tokenizer for counting tokens
            budget: Maximum allowed tokens
            max_passes: Maximum compression passes
            event_emitter: Optional event emitter for real-time updates
        """
        self.compressor = compressor
        self.tokenizer = tokenizer
        self.budget = budget
        self.max_passes = max_passes
        self.event_emitter = event_emitter
    
    def _emit(self, payload) -> None:
        """Emit an event if emitter is configured."""
        if self.event_emitter:
            self.event_emitter.emit(payload)
    
    def compress_to_budget(self, text: str) -> CompressionResult:
        """Recursively compress text until budget is met or limit reached.
        
        Args:
            text: Input text to compress
            
        Returns:
            CompressionResult with compression metadata
        """
        original_tokens = self.tokenizer.count_tokens(text)
        current_text = text
        current_tokens = original_tokens
        passes = 0
        log: List[CompressionStep] = []
        
        # If already under budget, return immediately
        if current_tokens <= self.budget:
            return CompressionResult(
                compressed_text=text,
                original_tokens=original_tokens,
                final_tokens=current_tokens,
                passes=0,
                log=[]
            )
        
        # Recursive compression loop
        while passes < self.max_passes and current_tokens > self.budget:
            # Compress
            compressed = self.compressor.compress(current_text)
            new_tokens = self.tokenizer.count_tokens(compressed)
            
            # Check for improvement before recording
            if new_tokens >= current_tokens:
                # No improvement, stop without recording this step
                break
            
            # Record step
            step = CompressionStep(
                input_tokens=current_tokens,
                output_tokens=new_tokens,
                input_text=current_text,
                output_text=compressed
            )
            log.append(step)
            
            # Update for next iteration
            current_text = compressed
            current_tokens = new_tokens
            passes += 1
            
            # Emit pass event for real-time visualization
            self._emit(create_compression_pass_event(
                pass_number=passes,
                input_tokens=step.input_tokens,
                output_tokens=new_tokens,
                ratio=new_tokens / original_tokens if original_tokens > 0 else 1.0
            ))
        
        return CompressionResult(
            compressed_text=current_text,
            original_tokens=original_tokens,
            final_tokens=current_tokens,
            passes=passes,
            log=log
        )
