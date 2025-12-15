"""Mediator orchestration for the minimal-signaling pipeline."""

import time
import uuid
from typing import Optional

from .interfaces import Compressor, SemanticKeyExtractor, Judge, Tokenizer, EventEmitter
from .models import MediatorResult, PipelineError, CompressionResult, ExtractionResult
from .config import MediatorConfig
from .compression import CompressionEngine


class Mediator:
    """Orchestrates the full minimal-signaling pipeline.
    
    The Mediator coordinates:
    1. Stage 1: Compression (if enabled)
    2. Stage 2: Semantic Key Extraction (if enabled)
    3. Optional Judge verification (if enabled)
    
    All stages can be toggled via configuration.
    """
    
    def __init__(
        self,
        config: MediatorConfig,
        compressor: Compressor,
        extractor: SemanticKeyExtractor,
        tokenizer: Tokenizer,
        judge: Optional[Judge] = None,
        event_emitter: Optional[EventEmitter] = None
    ):
        """Initialize the mediator.
        
        Args:
            config: Mediator configuration
            compressor: Compressor implementation
            extractor: Semantic key extractor implementation
            tokenizer: Tokenizer for token counting
            judge: Optional judge for verification
            event_emitter: Optional event emitter for real-time updates
        """
        self.config = config
        self.compressor = compressor
        self.extractor = extractor
        self.tokenizer = tokenizer
        self.judge = judge
        self.event_emitter = event_emitter
        
        # Create compression engine
        self.compression_engine = CompressionEngine(
            compressor=compressor,
            tokenizer=tokenizer,
            budget=config.compression.token_budget,
            max_passes=config.compression.max_recursion,
            event_emitter=event_emitter
        )
    
    def process(self, message: str) -> MediatorResult:
        """Process a message through the full pipeline.
        
        Args:
            message: Input message from Agent A
            
        Returns:
            MediatorResult with outputs from all stages
        """
        start_time = time.time()
        
        try:
            current_text = message
            compression_result: Optional[CompressionResult] = None
            extraction_result: Optional[ExtractionResult] = None
            judge_result = None
            
            # Stage 1: Compression (if enabled)
            if self.config.compression.enabled:
                try:
                    compression_result = self.compression_engine.compress_to_budget(
                        current_text
                    )
                    current_text = compression_result.compressed_text
                except Exception as e:
                    return self._create_error_result(
                        stage="compression",
                        error_type=type(e).__name__,
                        message=str(e),
                        start_time=start_time
                    )
            
            # Stage 2: Semantic Key Extraction (if enabled)
            if self.config.semantic_keys.enabled:
                try:
                    extraction_result = self.extractor.extract(current_text)
                except Exception as e:
                    return self._create_error_result(
                        stage="extraction",
                        error_type=type(e).__name__,
                        message=str(e),
                        start_time=start_time,
                        compression=compression_result
                    )
            
            # Optional Judge verification (if enabled)
            if self.config.judge.enabled and self.judge is not None:
                try:
                    if extraction_result is not None:
                        judge_result = self.judge.evaluate(
                            original=message,
                            keys=extraction_result.keys
                        )
                except Exception as e:
                    # Judge errors are non-fatal - log and continue
                    # (judge is optional verification)
                    pass
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            return MediatorResult(
                success=True,
                compression=compression_result,
                extraction=extraction_result,
                judge=judge_result,
                error=None,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            # Catch-all for unexpected errors
            return self._create_error_result(
                stage="pipeline",
                error_type=type(e).__name__,
                message=str(e),
                start_time=start_time
            )
    
    def _create_error_result(
        self,
        stage: str,
        error_type: str,
        message: str,
        start_time: float,
        compression: Optional[CompressionResult] = None,
        extraction: Optional[ExtractionResult] = None
    ) -> MediatorResult:
        """Create an error result.
        
        Args:
            stage: Stage where error occurred
            error_type: Type of error
            message: Error message
            start_time: Pipeline start time
            compression: Optional compression result if completed
            extraction: Optional extraction result if completed
            
        Returns:
            MediatorResult with error information
        """
        duration_ms = (time.time() - start_time) * 1000
        
        return MediatorResult(
            success=False,
            compression=compression,
            extraction=extraction,
            judge=None,
            error=PipelineError(
                stage=stage,
                error_type=error_type,
                message=message,
                recoverable=False
            ),
            duration_ms=duration_ms
        )
