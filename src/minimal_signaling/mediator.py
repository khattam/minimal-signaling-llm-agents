"""Mediator orchestration for the minimal-signaling pipeline."""

import time
import uuid
from typing import Optional

from .interfaces import Compressor, SemanticKeyExtractor, Judge, Tokenizer
from .models import MediatorResult, PipelineError, CompressionResult, ExtractionResult
from .config import MediatorConfig
from .compression import CompressionEngine
from .events import (
    SyncEventEmitter,
    create_message_received_event,
    create_compression_start_event,
    create_compression_complete_event,
    create_extraction_start_event,
    create_extraction_complete_event,
    create_judge_start_event,
    create_judge_complete_event,
    create_pipeline_complete_event,
    create_pipeline_error_event,
)


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
        event_emitter: Optional[SyncEventEmitter] = None
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
        
        # Create compression engine with event emitter for pass-level events
        self.compression_engine = CompressionEngine(
            compressor=compressor,
            tokenizer=tokenizer,
            budget=config.compression.token_budget,
            max_passes=config.compression.max_recursion,
            event_emitter=event_emitter
        )
    
    def _emit(self, event_payload) -> None:
        """Emit an event if emitter is configured."""
        if self.event_emitter:
            self.event_emitter.emit(event_payload)
    
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
            
            # Emit message received event
            original_tokens = self.tokenizer.count_tokens(message)
            self._emit(create_message_received_event(message, original_tokens))
            
            # Stage 1: Compression (if enabled)
            if self.config.compression.enabled:
                self._emit(create_compression_start_event(
                    input_tokens=original_tokens,
                    budget=self.config.compression.token_budget
                ))
                
                try:
                    compression_result = self.compression_engine.compress_to_budget(
                        current_text
                    )
                    current_text = compression_result.compressed_text
                    
                    self._emit(create_compression_complete_event(
                        original_tokens=compression_result.original_tokens,
                        final_tokens=compression_result.final_tokens,
                        passes=compression_result.passes,
                        total_ratio=compression_result.total_ratio
                    ))
                except Exception as e:
                    self._emit(create_pipeline_error_event(
                        stage="compression",
                        error_type=type(e).__name__,
                        message=str(e)
                    ))
                    return self._create_error_result(
                        stage="compression",
                        error_type=type(e).__name__,
                        message=str(e),
                        start_time=start_time
                    )
            
            # Stage 2: Semantic Key Extraction (if enabled)
            if self.config.semantic_keys.enabled:
                self._emit(create_extraction_start_event())
                
                try:
                    extraction_result = self.extractor.extract(current_text)
                    
                    self._emit(create_extraction_complete_event(
                        key_count=len(extraction_result.keys),
                        schema_version=extraction_result.schema_version
                    ))
                except Exception as e:
                    self._emit(create_pipeline_error_event(
                        stage="extraction",
                        error_type=type(e).__name__,
                        message=str(e)
                    ))
                    return self._create_error_result(
                        stage="extraction",
                        error_type=type(e).__name__,
                        message=str(e),
                        start_time=start_time,
                        compression=compression_result
                    )
            
            # Optional Judge verification (if enabled)
            if self.config.judge.enabled and self.judge is not None:
                self._emit(create_judge_start_event())
                
                try:
                    if extraction_result is not None:
                        judge_result = self.judge.evaluate(
                            original=message,
                            keys=extraction_result.keys
                        )
                        
                        self._emit(create_judge_complete_event(
                            passed=judge_result.passed,
                            confidence=judge_result.confidence,
                            issue_count=len(judge_result.issues)
                        ))
                except Exception as e:
                    # Judge errors are non-fatal - log and continue
                    # (judge is optional verification)
                    pass
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Emit pipeline complete event
            self._emit(create_pipeline_complete_event(
                success=True,
                duration_ms=duration_ms
            ))
            
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
            self._emit(create_pipeline_error_event(
                stage="pipeline",
                error_type=type(e).__name__,
                message=str(e)
            ))
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
