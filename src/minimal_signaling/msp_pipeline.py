"""MSP Pipeline - orchestrates encode → signal → decode → verify flow."""

import time
from typing import Optional

from .groq_client import GroqClient
from .msp_encoder import MSPEncoder
from .msp_decoder import MSPDecoder
from .semantic_judge import SemanticJudge
from .tokenization import TiktokenTokenizer
from .protocol import PipelineResult, PipelineMetrics, MinimalSignal
from .msp_config import MSPConfig
from .events import SyncEventEmitter, EventPayload, PipelineEvent


class MSPPipeline:
    """Orchestrates the full MSP pipeline."""
    
    def __init__(
        self,
        config: Optional[MSPConfig] = None,
        event_emitter: Optional[SyncEventEmitter] = None
    ):
        """Initialize MSP pipeline.
        
        Args:
            config: MSP configuration. If None, loads from env.
            event_emitter: Optional event emitter for real-time updates.
        """
        self.config = config or MSPConfig.from_env()
        self.event_emitter = event_emitter
        
        # Initialize components
        self.groq_client = GroqClient(
            api_key=self.config.groq_api_key,
            model=self.config.groq_model,
            requests_per_minute=self.config.rate_limit_rpm
        )
        
        self.encoder = MSPEncoder(self.groq_client)
        self.decoder = MSPDecoder(self.groq_client)
        self.judge = SemanticJudge(
            model_name=self.config.judge_model,
            threshold=self.config.judge_threshold
        )
        self.tokenizer = TiktokenTokenizer()
    
    def _emit(self, event: PipelineEvent, data: dict) -> None:
        """Emit an event if emitter is configured."""
        if self.event_emitter:
            from datetime import datetime
            payload = EventPayload(event=event, data=data, timestamp=datetime.utcnow())
            self.event_emitter.emit(payload)
    
    async def process(
        self,
        input_text: str,
        style: str = "professional"
    ) -> PipelineResult:
        """Process text through the full MSP pipeline.
        
        Args:
            input_text: Natural language input.
            style: Output style for decoder.
            
        Returns:
            PipelineResult with all outputs and metrics.
        """
        start_time = time.time()
        
        # Emit start event
        original_tokens = self.tokenizer.count_tokens(input_text)
        self._emit(PipelineEvent.MESSAGE_RECEIVED, {
            "message": input_text[:100],
            "token_count": original_tokens
        })
        
        # Encode NL → MSP
        self._emit(PipelineEvent.EXTRACTION_START, {})
        signal = await self.encoder.encode(input_text)
        signal_tokens = self.tokenizer.count_tokens(signal.model_dump_json())
        self._emit(PipelineEvent.EXTRACTION_COMPLETE, {
            "key_count": len(signal.params) + len(signal.constraints),
            "schema_version": signal.version
        })
        
        # Decode MSP → NL
        decoded = await self.decoder.decode(signal, style)
        decoded_tokens = self.tokenizer.count_tokens(decoded)
        
        # Judge semantic fidelity
        self._emit(PipelineEvent.JUDGE_START, {})
        judge_result = self.judge.evaluate(input_text, decoded)
        self._emit(PipelineEvent.JUDGE_COMPLETE, {
            "passed": judge_result.passed,
            "confidence": judge_result.confidence,
            "issue_count": len(judge_result.issues)
        })
        
        # Compute metrics
        latency_ms = (time.time() - start_time) * 1000
        compression_ratio = signal_tokens / original_tokens if original_tokens > 0 else 1.0
        
        metrics = PipelineMetrics(
            original_tokens=original_tokens,
            signal_tokens=signal_tokens,
            decoded_tokens=decoded_tokens,
            compression_ratio=compression_ratio,
            latency_ms=latency_ms
        )
        
        # Emit complete event
        self._emit(PipelineEvent.PIPELINE_COMPLETE, {
            "success": True,
            "duration_ms": latency_ms
        })
        
        return PipelineResult(
            original_text=input_text,
            signal=signal,
            decoded_text=decoded,
            judge=judge_result,
            metrics=metrics,
            trace_id=signal.trace_id,
            timestamp=signal.timestamp
        )
    
    def process_sync(
        self,
        input_text: str,
        style: str = "professional"
    ) -> PipelineResult:
        """Synchronous version of process."""
        start_time = time.time()
        
        original_tokens = self.tokenizer.count_tokens(input_text)
        
        # Encode
        signal = self.encoder.encode_sync(input_text)
        signal_tokens = self.tokenizer.count_tokens(signal.model_dump_json())
        
        # Decode
        decoded = self.decoder.decode_sync(signal, style)
        decoded_tokens = self.tokenizer.count_tokens(decoded)
        
        # Judge
        judge_result = self.judge.evaluate(input_text, decoded)
        
        # Metrics
        latency_ms = (time.time() - start_time) * 1000
        compression_ratio = signal_tokens / original_tokens if original_tokens > 0 else 1.0
        
        metrics = PipelineMetrics(
            original_tokens=original_tokens,
            signal_tokens=signal_tokens,
            decoded_tokens=decoded_tokens,
            compression_ratio=compression_ratio,
            latency_ms=latency_ms
        )
        
        return PipelineResult(
            original_text=input_text,
            signal=signal,
            decoded_text=decoded,
            judge=judge_result,
            metrics=metrics,
            trace_id=signal.trace_id,
            timestamp=signal.timestamp
        )
