"""Iterative MSP Encoder with Semantic Feedback Loop.

This is the novel contribution: instead of one-shot encoding, we use
a closed-loop system where the semantic judge provides feedback on
what information was lost, and the encoder retries with that guidance.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any
from enum import Enum

from .groq_client import GroqClient
from .protocol import MinimalSignal, ContentSection, EncoderError, VALID_INTENTS, VALID_PRIORITIES
from .semantic_judge import SemanticJudge
from .msp_decoder import MSPDecoder
from .msp_encoder import MSPEncoder
from .tokenization import TiktokenTokenizer


class PipelineStage(str, Enum):
    """Pipeline stages for real-time tracking."""
    IDLE = "idle"
    ENCODING = "encoding"
    DECODING = "decoding"
    JUDGING = "judging"
    ANALYZING = "analyzing"
    REFINING = "refining"
    AGENT_B = "agent_b"
    COMPLETE = "complete"


@dataclass
class StageEvent:
    """Event emitted when pipeline stage changes."""
    stage: PipelineStage
    iteration: int
    similarity: Optional[float] = None
    passed_threshold: Optional[bool] = None
    feedback: Optional[str] = None
    data: dict = field(default_factory=dict)


@dataclass
class RefinementStep:
    """Record of one refinement iteration."""
    iteration: int
    signal: MinimalSignal
    decoded_text: str
    similarity_score: float
    feedback: Optional[str]
    signal_tokens: int = 0  # Add this back
    signal_tokens: int


@dataclass 
class IterativeEncodingResult:
    """Result of iterative encoding process."""
    original_text: str
    original_tokens: int
    final_signal: MinimalSignal
    final_decoded: str
    final_similarity: float
    iterations: int
    refinement_history: List[RefinementStep]
    converged: bool
    signal_tokens: int


ENCODER_BASE_PROMPT = """You are a semantic encoder. Extract structured information from the input message.

Output a JSON object with these fields:
- intent: One of [ANALYZE, GENERATE, EVALUATE, TRANSFORM, QUERY, RESPOND, DELEGATE, REPORT]
- target: What the action is about (string, be concise)
- params: Key parameters as key-value pairs (object, extract important details)
- constraints: List of constraints/requirements (array of strings)
- state: Current state information (object)
- priority: One of [low, medium, high, critical]

Rules:
- Be concise. Extract only essential information.
- If no constraints mentioned, use empty array.
- If no state mentioned, use empty object.
- Default priority to "medium" if not specified.

Output ONLY valid JSON, no explanation."""


ENCODER_REFINEMENT_PROMPT = """You are a semantic encoder. Extract structured information from the input message.

IMPORTANT: A previous encoding attempt lost some information. The semantic judge detected these issues:
{feedback}

Please re-encode the message, making sure to capture the missing information identified above.

Output a JSON object with these fields:
- intent: One of [ANALYZE, GENERATE, EVALUATE, TRANSFORM, QUERY, RESPOND, DELEGATE, REPORT]
- target: What the action is about (string, be concise)
- params: Key parameters as key-value pairs (object, extract important details)
- constraints: List of constraints/requirements (array of strings)
- state: Current state information (object)
- priority: One of [low, medium, high, critical]

Focus especially on: {focus_areas}

Output ONLY valid JSON, no explanation."""


FEEDBACK_ANALYSIS_PROMPT = """Compare the original message with the decoded version and identify what information was LOST or DISTORTED.

Original message:
{original}

Decoded version:
{decoded}

List the specific pieces of information that are:
1. Missing entirely from the decoded version
2. Significantly changed or distorted
3. Key details that were oversimplified

Be specific and concise. Output a brief list of what needs to be preserved better."""


class IterativeEncoder:
    """Encoder with semantic feedback loop for iterative refinement.
    
    Novel contribution: Uses judge feedback to guide re-encoding,
    creating a closed-loop optimization for semantic preservation.
    """
    
    def __init__(
        self,
        groq_client: GroqClient,
        judge: SemanticJudge,
        decoder: MSPDecoder,
        max_iterations: int = 3,
        target_similarity: float = 0.85,
        on_stage_change: Optional[Callable[[StageEvent], Any]] = None
    ):
        self.client = groq_client
        self.encoder = MSPEncoder(groq_client)  # Use MSPEncoder
        self.judge = judge
        self.decoder = decoder
        self.tokenizer = TiktokenTokenizer()
        self.max_iterations = max_iterations
        self.target_similarity = target_similarity
        self.on_stage_change = on_stage_change
    
    def _emit(self, stage: PipelineStage, iteration: int, **kwargs):
        """Emit a stage change event."""
        if self.on_stage_change:
            event = StageEvent(stage=stage, iteration=iteration, **kwargs)
            self.on_stage_change(event)
    
    async def encode_with_refinement(
        self,
        natural_language: str,
        style: str = "professional"
    ) -> IterativeEncodingResult:
        """Encode with iterative refinement until semantic threshold is met.
        
        Args:
            natural_language: Input text to encode.
            style: Decoding style for verification.
            
        Returns:
            IterativeEncodingResult with full refinement history.
        """
        if not natural_language or not natural_language.strip():
            raise EncoderError("Input cannot be empty")
        
        original_tokens = self.tokenizer.count_tokens(natural_language)
        refinement_history: List[RefinementStep] = []
        
        feedback = None
        focus_areas = None
        
        for iteration in range(self.max_iterations):
            # Emit encoding stage
            if iteration == 0:
                self._emit(PipelineStage.ENCODING, iteration + 1)
                signal = await self._encode_initial(natural_language)
            else:
                self._emit(PipelineStage.REFINING, iteration + 1, feedback=feedback)
                signal = await self._encode_with_feedback(
                    natural_language, feedback, focus_areas
                )
            
            signal_tokens = self.tokenizer.count_tokens(signal.model_dump_json())
            
            # Emit decoding stage
            self._emit(PipelineStage.DECODING, iteration + 1)
            decoded = await self.decoder.decode(signal, style)
            
            # Emit judging stage
            self._emit(PipelineStage.JUDGING, iteration + 1)
            judge_result = self.judge.evaluate(natural_language, decoded)
            
            passed = judge_result.similarity_score >= self.target_similarity
            
            # Generate feedback for next iteration (if needed)
            if not passed and iteration < self.max_iterations - 1:
                self._emit(
                    PipelineStage.ANALYZING, 
                    iteration + 1,
                    similarity=judge_result.similarity_score,
                    passed_threshold=False
                )
                feedback = await self._analyze_loss(natural_language, decoded)
                focus_areas = self._extract_focus_areas(feedback)
            else:
                feedback = None
            
            # Record this iteration
            step = RefinementStep(
                iteration=iteration + 1,
                signal=signal,
                decoded_text=decoded,
                similarity_score=judge_result.similarity_score,
                feedback=feedback,
                signal_tokens=signal_tokens
            )
            refinement_history.append(step)
            
            # Check convergence
            if passed:
                return IterativeEncodingResult(
                    original_text=natural_language,
                    original_tokens=original_tokens,
                    final_signal=signal,
                    final_decoded=decoded,
                    final_similarity=judge_result.similarity_score,
                    iterations=iteration + 1,
                    refinement_history=refinement_history,
                    converged=True,
                    signal_tokens=signal_tokens
                )
        
        # Max iterations reached without convergence
        final_step = refinement_history[-1]
        return IterativeEncodingResult(
            original_text=natural_language,
            original_tokens=original_tokens,
            final_signal=final_step.signal,
            final_decoded=final_step.decoded_text,
            final_similarity=final_step.similarity_score,
            iterations=self.max_iterations,
            refinement_history=refinement_history,
            converged=False,
            signal_tokens=final_step.signal_tokens
        )
    
    async def _encode_initial(self, text: str) -> MinimalSignal:
        """First encoding attempt without feedback."""
        return await self.encoder.encode(text)
    
    async def _encode_with_feedback(
        self, 
        text: str, 
        feedback: str,
        focus_areas: str
    ) -> MinimalSignal:
        """Re-encode with feedback from previous iteration."""
        # Determine strategy based on length
        token_count = self.tokenizer.count_tokens(text)
        
        if token_count < 500:
            strategy_note = "Use compact encoding with summary only."
        elif token_count < 1500:
            strategy_note = "Use detailed encoding with summary and sections."
        else:
            strategy_note = "Use chunked encoding with multiple detailed sections."
        
        refinement_prompt = f"""You are a semantic encoder. A previous encoding lost information.

FEEDBACK FROM JUDGE:
{feedback}

FOCUS ON: {focus_areas}

{strategy_note}

Output JSON with:
- intent: One of [ANALYZE, GENERATE, EVALUATE, TRANSFORM, QUERY, RESPOND, DELEGATE, REPORT]
- target: What the action is about
- summary: High-level structured overview
- sections: Array of detailed sections (if message is complex):
  - title: Section name
  - content: FULL details for this section
  - importance: critical|high|medium|low
- constraints: All constraints
- state: Current state
- priority: One of [low, medium, high, critical]

CRITICAL: Address the missing information identified in feedback.
In sections, preserve ALL specific details mentioned in the original message.

Output ONLY valid JSON."""
        
        response = await self.client.chat(
            messages=[
                {"role": "system", "content": refinement_prompt},
                {"role": "user", "content": text}
            ],
            json_mode=True,
            temperature=0.1
        )
        return self._parse_signal(response)
    
    async def _analyze_loss(self, original: str, decoded: str) -> str:
        """Use LLM to analyze what information was lost."""
        prompt = FEEDBACK_ANALYSIS_PROMPT.format(
            original=original,
            decoded=decoded
        )
        response = await self.client.chat(
            messages=[
                {"role": "system", "content": "You are a precise analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response
    
    def _extract_focus_areas(self, feedback: str) -> str:
        """Extract key focus areas from feedback for next iteration."""
        # Simple extraction - could be more sophisticated
        lines = feedback.strip().split('\n')
        focus = [line.strip('- •1234567890.') for line in lines if line.strip()]
        return ', '.join(focus[:5])  # Top 5 focus areas
    
    def _parse_signal(self, response: str) -> MinimalSignal:
        """Parse LLM response into MinimalSignal."""
        try:
            data = json.loads(response)
            
            intent = data.get("intent", "QUERY").upper()
            if intent not in VALID_INTENTS:
                intent = "QUERY"
            
            priority = data.get("priority", "medium").lower()
            if priority not in VALID_PRIORITIES:
                priority = "medium"
            
            # Parse sections
            sections = []
            for section_data in data.get("sections", []):
                sections.append(ContentSection(
                    title=section_data.get("title", ""),
                    content=section_data.get("content", ""),
                    importance=section_data.get("importance", "medium")
                ))
            
            # Coerce state to dict if it's a string
            state = data.get("state", {})
            if isinstance(state, str):
                state = {"status": state}
            elif not isinstance(state, dict):
                state = {}
            
            # Coerce constraints to list if it's not
            constraints = data.get("constraints", [])
            if isinstance(constraints, dict):
                constraints = [f"{k}: {v}" for k, v in constraints.items()]
            elif not isinstance(constraints, list):
                constraints = []
            
            # Determine strategy
            if len(sections) == 0:
                strategy = "compact"
            elif len(sections) <= 5:
                strategy = "detailed"
            else:
                strategy = "chunked"
            
            return MinimalSignal(
                version="2.0",
                intent=intent,
                target=data.get("target", ""),
                summary=data.get("summary", {}),
                sections=sections,
                constraints=constraints,
                state=state,
                priority=priority,
                encoding_strategy=strategy,
                total_sections=len(sections)
            )
        except json.JSONDecodeError as e:
            raise EncoderError(f"Failed to parse LLM response: {e}")
