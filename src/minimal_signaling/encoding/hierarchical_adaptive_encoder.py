"""Hierarchical Adaptive Encoder - Enhanced iterative encoding for long messages.

NOVEL CONTRIBUTIONS:
1. Hierarchical Section Decomposition - breaks long messages into nested subsections
2. Importance-Weighted Preservation - critical sections preserved with more detail
3. Adaptive Compression Depth - compress based on importance scores
4. Multi-pass Refinement - structure extraction + detail filling based on feedback

This extends the working two-tier + iterative feedback architecture to scale
to longer messages (>1500 tokens) while maintaining 80%+ semantic similarity.
"""

import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from ..groq_client import GroqClient
from ..protocol import MinimalSignal, ContentSection, EncoderError, VALID_INTENTS, VALID_PRIORITIES
from ..semantic_judge import SemanticJudge
from ..msp_decoder import MSPDecoder
from ..tokenization import TiktokenTokenizer


@dataclass
class SectionImportance:
    """Importance analysis for a section."""
    title: str
    importance: str  # critical, high, medium, low
    key_concepts: List[str]
    detail_level: float  # 0.0-1.0, how much detail to preserve


@dataclass
class HierarchicalRefinementStep:
    """Record of one refinement iteration with importance tracking."""
    iteration: int
    signal: MinimalSignal
    decoded_text: str
    similarity_score: float
    section_importances: List[SectionImportance]
    feedback: Optional[str]
    signal_tokens: int


@dataclass
class HierarchicalEncodingResult:
    """Result of hierarchical encoding process."""
    original_text: str
    original_tokens: int
    final_signal: MinimalSignal
    final_decoded: str
    final_similarity: float
    iterations: int
    refinement_history: List[HierarchicalRefinementStep]
    converged: bool
    signal_tokens: int


# First pass: Extract structure and identify sections
STRUCTURE_EXTRACTION_PROMPT = """You are analyzing a long message to identify its logical structure.

Your task: Identify the main sections/topics in this message and assess their importance.

Output a JSON object with:
- sections: Array of sections found, each with:
  - title: Section name (e.g., "Authentication Migration", "Budget Planning")
  - importance: critical|high|medium|low (based on urgency, impact, action items)
  - key_concepts: Array of 3-5 key concepts/entities in this section
  - summary: One sentence describing what this section covers

Importance guidelines:
- critical: Urgent issues, blockers, security problems, immediate action items
- high: Important updates, significant decisions needed, major milestones
- medium: Regular updates, context information, background details
- low: Nice-to-know information, tangential details

Output ONLY valid JSON, no explanation."""


# Second pass: Encode with importance-weighted detail preservation
HIERARCHICAL_ENCODING_PROMPT = """You are a semantic encoder using hierarchical compression.

IMPORTANCE ANALYSIS:
{importance_analysis}

ENCODING STRATEGY:
- For CRITICAL sections: Preserve ALL details, numbers, names, technical terms
- For HIGH sections: Preserve key details and specific information
- For MEDIUM sections: Preserve main points and important specifics
- For LOW sections: Summarize main idea concisely

Output a JSON object with:
- intent: One of [ANALYZE, GENERATE, EVALUATE, TRANSFORM, QUERY, RESPOND, DELEGATE, REPORT]
- target: What the message is about (concise)
- summary: High-level overview with key metrics/counts (object)
- sections: Array of content sections, each with:
  - title: Section name (from importance analysis)
  - content: Detailed content (detail level based on importance)
  - importance: critical|high|medium|low (from analysis)
- constraints: All constraints, deadlines, action items (array)
- state: Current state information (object)
- priority: One of [low, medium, high, critical]

CRITICAL: For sections marked critical/high, include COMPLETE information.
For medium/low sections, you can be more concise.

Output ONLY valid JSON, no explanation."""


# Refinement pass: Address missing information
REFINEMENT_WITH_IMPORTANCE_PROMPT = """You are re-encoding a message. Previous attempt lost information.

FEEDBACK FROM JUDGE:
{feedback}

IMPORTANCE ANALYSIS:
{importance_analysis}

MISSING CONCEPTS (must be added):
{missing_concepts}

Re-encode the message, focusing on:
1. Adding the missing concepts identified above
2. Preserving detail based on section importance
3. Ensuring critical/high importance sections are complete

Output a JSON object with:
- intent: One of [ANALYZE, GENERATE, EVALUATE, TRANSFORM, QUERY, RESPOND, DELEGATE, REPORT]
- target: What the message is about
- summary: High-level overview (object)
- sections: Array of sections with:
  - title: Section name
  - content: FULL content (especially for critical/high importance)
  - importance: critical|high|medium|low
- constraints: All constraints and deadlines
- state: Current state
- priority: One of [low, medium, high, critical]

CRITICAL: Address ALL missing concepts. For critical sections, preserve EVERYTHING.

Output ONLY valid JSON, no explanation."""


class HierarchicalAdaptiveEncoder:
    """Enhanced encoder with hierarchical decomposition and importance weighting.
    
    Novel approach:
    1. First pass: Analyze structure and assess section importance
    2. Second pass: Encode with importance-weighted detail preservation
    3. Iterative refinement: Use judge feedback to identify and add missing critical info
    """
    
    def __init__(
        self,
        groq_client: GroqClient,
        judge: SemanticJudge,
        decoder: MSPDecoder,
        max_iterations: int = 5,
        target_similarity: float = 0.80
    ):
        self.client = groq_client
        self.judge = judge
        self.decoder = decoder
        self.tokenizer = TiktokenTokenizer()
        self.max_iterations = max_iterations
        self.target_similarity = target_similarity
    
    async def encode_with_refinement(
        self,
        natural_language: str,
        style: str = "professional"
    ) -> HierarchicalEncodingResult:
        """Encode long message with hierarchical adaptive compression.
        
        Args:
            natural_language: Input text to encode
            style: Decoding style for verification
            
        Returns:
            HierarchicalEncodingResult with full refinement history
        """
        if not natural_language or not natural_language.strip():
            raise EncoderError("Input cannot be empty")
        
        print(f"\n{'='*80}")
        print(f"HIERARCHICAL ADAPTIVE ENCODING")
        print(f"{'='*80}")
        
        original_tokens = self.tokenizer.count_tokens(natural_language)
        print(f"\n📝 Original: {original_tokens} tokens")
        
        # PASS 1: Analyze structure and importance
        print(f"\n🔍 PASS 1: Analyzing structure and importance...")
        importance_analysis = await self._analyze_structure(natural_language)
        section_importances = self._parse_importance_analysis(importance_analysis)
        
        print(f"   Found {len(section_importances)} sections:")
        for sec in section_importances:
            print(f"   - {sec.title}: {sec.importance} importance")
        
        refinement_history: List[HierarchicalRefinementStep] = []
        
        # PASS 2: Initial encoding with importance weighting
        print(f"\n🗜️  PASS 2: Encoding with importance-weighted compression...")
        signal = await self._encode_hierarchical(natural_language, importance_analysis)
        signal_tokens = self.tokenizer.count_tokens(signal.model_dump_json())
        
        print(f"   Signal: {signal_tokens} tokens ({signal_tokens/original_tokens:.1%} of original)")
        
        # Decode and judge
        decoded = await self.decoder.decode(signal, style)
        judge_result = self.judge.evaluate(natural_language, decoded)
        
        print(f"   Similarity: {judge_result.similarity_score:.1%}")
        
        # Record first iteration
        step = HierarchicalRefinementStep(
            iteration=1,
            signal=signal,
            decoded_text=decoded,
            similarity_score=judge_result.similarity_score,
            section_importances=section_importances,
            feedback=None,
            signal_tokens=signal_tokens
        )
        refinement_history.append(step)
        
        # Check if we already hit target
        if judge_result.similarity_score >= self.target_similarity:
            print(f"\n✅ SUCCESS in 1 iteration!")
            return HierarchicalEncodingResult(
                original_text=natural_language,
                original_tokens=original_tokens,
                final_signal=signal,
                final_decoded=decoded,
                final_similarity=judge_result.similarity_score,
                iterations=1,
                refinement_history=refinement_history,
                converged=True,
                signal_tokens=signal_tokens
            )
        
        # ITERATIVE REFINEMENT
        for iteration in range(2, self.max_iterations + 1):
            print(f"\n{'─'*80}")
            print(f"ITERATION {iteration}")
            print(f"{'─'*80}")
            
            # Analyze what's missing
            print(f"🔍 Analyzing missing information...")
            feedback = await self._analyze_loss(natural_language, decoded)
            missing_concepts = self._extract_missing_concepts(feedback, section_importances)
            
            print(f"   Missing {len(missing_concepts)} key concepts")
            
            # Re-encode with feedback
            print(f"🔧 Re-encoding with focus on missing information...")
            signal = await self._encode_with_feedback(
                natural_language,
                importance_analysis,
                feedback,
                missing_concepts
            )
            signal_tokens = self.tokenizer.count_tokens(signal.model_dump_json())
            
            print(f"   Signal: {signal_tokens} tokens ({signal_tokens/original_tokens:.1%} of original)")
            
            # Decode and judge
            decoded = await self.decoder.decode(signal, style)
            judge_result = self.judge.evaluate(natural_language, decoded)
            
            print(f"   Similarity: {judge_result.similarity_score:.1%}")
            
            # Record iteration
            step = HierarchicalRefinementStep(
                iteration=iteration,
                signal=signal,
                decoded_text=decoded,
                similarity_score=judge_result.similarity_score,
                section_importances=section_importances,
                feedback=feedback,
                signal_tokens=signal_tokens
            )
            refinement_history.append(step)
            
            # Check convergence
            if judge_result.similarity_score >= self.target_similarity:
                print(f"\n✅ SUCCESS in {iteration} iterations!")
                return HierarchicalEncodingResult(
                    original_text=natural_language,
                    original_tokens=original_tokens,
                    final_signal=signal,
                    final_decoded=decoded,
                    final_similarity=judge_result.similarity_score,
                    iterations=iteration,
                    refinement_history=refinement_history,
                    converged=True,
                    signal_tokens=signal_tokens
                )
        
        # Max iterations reached
        final_step = refinement_history[-1]
        print(f"\n⚠️  Max iterations reached. Final similarity: {final_step.similarity_score:.1%}")
        
        return HierarchicalEncodingResult(
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
    
    async def _analyze_structure(self, text: str) -> str:
        """First pass: Analyze structure and assess importance."""
        response = await self.client.chat(
            messages=[
                {"role": "system", "content": STRUCTURE_EXTRACTION_PROMPT},
                {"role": "user", "content": text}
            ],
            json_mode=True,
            temperature=0.0
        )
        return response
    
    def _parse_importance_analysis(self, analysis_json: str) -> List[SectionImportance]:
        """Parse importance analysis into structured format."""
        data = json.loads(analysis_json)
        sections = []
        
        for sec in data.get("sections", []):
            importance = sec.get("importance", "medium")
            
            # Map importance to detail level
            detail_map = {
                "critical": 1.0,  # Preserve everything
                "high": 0.8,      # Preserve most details
                "medium": 0.5,    # Preserve key points
                "low": 0.3        # Summarize
            }
            
            sections.append(SectionImportance(
                title=sec.get("title", ""),
                importance=importance,
                key_concepts=sec.get("key_concepts", []),
                detail_level=detail_map.get(importance, 0.5)
            ))
        
        return sections
    
    async def _encode_hierarchical(self, text: str, importance_analysis: str) -> MinimalSignal:
        """Second pass: Encode with importance-weighted compression."""
        prompt = HIERARCHICAL_ENCODING_PROMPT.format(
            importance_analysis=importance_analysis
        )
        
        response = await self.client.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            json_mode=True,
            temperature=0.0
        )
        
        return self._parse_signal(response)
    
    async def _encode_with_feedback(
        self,
        text: str,
        importance_analysis: str,
        feedback: str,
        missing_concepts: List[str]
    ) -> MinimalSignal:
        """Refinement pass: Re-encode addressing missing information."""
        prompt = REFINEMENT_WITH_IMPORTANCE_PROMPT.format(
            feedback=feedback,
            importance_analysis=importance_analysis,
            missing_concepts="\n".join(f"- {c}" for c in missing_concepts)
        )
        
        response = await self.client.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            json_mode=True,
            temperature=0.1
        )
        
        return self._parse_signal(response)
    
    async def _analyze_loss(self, original: str, decoded: str) -> str:
        """Analyze what information was lost in decoding."""
        prompt = f"""Compare original and decoded messages. Identify what's MISSING or DISTORTED.

Original:
{original}

Decoded:
{decoded}

List specific missing information:
1. Missing facts, numbers, names
2. Missing action items or deadlines
3. Missing technical details
4. Distorted or oversimplified information

Be specific and concise."""
        
        response = await self.client.chat(
            messages=[
                {"role": "system", "content": "You are a precise analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response
    
    def _extract_missing_concepts(
        self,
        feedback: str,
        section_importances: List[SectionImportance]
    ) -> List[str]:
        """Extract key missing concepts from feedback."""
        # Parse feedback lines
        lines = feedback.strip().split('\n')
        concepts = []
        
        for line in lines:
            line = line.strip('- •1234567890.')
            if line and len(line) > 10:  # Skip empty or very short lines
                concepts.append(line)
        
        # Prioritize concepts from critical/high importance sections
        critical_keywords = []
        for sec in section_importances:
            if sec.importance in ["critical", "high"]:
                critical_keywords.extend(sec.key_concepts)
        
        # Sort concepts by relevance to critical sections
        def relevance_score(concept: str) -> int:
            return sum(1 for kw in critical_keywords if kw.lower() in concept.lower())
        
        concepts.sort(key=relevance_score, reverse=True)
        
        return concepts[:10]  # Top 10 missing concepts
    
    def _parse_signal(self, response: str) -> MinimalSignal:
        """Parse LLM response into MinimalSignal."""
        try:
            data = json.loads(response)
            
            intent = data.get("intent", "REPORT").upper()
            if intent not in VALID_INTENTS:
                intent = "REPORT"
            
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
            
            # Coerce state to dict
            state = data.get("state", {})
            if not isinstance(state, dict):
                state = {"status": str(state)} if state else {}
            
            # Coerce constraints to list
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
