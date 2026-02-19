"""MSP Encoder - translates natural language to MinimalSignal format with adaptive two-tier encoding."""

import json
from typing import Optional

from .groq_client import GroqClient
from .protocol import MinimalSignal, ContentSection, EncoderError, VALID_INTENTS, VALID_PRIORITIES
from .tokenization import TiktokenTokenizer


# Compact strategy for short messages (<500 tokens)
ENCODER_COMPACT_PROMPT = """You are a semantic encoder. Extract structured information from the input message.

Output a JSON object with these fields:
- intent: One of [ANALYZE, GENERATE, EVALUATE, TRANSFORM, QUERY, RESPOND, DELEGATE, REPORT]
- target: What the action is about (string, be concise)
- summary: Key information as nested key-value pairs (object)
- constraints: List of constraints/requirements (array of strings)
- state: Current state information (object)
- priority: One of [low, medium, high, critical]

Output ONLY valid JSON, no explanation."""


# Detailed strategy for medium messages (500-1500 tokens)
ENCODER_DETAILED_PROMPT = """You are a semantic encoder. Extract structured information from the input message.

For this medium-length message, use a two-tier approach:

TIER 1 - Summary: High-level structured overview
TIER 2 - Sections: Detailed content organized by topic

Output a JSON object with these fields:
- intent: One of [ANALYZE, GENERATE, EVALUATE, TRANSFORM, QUERY, RESPOND, DELEGATE, REPORT]
- target: What the action is about
- summary: High-level structured overview (object with key metrics, counts, categories)
- sections: Array of detailed content sections, each with:
  - title: Section name (e.g., "Security Issues", "Performance Problems", "Required Actions")
  - content: Full detailed content for this section (preserve ALL details)
  - importance: critical|high|medium|low
- constraints: List of constraints/requirements
- state: Current state information
- priority: One of [low, medium, high, critical]

CRITICAL: In sections, preserve ALL specific details, numbers, names, technical terms.
Do NOT summarize in sections - include complete information.

Output ONLY valid JSON, no explanation."""


# Chunked strategy for long messages (>1500 tokens)
ENCODER_CHUNKED_PROMPT = """You are a semantic encoder. Extract structured information from this LONG message.

Use a two-tier approach with multiple detailed sections:

TIER 1 - Summary: High-level overview with key metrics
TIER 2 - Sections: Break content into logical sections, preserving ALL details

Output a JSON object with these fields:
- intent: One of [ANALYZE, GENERATE, EVALUATE, TRANSFORM, QUERY, RESPOND, DELEGATE, REPORT]
- target: What the action is about
- summary: High-level overview (object with counts, categories, key metrics)
- sections: Array of detailed sections (identify natural sections in the message):
  - title: Section name
  - content: COMPLETE detailed content for this section (do NOT summarize)
  - importance: critical|high|medium|low
- constraints: All constraints and deadlines
- state: Current state
- priority: One of [low, medium, high, critical]

STRATEGY FOR LONG MESSAGES:
1. Identify natural sections (e.g., different issue categories, action items, context)
2. Create a section for each with FULL details
3. Preserve specific numbers, names, technical terms, examples
4. Do NOT omit information - completeness over brevity

Output ONLY valid JSON, no explanation."""


class MSPEncoder:
    """Encodes natural language into Minimal Signal Protocol format with adaptive strategy."""
    
    def __init__(self, groq_client: GroqClient):
        """Initialize encoder with Groq client.
        
        Args:
            groq_client: Configured Groq client for LLM inference.
        """
        self.client = groq_client
        self.tokenizer = TiktokenTokenizer()
    
    def _select_strategy(self, token_count: int) -> tuple[str, str]:
        """Select encoding strategy based on message length.
        
        Returns:
            (strategy_name, prompt)
        """
        if token_count < 500:
            return ("compact", ENCODER_COMPACT_PROMPT)
        elif token_count < 1500:
            return ("detailed", ENCODER_DETAILED_PROMPT)
        else:
            return ("chunked", ENCODER_CHUNKED_PROMPT)
    
    async def encode(self, natural_language: str) -> MinimalSignal:
        """Encode natural language to MSP format with adaptive strategy.
        
        Args:
            natural_language: Input text to encode.
            
        Returns:
            MinimalSignal object.
            
        Raises:
            EncoderError: If encoding fails.
        """
        if not natural_language or not natural_language.strip():
            raise EncoderError("Input cannot be empty")
        
        # Determine strategy based on length
        token_count = self.tokenizer.count_tokens(natural_language)
        strategy, prompt = self._select_strategy(token_count)
        
        try:
            response = await self.client.chat(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": natural_language}
                ],
                json_mode=True,
                temperature=0.0
            )
            
            # Parse JSON response
            data = json.loads(response)
            
            # Validate and normalize intent
            intent = data.get("intent", "QUERY").upper()
            if intent not in VALID_INTENTS:
                intent = "QUERY"
            
            # Validate priority
            priority = data.get("priority", "medium").lower()
            if priority not in VALID_PRIORITIES:
                priority = "medium"
            
            # Parse sections if present
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
            
            return MinimalSignal(
                version="2.0",
                intent=intent,
                target=data.get("target", ""),
                summary=data.get("summary", {}),
                sections=sections,
                constraints=data.get("constraints", []),
                state=data.get("state", {}),
                priority=priority,
                encoding_strategy=strategy,
                total_sections=len(sections)
            )
            
        except json.JSONDecodeError as e:
            raise EncoderError(f"Failed to parse LLM response: {e}")
        except Exception as e:
            raise EncoderError(f"Encoding failed: {e}")
    
    def encode_sync(self, natural_language: str) -> MinimalSignal:
        """Synchronous version of encode."""
        if not natural_language or not natural_language.strip():
            raise EncoderError("Input cannot be empty")
        
        token_count = self.tokenizer.count_tokens(natural_language)
        strategy, prompt = self._select_strategy(token_count)
        
        try:
            response = self.client.chat_sync(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": natural_language}
                ],
                json_mode=True,
                temperature=0.0
            )
            
            data = json.loads(response)
            
            intent = data.get("intent", "QUERY").upper()
            if intent not in VALID_INTENTS:
                intent = "QUERY"
            
            priority = data.get("priority", "medium").lower()
            if priority not in VALID_PRIORITIES:
                priority = "medium"
            
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
            
            return MinimalSignal(
                version="2.0",
                intent=intent,
                target=data.get("target", ""),
                summary=data.get("summary", {}),
                sections=sections,
                constraints=data.get("constraints", []),
                state=data.get("state", {}),
                priority=priority,
                encoding_strategy=strategy,
                total_sections=len(sections)
            )
            
        except json.JSONDecodeError as e:
            raise EncoderError(f"Failed to parse LLM response: {e}")
        except Exception as e:
            raise EncoderError(f"Encoding failed: {e}")
