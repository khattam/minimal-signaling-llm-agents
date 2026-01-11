"""MSP Encoder - translates natural language to MinimalSignal format."""

import json
from typing import Optional

from .groq_client import GroqClient
from .protocol import MinimalSignal, EncoderError, VALID_INTENTS, VALID_PRIORITIES


ENCODER_SYSTEM_PROMPT = """You are a semantic encoder. Extract structured information from the input message.

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
- The target should be a short description of what the action is about.

Output ONLY valid JSON, no explanation."""


class MSPEncoder:
    """Encodes natural language into Minimal Signal Protocol format."""
    
    def __init__(self, groq_client: GroqClient):
        """Initialize encoder with Groq client.
        
        Args:
            groq_client: Configured Groq client for LLM inference.
        """
        self.client = groq_client
    
    async def encode(self, natural_language: str) -> MinimalSignal:
        """Encode natural language to MSP format.
        
        Args:
            natural_language: Input text to encode.
            
        Returns:
            MinimalSignal object.
            
        Raises:
            EncoderError: If encoding fails.
        """
        if not natural_language or not natural_language.strip():
            raise EncoderError("Input cannot be empty")
        
        try:
            response = await self.client.chat(
                messages=[
                    {"role": "system", "content": ENCODER_SYSTEM_PROMPT},
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
            
            return MinimalSignal(
                intent=intent,
                target=data.get("target", ""),
                params=data.get("params", {}),
                constraints=data.get("constraints", []),
                state=data.get("state", {}),
                priority=priority
            )
            
        except json.JSONDecodeError as e:
            raise EncoderError(f"Failed to parse LLM response: {e}")
        except Exception as e:
            raise EncoderError(f"Encoding failed: {e}")
    
    def encode_sync(self, natural_language: str) -> MinimalSignal:
        """Synchronous version of encode."""
        if not natural_language or not natural_language.strip():
            raise EncoderError("Input cannot be empty")
        
        try:
            response = self.client.chat_sync(
                messages=[
                    {"role": "system", "content": ENCODER_SYSTEM_PROMPT},
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
            
            return MinimalSignal(
                intent=intent,
                target=data.get("target", ""),
                params=data.get("params", {}),
                constraints=data.get("constraints", []),
                state=data.get("state", {}),
                priority=priority
            )
            
        except json.JSONDecodeError as e:
            raise EncoderError(f"Failed to parse LLM response: {e}")
        except Exception as e:
            raise EncoderError(f"Encoding failed: {e}")
