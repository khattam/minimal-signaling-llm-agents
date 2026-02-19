"""MSP Decoder - translates MinimalSignal back to natural language."""

from .groq_client import GroqClient
from .protocol import MinimalSignal, DecoderError


DECODER_SYSTEM_PROMPT = """You are a semantic decoder. Convert the structured signal into clear natural language.

The signal uses a two-tier architecture:
- TIER 1 (summary): High-level structured overview
- TIER 2 (sections): Detailed content sections

CRITICAL: You must reconstruct the complete message from both tiers.

Decoding strategy:
1. Start with the intent and target to establish context
2. If sections are present, expand EACH section fully with all details
3. Include information from summary for context
4. Preserve all constraints and state information
5. Match the requested style: {style}

For signals with sections:
- Organize output by section titles
- Expand each section's content completely
- Maintain logical flow between sections
- Do NOT summarize or omit details from sections

Style guidelines:
- professional: Clear, formal business language with complete details
- casual: Friendly, conversational tone but still comprehensive
- technical: Precise, detailed technical language with all specifics

Do not add information not present in the signal, but DO expand everything that is present.
Output ONLY the natural language message, no JSON or explanation."""


class MSPDecoder:
    """Decodes Minimal Signal Protocol back to natural language."""
    
    def __init__(self, groq_client: GroqClient):
        """Initialize decoder with Groq client.
        
        Args:
            groq_client: Configured Groq client for LLM inference.
        """
        self.client = groq_client
    
    async def decode(
        self,
        signal: MinimalSignal,
        style: str = "professional"
    ) -> str:
        """Decode MSP signal to natural language.
        
        Args:
            signal: MinimalSignal to decode.
            style: Output style (professional, casual, technical).
            
        Returns:
            Natural language string.
            
        Raises:
            DecoderError: If decoding fails.
        """
        try:
            prompt = DECODER_SYSTEM_PROMPT.format(style=style)
            
            response = await self.client.chat(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": signal.model_dump_json()}
                ],
                json_mode=False,
                temperature=0.0
            )
            
            if not response or not response.strip():
                raise DecoderError("Decoder produced empty output")
            
            return response.strip()
            
        except DecoderError:
            raise
        except Exception as e:
            raise DecoderError(f"Decoding failed: {e}")
    
    def decode_sync(
        self,
        signal: MinimalSignal,
        style: str = "professional"
    ) -> str:
        """Synchronous version of decode."""
        try:
            prompt = DECODER_SYSTEM_PROMPT.format(style=style)
            
            response = self.client.chat_sync(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": signal.model_dump_json()}
                ],
                json_mode=False,
                temperature=0.0
            )
            
            if not response or not response.strip():
                raise DecoderError("Decoder produced empty output")
            
            return response.strip()
            
        except DecoderError:
            raise
        except Exception as e:
            raise DecoderError(f"Decoding failed: {e}")
