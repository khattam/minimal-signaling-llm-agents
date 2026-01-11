"""MSP Decoder - translates MinimalSignal back to natural language."""

from .groq_client import GroqClient
from .protocol import MinimalSignal, DecoderError


DECODER_SYSTEM_PROMPT = """You are a semantic decoder. Convert the structured signal into clear natural language.

The output should:
- Be a complete, coherent message
- Include all information from the signal
- Match the requested style: {style}

Style guidelines:
- professional: Clear, formal business language
- casual: Friendly, conversational tone
- technical: Precise, detailed technical language

Do not add information not present in the signal.
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
