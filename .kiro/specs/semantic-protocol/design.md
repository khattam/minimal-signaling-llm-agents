# Design Document: Minimal Signaling Protocol (MSP)

## Overview

The Minimal Signaling Protocol (MSP) provides a structured communication layer for multi-agent LLM systems. Instead of compressing natural language, MSP translates verbose agent messages into a compact, human-readable JSON schema. This approach ensures:

1. **Compactness by design** - structured fields are inherently smaller than prose
2. **Human traceability** - every field is readable and auditable
3. **Cross-model compatibility** - JSON is model-agnostic
4. **Semantic preservation** - verified through embedding similarity

The system uses Groq's free API tier for LLM inference and sentence-transformers for local semantic verification.

## Minimality Definition

MSP defines "minimality" along two dimensions:

1. **Syntactic Minimality**: Token count of MSP JSON vs original natural language
   - Target: compression_ratio < 0.5 (MSP uses less than half the tokens)
   
2. **Semantic Adequacy**: Preservation of task-critical information
   - Measured by: embedding similarity ≥ 0.80 AND presence of intent/target in decoded text

This is distinct from:
- **MCP/ANP**: Focus on tool integration and network interoperability, not message compression
- **Emergent protocols**: Optimize for latent efficiency but lose human readability
- **FIPA-ACL/KQML**: Formal agent communication languages, heavier than MSP

MSP is a **per-message protocol for intra-system communication**, not a network interoperability layer.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MEDIATOR SERVICE                                │
│                                                                             │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐               │
│  │   ENCODER    │    │   MSP SIGNAL    │    │   DECODER    │               │
│  │  (Groq LLM)  │───▶│   (JSON Schema) │───▶│  (Groq LLM)  │               │
│  │              │    │                 │    │              │               │
│  │ NL → JSON    │    │ {intent, target,│    │ JSON → NL    │               │
│  └──────────────┘    │  params, state} │    └──────────────┘               │
│         │            └────────┬────────┘           │                       │
│         │                     │                    │                       │
│         ▼                     ▼                    ▼                       │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │                    SEMANTIC JUDGE                            │          │
│  │              (sentence-transformers, local)                  │          │
│  │                                                              │          │
│  │  Original NL ←──── Embedding Similarity ────→ Decoded NL    │          │
│  │                         Score: 0.0 - 1.0                     │          │
│  └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │                    METRICS & TRACING                         │          │
│  │  • Token counts (original, MSP, decoded)                     │          │
│  │  • Compression ratio                                         │          │
│  │  • Latency measurements                                      │          │
│  │  • Full audit trail with trace IDs                          │          │
│  └─────────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. MSP Schema (`protocol.py`)

The core data structure for minimal signals:

```python
class MinimalSignal(BaseModel):
    """Structured representation of agent communication."""
    
    model_config = {"extra": "forbid"}  # Strict validation
    
    # Protocol version for evolution
    version: str = "1.0"
    
    # What action is being requested/performed
    intent: Literal["ANALYZE", "GENERATE", "EVALUATE", "TRANSFORM", "QUERY", "RESPOND", "DELEGATE", "REPORT"]
    
    # What the action targets
    target: str
    
    # Key parameters for the task
    params: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution constraints
    constraints: List[str] = Field(default_factory=list)
    
    # Current state information
    state: Dict[str, Any] = Field(default_factory=dict)
    
    # Priority level
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    
    # Tracing metadata
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    parent_id: Optional[str] = None
```

### 2. Groq Client (`groq_client.py`)

Wrapper for Groq API with rate limiting:

```python
class GroqClient:
    """Client for Groq API with rate limiting for free tier."""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.rate_limiter = RateLimiter(requests_per_minute=30)
    
    async def chat(self, messages: List[Dict], json_mode: bool = False) -> str:
        await self.rate_limiter.acquire()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"} if json_mode else None
        )
        return response.choices[0].message.content
```

### 3. Encoder (`encoder.py`)

Translates natural language to MSP:

```python
class MSPEncoder:
    """Encodes natural language into Minimal Signal Protocol format."""
    
    SYSTEM_PROMPT = """You are a semantic encoder. Extract structured information from the input message.
    
    Output a JSON object with these fields:
    - intent: One of [ANALYZE, GENERATE, EVALUATE, TRANSFORM, QUERY, RESPOND, DELEGATE, REPORT]
    - target: What the action is about (string)
    - params: Key parameters as key-value pairs (object)
    - constraints: List of constraints/requirements (array of strings)
    - state: Current state information (object)
    - priority: One of [low, medium, high, critical]
    
    Be concise. Extract only essential information."""
    
    def __init__(self, groq_client: GroqClient):
        self.client = groq_client
    
    async def encode(self, natural_language: str) -> MinimalSignal:
        response = await self.client.chat(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": natural_language}
            ],
            json_mode=True
        )
        return MinimalSignal.model_validate_json(response)
```

### 4. Decoder (`decoder.py`)

Translates MSP back to natural language:

```python
class MSPDecoder:
    """Decodes Minimal Signal Protocol back to natural language."""
    
    SYSTEM_PROMPT = """You are a semantic decoder. Convert the structured signal into clear natural language.
    
    The output should:
    - Be a complete, coherent message
    - Include all information from the signal
    - Match the requested style: {style}
    
    Do not add information not present in the signal."""
    
    def __init__(self, groq_client: GroqClient):
        self.client = groq_client
    
    async def decode(self, signal: MinimalSignal, style: str = "professional") -> str:
        prompt = self.SYSTEM_PROMPT.format(style=style)
        response = await self.client.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": signal.model_dump_json()}
            ]
        )
        return response
```

### 5. Semantic Judge (`semantic_judge.py`)

Verifies semantic preservation using embeddings:

```python
class SemanticJudge:
    """Judges semantic fidelity using embedding similarity."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.80):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold  # Configurable via MSPConfig
    
    def evaluate(self, original: str, decoded: str) -> JudgeResult:
        # Compute embeddings
        emb_original = self.model.encode(original)
        emb_decoded = self.model.encode(decoded)
        
        # Cosine similarity
        similarity = cosine_similarity([emb_original], [emb_decoded])[0][0]
        
        return JudgeResult(
            passed=similarity >= self.threshold,
            confidence=float(similarity),
            similarity_score=float(similarity),
            issues=[] if similarity >= self.threshold else ["Semantic drift detected"]
        )
```

### 6. Pipeline Orchestrator (`pipeline.py`)

Coordinates the full encode → signal → decode → verify flow:

```python
class MSPPipeline:
    """Orchestrates the full MSP pipeline."""
    
    def __init__(self, encoder: MSPEncoder, decoder: MSPDecoder, judge: SemanticJudge):
        self.encoder = encoder
        self.decoder = decoder
        self.judge = judge
        self.tokenizer = TiktokenTokenizer()
    
    async def process(self, input_text: str, style: str = "professional") -> PipelineResult:
        start_time = time.time()
        
        # Encode NL → MSP
        signal = await self.encoder.encode(input_text)
        
        # Decode MSP → NL
        decoded = await self.decoder.decode(signal, style)
        
        # Judge semantic fidelity
        judge_result = self.judge.evaluate(input_text, decoded)
        
        # Compute metrics
        original_tokens = self.tokenizer.count_tokens(input_text)
        signal_tokens = self.tokenizer.count_tokens(signal.model_dump_json())
        decoded_tokens = self.tokenizer.count_tokens(decoded)
        
        return PipelineResult(
            original_text=input_text,
            signal=signal,
            decoded_text=decoded,
            judge=judge_result,
            metrics=PipelineMetrics(
                original_tokens=original_tokens,
                signal_tokens=signal_tokens,
                decoded_tokens=decoded_tokens,
                compression_ratio=signal_tokens / original_tokens,
                latency_ms=(time.time() - start_time) * 1000
            ),
            trace_id=signal.trace_id,
            timestamp=signal.timestamp
        )
```

## Data Models

### TiktokenTokenizer

```python
class TiktokenTokenizer:
    """Wrapper around tiktoken for consistent token counting."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.enc = tiktoken.encoding_for_model(model_name)
    
    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))
```

### PipelineResult

```python
class PipelineResult(BaseModel):
    """Complete result from MSP pipeline processing."""
    original_text: str
    signal: MinimalSignal
    decoded_text: str
    judge: JudgeResult
    metrics: PipelineMetrics
    trace_id: str
    timestamp: datetime

class PipelineMetrics(BaseModel):
    """Metrics from pipeline execution."""
    original_tokens: int
    signal_tokens: int
    decoded_tokens: int
    compression_ratio: float
    latency_ms: float

class JudgeResult(BaseModel):
    """Result from semantic judge evaluation."""
    passed: bool
    confidence: float
    similarity_score: float
    issues: List[str]
```

### Configuration

```python
class MSPConfig(BaseModel):
    """Configuration for MSP system."""
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    judge_model: str = "all-MiniLM-L6-v2"
    judge_threshold: float = 0.80
    rate_limit_rpm: int = 30
    default_style: str = "professional"
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: MSP Schema Validity
*For any* MinimalSignal object, it SHALL contain all required fields (intent, target, params, constraints, state, priority, trace_id, timestamp), serialize to valid JSON, and deserialize back to an equivalent object.
**Validates: Requirements 1.1, 1.2, 1.4, 1.5**

### Property 2: Encoder Produces Valid MSP
*For any* non-empty natural language input, the Encoder SHALL produce a MinimalSignal object that passes schema validation.
**Validates: Requirements 2.1**

### Property 3: Decoder Produces Non-Empty Output
*For any* valid MinimalSignal object, the Decoder SHALL produce a non-empty string containing natural language.
**Validates: Requirements 3.1**

### Property 4: Decoder Preserves MSP Content
*For any* valid MinimalSignal object, the decoded natural language SHALL contain references to the intent and target fields from the signal.
**Validates: Requirements 3.5**

### Property 5: Judge Score Bounds
*For any* pair of text strings, the SemanticJudge SHALL return a similarity score in the range [0.0, 1.0], and SHALL set passed=False when score < threshold.
**Validates: Requirements 4.1, 4.3, 4.4**

### Property 6: Metrics Computation Correctness
*For any* PipelineResult, the compression_ratio SHALL equal signal_tokens / original_tokens, and all token counts SHALL be positive integers.
**Validates: Requirements 5.1, 5.2, 5.4**

### Property 7: Round-Trip Semantic Preservation
*For any* natural language input processed through the full pipeline (encode → decode), the semantic similarity between original and decoded text SHALL exceed the configured threshold for well-formed agent messages.
**Validates: Requirements 4.1, 3.5**

### Property 8: MSP JSON Round-Trip
*For any* MinimalSignal object, serializing to JSON and parsing back SHALL produce an equivalent object (round-trip identity).
**Validates: Requirements 1.2**

## Error Handling

### Encoder Errors
- Empty/whitespace input → `EncoderError("Input cannot be empty")`
- Groq API failure → `EncoderError("LLM service unavailable")` with retry logic
- Invalid JSON response → `EncoderError("Failed to parse LLM response")`
- Rate limit exceeded → `RateLimitError` with backoff

### Decoder Errors
- Invalid MSP object → `DecoderError("Invalid signal schema")`
- Groq API failure → `DecoderError("LLM service unavailable")` with retry logic
- Empty response → `DecoderError("Decoder produced empty output")`

### Judge Errors
- Embedding failure → `JudgeError("Failed to compute embeddings")`
- Model not loaded → `JudgeError("Sentence transformer model not available")`

### Error Class Hierarchy

```python
class MSPError(Exception):
    """Base exception for MSP errors."""
    pass

class EncoderError(MSPError):
    """Error during encoding NL → MSP."""
    pass

class DecoderError(MSPError):
    """Error during decoding MSP → NL."""
    pass

class JudgeError(MSPError):
    """Error during semantic evaluation."""
    pass

class RateLimitError(MSPError):
    """Rate limit exceeded for API calls."""
    pass
```

## Testing Strategy

### Property-Based Testing
- **Framework:** Hypothesis (Python)
- **Minimum iterations:** 100 per property
- **Generators:** Custom strategies for MinimalSignal, natural language text

### Unit Tests
- Schema validation edge cases
- Error handling paths
- Configuration loading

### Integration Tests
- Full pipeline round-trip with real Groq API
- Rate limiting behavior
- Concurrent request handling

## Dependencies

### Python Packages
```
groq>=0.4.0           # Groq API client
sentence-transformers>=2.2.0  # Local embeddings
tiktoken>=0.5.0       # Token counting
pydantic>=2.0.0       # Schema validation
httpx>=0.25.0         # Async HTTP (for rate limiting)
```

### External Services
- **Groq API** (free tier): https://console.groq.com
  - Rate limit: 30 requests/minute, 6000 tokens/minute
  - Models: llama-3.3-70b-versatile, mixtral-8x7b-32768

### Local Models (downloaded automatically)
- **sentence-transformers/all-MiniLM-L6-v2** (~80MB)
  - Used for semantic similarity computation
  - Runs entirely locally, no API needed
