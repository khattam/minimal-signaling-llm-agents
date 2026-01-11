# Requirements Document

## Introduction

This specification defines a Minimal Signaling Protocol (MSP) for efficient, human-traceable communication between LLM agents. Instead of compressing natural language, the system translates agent messages into a structured JSON schema that is compact by design, unambiguous, and fully auditable. 

The protocol is designed to be LLM-backend agnostic. This implementation uses Groq's free tier for encoder/decoder inference (for convenience and speed) while keeping the semantic judge fully local via sentence-transformers. The architecture supports future migration to local inference (e.g., Ollama) without protocol changes.

## Glossary

- **MSP (Minimal Signal Protocol)**: A structured JSON schema for agent-to-agent communication
- **Encoder**: Component that translates natural language into MSP format
- **Decoder**: Component that translates MSP back into natural language
- **Semantic Fidelity**: Measure of how well the decoded message preserves the original meaning
- **Groq**: Free cloud LLM inference API with fast response times (free tier: 30 req/min)
- **Round-trip**: The process of encoding NL → MSP → decoding back to NL

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to define a structured protocol schema for agent communication, so that messages are compact and human-readable by design.

#### Acceptance Criteria

1. THE MSP_Schema SHALL define fields for intent, target, parameters, constraints, state, and metadata
2. THE MSP_Schema SHALL use JSON format with strict type validation via Pydantic
3. WHEN a message is encoded THEN the MSP_Schema SHALL produce valid JSON under 500 tokens for typical agent messages
4. THE MSP_Schema SHALL include a trace_id field for audit trail purposes
5. THE MSP_Schema SHALL support nested parameters for complex task specifications

### Requirement 2

**User Story:** As a researcher, I want an encoder that converts natural language to MSP format, so that verbose agent messages become structured signals.

#### Acceptance Criteria

1. WHEN natural language is provided THEN the Encoder SHALL produce a valid MSP JSON object
2. THE Encoder SHALL use Groq API with Llama-3.3-70B or Mixtral-8x7B models (free tier)
3. THE Encoder SHALL use JSON mode for guaranteed schema compliance
4. WHEN the input is empty or invalid THEN the Encoder SHALL return an error with descriptive message
5. THE Encoder SHALL complete encoding within 2 seconds for messages under 1000 tokens

### Requirement 3

**User Story:** As a researcher, I want a decoder that converts MSP back to natural language, so that receiving agents can process the signal in their native format.

#### Acceptance Criteria

1. WHEN an MSP object is provided THEN the Decoder SHALL produce coherent natural language
2. THE Decoder SHALL use Groq API with the same model as the encoder for consistency
3. THE Decoder SHALL support configurable output styles (formal, casual, technical)
4. WHEN the MSP object is malformed THEN the Decoder SHALL return an error with validation details
5. THE Decoder SHALL preserve all semantic content from the MSP fields in the output

### Requirement 4

**User Story:** As a researcher, I want a semantic fidelity judge that verifies round-trip accuracy, so that I can prove the protocol preserves meaning.

#### Acceptance Criteria

1. WHEN original and decoded texts are compared THEN the Judge SHALL compute a semantic similarity score
2. THE Judge SHALL use sentence-transformers embeddings (all-MiniLM-L6-v2) running locally
3. THE Judge SHALL return a score between 0.0 and 1.0 where 1.0 indicates perfect preservation
4. WHEN similarity score is below 0.80 THEN the Judge SHALL flag the round-trip as potentially lossy
5. THE Judge SHALL provide detailed comparison metrics including cosine similarity and token overlap

### Requirement 5

**User Story:** As a researcher, I want to measure compression efficiency, so that I can quantify the token savings of MSP vs natural language.

#### Acceptance Criteria

1. THE Metrics_Module SHALL compute token counts for original NL, MSP, and decoded NL
2. THE Metrics_Module SHALL calculate compression ratio as tokens(MSP) / tokens(original)
3. THE Metrics_Module SHALL track average compression across multiple messages
4. WHEN compression ratio exceeds 1.0 THEN the Metrics_Module SHALL flag expansion (MSP larger than original)
5. THE Metrics_Module SHALL use tiktoken for consistent token counting across comparisons

### Requirement 6

**User Story:** As a researcher, I want a visualization dashboard that shows the encoding/decoding pipeline, so that I can trace and debug agent communication.

#### Acceptance Criteria

1. WHEN a message is processed THEN the Dashboard SHALL display original NL, MSP JSON, and decoded NL side-by-side
2. THE Dashboard SHALL highlight which MSP fields were extracted from which parts of the input
3. THE Dashboard SHALL show real-time semantic fidelity scores
4. THE Dashboard SHALL display token counts and compression metrics
5. THE Dashboard SHALL support step-by-step pipeline replay for debugging

### Requirement 7

**User Story:** As a researcher, I want to run evaluation benchmarks, so that I can measure protocol effectiveness across diverse message types.

#### Acceptance Criteria

1. THE Evaluation_Harness SHALL support batch processing of test message datasets
2. THE Evaluation_Harness SHALL compute aggregate metrics (mean compression, mean fidelity, latency)
3. THE Evaluation_Harness SHALL generate comparison reports between MSP and baseline (raw NL)
4. WHEN running benchmarks THEN the Evaluation_Harness SHALL log all intermediate results for analysis
5. THE Evaluation_Harness SHALL support custom test datasets in JSON format

### Requirement 8

**User Story:** As a researcher, I want the system to work with free API tiers, so that I can run experiments without costs.

#### Acceptance Criteria

1. THE System SHALL use Groq API free tier for all LLM inference (encoder, decoder)
2. THE System SHALL use sentence-transformers for semantic similarity (runs locally, no API)
3. THE System SHALL use tiktoken for token counting (runs locally, no API)
4. WHEN Groq API key is missing THEN the System SHALL provide clear setup instructions
5. THE System SHALL support rate limiting to stay within Groq free tier (30 req/min, 6000 tokens/min)
6. THE System SHALL support model selection via configuration (llama-3.3-70b-versatile, mixtral-8x7b-32768)
