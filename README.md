# Minimal Signaling LLM Agents

A research prototype for studying minimal information exchange between LLM agents via a mediated communication pipeline.

## Overview

This project implements a **Mediated Minimal-Signaling Architecture** that enforces communication bottlenecks between LLM agents. The system transforms verbose natural language messages into compact, structured representations through a two-stage pipeline:

```
Agent A → [Compression] → [Semantic Keys] → [Judge] → Agent B
```

### Key Features

- **Stage 1: Recursive Compression** - Reduces messages to fit a configurable token budget using DistilBART summarization
- **Stage 2: Semantic Key Extraction** - Converts compressed text to structured symbolic keys (INSTRUCTION, STATE, GOAL, CONTEXT, CONSTRAINT)
- **Optional Judge** - Verifies semantic key fidelity against the original message
- **Real-time Dashboard** - WebSocket-based visualization of pipeline execution
- **Comprehensive Tracing** - JSONL trace logs for analysis and debugging

## Installation

```bash
# Clone the repository
git clone https://github.com/khattam/minimal-signaling-llm-agents.git
cd minimal-signaling-llm-agents

# Install with Poetry
poetry install
```

## Quick Start

### Run the Demo

```bash
# Run with mock compressor (fast)
poetry run minimal-signaling demo

# Run with real DistilBART compressor (slower, more realistic)
poetry run minimal-signaling demo --real-compressor
```

### Start the Dashboard

```bash
# Start the web dashboard at http://localhost:8080
poetry run minimal-signaling serve

# With custom port
poetry run minimal-signaling serve --port 3000
```

### Process a Single Message

```bash
# Process and output JSON result
poetry run minimal-signaling process "Your message here"

# Save to file
poetry run minimal-signaling process "Your message" --output result.json
```

## Configuration

Create a `config.yaml` file:

```yaml
mediator:
  compression:
    enabled: true
    token_budget: 50
    max_recursion: 5
    model: "sshleifer/distilbart-cnn-12-6"
  
  semantic_keys:
    enabled: true
    schema_version: "1.0"
    extractor: "placeholder"
  
  judge:
    enabled: true

logging:
  level: "INFO"
  trace_dir: "traces"
```

Use with:
```bash
poetry run minimal-signaling demo --config config.yaml
```

## Architecture

### Pipeline Flow

1. **Message Received** - Agent A sends a natural language message
2. **Compression** - Recursive summarization until token budget is met
3. **Extraction** - Parse compressed text into semantic keys
4. **Judge** (optional) - Verify key fidelity
5. **Delivery** - Semantic keys sent to Agent B

### Semantic Key Types

| Type | Description |
|------|-------------|
| `INSTRUCTION` | Actions to perform |
| `STATE` | Current system/context state |
| `GOAL` | Desired outcomes |
| `CONTEXT` | Background information |
| `CONSTRAINT` | Limitations or requirements |

### Example Output

```
Input: "INSTRUCTION: Analyze sales data. STATE: Data collected. GOAL: Find trends."
Tokens: 45 → 12 (73% compression)

Extracted Keys:
  [INSTRUCTION] Analyze sales data
  [STATE] Data collected
  [GOAL] Find trends

Judge: PASSED (confidence: 85%)
```

## Development

### Run Tests

```bash
# Run all property-based tests
poetry run pytest tests/property/ -v

# Run with coverage
poetry run pytest tests/property/ --cov=minimal_signaling
```

### Project Structure

```
src/minimal_signaling/
├── models.py        # Pydantic data models
├── interfaces.py    # Abstract base classes
├── config.py        # Configuration management
├── tokenization.py  # Token counting (tiktoken)
├── compression.py   # Stage 1: DistilBART compression
├── extraction.py    # Stage 2: Semantic key extraction
├── judge.py         # Optional verification layer
├── mediator.py      # Pipeline orchestration
├── events.py        # Real-time event system
├── trace.py         # JSONL trace logging
├── websocket.py     # WebSocket server
├── server.py        # FastAPI dashboard
└── cli.py           # Command-line interface
```

## Research Context

This prototype supports research into:
- **Minimal information exchange** - What's the least information agents need to stay aligned?
- **Communication bottlenecks** - How do constraints affect coordination quality?
- **Semantic stability** - Can structured keys reduce cross-model interpretation drift?

See the thesis document for detailed research motivation and methodology.

## License

MIT
