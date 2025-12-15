# Minimal Signaling LLM Agents

**Mediated Minimal-Signaling Architecture for LLM Agent Communication**

A research prototype exploring the question: *"What is the minimal information two LLM agents must exchange to stay aligned on a shared task?"*

## Research Motivation

Multi-agent LLM systems currently communicate via unrestricted natural language, which is:
- **Verbose** - expensive and eats context window
- **Ambiguous** - cross-model interpretation drift
- **Unstable** - shared-belief drift over multiple turns

This project enforces a communication bottleneck through a mediated pipeline:

```
Agent A → Stage 1 (Compression) → Stage 2 (Semantic Keys) → [Judge] → Agent B
```

## Architecture

### Stage 1: Learned Compression
- Recursive re-compression until token budget is met
- Configurable budget and recursion limits
- DistilBART-based summarization (swappable)

### Stage 2: Semantic Key Extraction
- Converts compressed text to structured schema
- Stable symbolic representation across LLMs
- Keys like: `INSTRUCTION: update`, `STATE: confidence=high`

### Optional: Judge/Verification Layer
- Validates semantic keys against original intent
- Pass/fail scoring with issue detection

## Quick Start

```bash
# Install dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Run the demo (coming soon)
poetry run python -m minimal_signaling.demo
```

## Project Structure

```
├── src/minimal_signaling/    # Core library
├── config/                   # Configuration files
├── tests/                    # Test suite
├── traces/                   # Run traces (JSONL)
└── notebooks/                # Experimentation
```

## Configuration

See `config/default.yaml` for all options:
- Token budgets
- Recursion limits
- Stage enable/disable
- Judge toggle

## Development

```bash
# Code quality
poetry run ruff check .
poetry run mypy src/

# Run tests with coverage
poetry run pytest --cov
```

## License

MIT
